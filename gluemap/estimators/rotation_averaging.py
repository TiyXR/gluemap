import logging

import networkx as nx
import numpy as np
import pyceres
import pycolmap
import pygluemap
import torch
from scipy.spatial.transform import Rotation

from gluemap.math.geometry import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from gluemap.utils.runtime_capacity import resolve_native_thread_count

logger = logging.getLogger(__name__)


def collect_relative_rotations_ministar(
    prediction_dict: dict,
) -> tuple[dict[tuple[int, int], torch.Tensor], dict[tuple[int, int], float]]:
    """
    Pick the best-scoring relative pose per (image_id1, image_id2) pair.

    Iterates over every ministar's edges and, for each ordered pair starting
    at the ministar center, keeps the highest-scoring 3x4 extrinsics tensor.

    Args:
        prediction_dict: Star inference output with ``"indexes"``,
            ``"pose_scores"`` and ``"extrinsics"`` entries.

    Returns:
        ``(poses_rel, poses_rel_scores)`` where ``poses_rel`` maps
        ``(idx1, idx2)`` to a 3x4 float64 tensor and ``poses_rel_scores``
        maps the same key to a scalar score.
    """
    poses_rel = {}
    poses_rel_scores = {}
    for idx_star in range(len(prediction_dict["indexes"])):
        scores = prediction_dict["pose_scores"][idx_star][0]
        idx1 = prediction_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()
        # # for idx in range(1, len(prediction_dict["indexes"][idx_star])):
        for idx in valid_j:
            idx2 = prediction_dict["indexes"][idx_star][idx]

            if idx1 == idx2:
                continue

            score = scores[idx].item()
            existing_score = poses_rel_scores.get((idx1, idx2))
            if existing_score is not None and existing_score > score:
                continue

            poses_rel[(idx1, idx2)] = (
                prediction_dict["extrinsics"][idx_star][0, idx, :3]
                .cpu()
                .to(torch.float64)
            )
            poses_rel_scores[(idx1, idx2)] = score

    return poses_rel, poses_rel_scores


def _mst_init_rotations(
    prediction_dict: dict,
    indexes: set[int],
) -> dict[int, np.ndarray]:
    """
    Initialize rotations by chaining relative rotations along a maximum
    spanning tree.

    Args:
        prediction_dict: Star inference output passed through to
            :func:`collect_relative_rotations_ministar`.
        indexes: Set of image ids that should appear in the output.

    Returns:
        Mapping ``image_id -> quaternion (w, x, y, z)`` as a length-4
        float64 numpy array, identity for any node disconnected in the MST.
    """
    poses_rel, poses_rel_scores = collect_relative_rotations_ministar(
        prediction_dict
    )

    # Build graph with score weights
    G = nx.Graph()
    G.add_nodes_from(indexes)
    for (i, j), score in poses_rel_scores.items():
        G.add_edge(i, j, weight=score)

    mst = nx.maximum_spanning_tree(G)

    # Initialize all to identity quaternion [w, x, y, z]
    rotations = {
        idx: np.array([1.0, 0, 0, 0], dtype=np.float64) for idx in indexes
    }

    # Pick root as the node with highest degree in MST
    if len(mst.nodes) == 0:
        return rotations
    root = max(mst.nodes, key=lambda n: mst.degree(n))

    # Iterative DFS to propagate rotations
    visited = {root}
    stack = [(root, iter(mst.neighbors(root)))]
    while stack:
        node, neighbors_iter = stack[-1]
        try:
            neighbor = next(neighbors_iter)
        except StopIteration:
            stack.pop()
            continue
        if neighbor in visited:
            continue
        visited.add(neighbor)

        # Get relative rotation R_ij: from node -> neighbor
        R_node = quaternion_to_rotation_matrix(rotations[node])
        if (node, neighbor) in poses_rel:
            R_rel = poses_rel[(node, neighbor)][:3, :3].numpy()
            # R_neighbor = R_rel @ R_node
            R_neighbor = R_rel @ R_node
        elif (neighbor, node) in poses_rel:
            R_rel = poses_rel[(neighbor, node)][:3, :3].numpy()
            # R_rel is neighbor->node, so R_node = R_rel @ R_neighbor
            # => R_neighbor = R_rel^T @ R_node
            R_neighbor = R_rel.T @ R_node
        else:
            R_neighbor = R_node

        rotations[neighbor] = rotation_matrix_to_quaternion(
            torch.from_numpy(R_neighbor)
        ).astype(np.float64)
        stack.append((neighbor, iter(mst.neighbors(neighbor))))

    return rotations


def rotation_averaging(
    prediction_dict: dict,
    init_rotations: dict[int, np.ndarray] | None = None,
    fixed_rotation_ids: set[int] | None = None,
) -> dict[int, np.ndarray]:
    """
    Solve global rotations from ministar relative rotations using Ceres.

    Builds a Ceres problem with a ``RotationGeodesicError`` per valid
    relative-rotation edge, optimizes quaternions on the quaternion
    manifold, then converts back to 3x3 rotation matrices.

    Args:
        prediction_dict: Star inference output with ``"indexes"``,
            ``"pose_scores"`` and ``"extrinsics"``.
        init_rotations: Optional initial rotations as 3x3 matrices keyed by
            image id. Missing ids fall back to identity. When ``None``, MST
            initialization via :func:`_mst_init_rotations` is used.
        fixed_rotation_ids: Image ids whose warm-start rotations define the
            already frozen fixed-anchor gauge and must remain constant.

    Returns:
        Mapping ``image_id -> 3x3 rotation matrix`` (float64 numpy array).
    """
    logger.info("Rotation averaging with ministar ...")

    # Collect all indexes
    indexes = set(
        [
            prediction_dict["indexes"][i][j]
            for i in range(len(prediction_dict["indexes"]))
            for j in range(len(prediction_dict["indexes"][i]))
        ]
    )

    if init_rotations is not None:
        # Use provided rotation matrices as initialization
        rotations = {
            idx: rotation_matrix_to_quaternion(init_rotations[idx]).astype(
                np.float64
            )
            if idx in init_rotations
            else np.array([1.0, 0, 0, 0], dtype=np.float64)
            for idx in indexes
        }
    else:
        # MST initialization
        rotations = _mst_init_rotations(prediction_dict, indexes)

    prob = pyceres.Problem()
    costs = []
    losses = []
    for idx_star in range(len(prediction_dict["indexes"])):
        scores = prediction_dict["pose_scores"][idx_star][0]
        idx1 = prediction_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()
        for idx in valid_j:
            idx2 = prediction_dict["indexes"][idx_star][idx]

            if idx1 == idx2:
                continue

            rotation_rel = rotation_matrix_to_quaternion(
                prediction_dict["extrinsics"][idx_star][0, idx, :3, :3]
            )
            cost = pygluemap.RotationGeodesicError(rotation_rel)
            costs.append(cost)

            loss_scaled = pyceres.LossFunction(
                {"name": "huber", "params": [1e-2], "magnitude": scores[idx]}
            )
            losses.append(loss_scaled)
            prob.add_residual_block(
                cost, loss_scaled, [rotations[idx1], rotations[idx2]]
            )

    for idx in rotations:
        if prob.has_parameter_block(rotations[idx]):
            prob.set_manifold(rotations[idx], pyceres.QuaternionManifold())
            if fixed_rotation_ids and idx in fixed_rotation_ids:
                prob.set_parameter_block_constant(rotations[idx])

    options = pyceres.SolverOptions()
    if len(rotations) < 200:
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False
    else:
        options.linear_solver_type = (
            pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        )
        options.minimizer_progress_to_stdout = True

    options.num_threads = resolve_native_thread_count()
    options.max_num_iterations = 200
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    logger.info(summary.BriefReport())

    # Now, we have the estimation result for the community center and the
    # overlapping images
    # We now estimate the rotation for all images
    # Convert all quaternion to rotation matrix
    for idx in rotations:
        rotations[idx] = quaternion_to_rotation_matrix(rotations[idx])

    return rotations


def rotation_averaging_pycolmap(
    prediction_dict: dict,
    max_rotation_error_deg: float = 5.0,
) -> dict[int, np.ndarray]:
    """
    Rotation averaging using pycolmap's L1+IRLS solver with MST initialization.

    Args:
        prediction_dict: Star inference output with ``"indexes"``,
            ``"pose_scores"`` and ``"extrinsics"``.
        max_rotation_error_deg: Inlier threshold passed to
            ``pycolmap.RotationEstimatorOptions``.

    Returns:
        Mapping ``image_id -> 3x3 rotation matrix`` (float64 numpy array);
        identity for images without an estimated pose.
    """
    logger.info("Rotation averaging with pycolmap ...")

    # Collect all image indexes
    indexes = set(
        prediction_dict["indexes"][i][j]
        for i in range(len(prediction_dict["indexes"]))
        for j in range(len(prediction_dict["indexes"][i]))
    )

    # Build a minimal Reconstruction: one dummy camera + one image per index
    reconstruction = pycolmap.Reconstruction()
    colmap_ids = {idx: idx + 1 for idx in indexes}
    camera = pycolmap.Camera(
        camera_id=1,
        model="SIMPLE_PINHOLE",
        width=1,
        height=1,
        params=[1.0, 0.0, 0.0],
    )
    reconstruction.add_camera_with_trivial_rig(camera)
    for idx in sorted(indexes):
        image = pycolmap.Image(image_id=colmap_ids[idx], camera_id=1)
        reconstruction.add_image_with_trivial_frame(image)

    # Build PoseGraph: collect best-scoring edge per (i, j) pair
    best_edges = {}  # (idx1, idx2) -> (score, star_idx, local_idx)
    for idx_star in range(len(prediction_dict["indexes"])):
        scores = prediction_dict["pose_scores"][idx_star][0]
        idx1 = prediction_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()
        for j in valid_j:
            idx2 = prediction_dict["indexes"][idx_star][j]
            if idx1 == idx2:
                continue
            score = scores[j].item()
            pair = (min(idx1, idx2), max(idx1, idx2))
            if pair not in best_edges or score > best_edges[pair][0]:
                best_edges[pair] = (score, idx_star, j, idx1, idx2)

    pose_graph = pycolmap.PoseGraph()
    for _pair, (score, idx_star, j, idx1, idx2) in best_edges.items():
        pose_3x4 = (
            prediction_dict["extrinsics"][idx_star][0, j]
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        R = pose_3x4[:3, :3]
        t = pose_3x4[:3, 3]
        quat = Rotation.from_matrix(R).as_quat()  # (x, y, z, w)
        rigid = pycolmap.Rigid3d(
            rotation=np.array([quat[0], quat[1], quat[2], quat[3]]),
            translation=t,
        )
        edge = pycolmap.PoseGraphEdge(cam2_from_cam1=rigid)
        edge.num_matches = max(int(score * 1000), 1)
        pose_graph.add_edge(colmap_ids[idx1], colmap_ids[idx2], edge)

    # Configure options
    options = pycolmap.RotationEstimatorOptions()
    options.max_rotation_error_deg = max_rotation_error_deg
    options.weight_type = pycolmap.RotationWeightType.GEMAN_MCCLURE

    # Solve
    success = pycolmap.run_rotation_averaging(
        options, pose_graph, reconstruction, []
    )
    logger.info(f"Rotation averaging {'succeeded' if success else 'failed'}")

    # Extract rotations from reconstruction frames
    rotations = {}
    for idx in indexes:
        frame = reconstruction.frames[colmap_ids[idx]]
        if frame.has_pose():
            rotations[idx] = np.array(
                frame.rig_from_world.rotation.matrix(), dtype=np.float64
            )
        else:
            rotations[idx] = np.eye(3, dtype=np.float64)

    return rotations
