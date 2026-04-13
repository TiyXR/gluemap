import logging
from collections import defaultdict

import numpy as np
import pyceres
import pygluemap
import torch

from gluemap.estimators.track_establishment import (
    TrackEstablishmentOptions,
    establish_tracks_from_tracks_dict,
)

logger = logging.getLogger(__name__)
MIN_TRI_ANGLE = 1  # Minimum angle (in degrees) for a triangle to be considered valid for scale estimation


def initialize_parameters(
    predictions_dict,
    global_rotations,
    global_centers,
    global_scales,
):
    num_ministar = len(predictions_dict["indexes"])
    if global_centers is None:
        global_centers = {
            idx: np.random.rand(3).astype(np.float64)
            for idx in global_rotations
        }
    else:
        for idx in global_rotations:
            if idx not in global_centers:
                global_centers[idx] = np.random.rand(3).astype(np.float64)
    if global_scales is None:
        global_scales = [
            np.ones((1,)).astype(np.float64) for i in range(num_ministar)
        ]
    elif isinstance(global_scales, dict):
        temp_scales = []
        for idx_star in range(num_ministar):
            if idx_star in global_scales:
                temp_scales.append(
                    np.ones((1,)).astype(np.float64) * global_scales[idx_star]
                )
            else:
                temp_scales.append(np.ones((1,)).astype(np.float64))
        global_scales = temp_scales

    return num_ministar, global_centers, global_scales


def add_star_edge_error(
    prob,
    predictions_dict,
    global_rotations,
    global_centers,
    global_scales,
    num_ministar,
    costs,
    losses,
):
    center = -1
    for idx_star in range(num_ministar):
        scores = predictions_dict["pose_scores"][idx_star][0]
        idx1 = predictions_dict["indexes"][idx_star][0]
        valid_j = torch.where(scores > 0.0)[0].tolist()

        # Now, only consider the scale of the center image
        for idx in valid_j:
            idx2 = predictions_dict["indexes"][idx_star][idx]

            if idx1 == idx2:
                continue

            if (
                predictions_dict["median_tri_angle"][idx_star][idx - 1].item()
                < MIN_TRI_ANGLE
            ):
                continue  # Skip low-confidence edges

            # s_i * (c_j - c_i) = -R_j^T * t_ij
            t_ij_rotated = (
                -global_rotations[idx2].T
                @ predictions_dict["extrinsics"][idx_star][0, idx, :3, 3:]
                .cpu()
                .numpy()
            )
            loss_scaled = pyceres.LossFunction(
                {"name": "huber", "params": [1e-2], "magnitude": scores[idx]}
            )

            cost = pygluemap.BATAPairwiseDirectionError(t_ij_rotated)
            prob.add_residual_block(
                cost,
                loss_scaled,
                [
                    global_centers[idx1],
                    global_centers[idx2],
                    global_scales[idx_star],
                ],
            )

            costs.append(cost)
            losses.append(loss_scaled)

            if center < 0:
                prob.set_parameter_block_constant(global_centers[idx1])
                prob.set_parameter_block_constant(global_scales[idx_star])
                center = idx1

    return center


def update_points3d(
    prob,
    predictions_dict,
    global_rotations,
    global_centers,
    global_scales,
    num_ministar,
):
    if (
        "world_points" in predictions_dict
        and len(predictions_dict["world_points"]) > 0
    ):
        # Rescale the world points
        for idx in range(num_ministar):
            idx_center = predictions_dict["indexes"][idx][0]
            predictions_dict["world_points"][idx] = (
                predictions_dict["world_points"][idx]
                @ global_rotations[idx_center]
                / global_scales[idx]
                + global_centers[idx_center]
            )
    # Since it is in camera frame, we only need to change the scale
    if (
        "cam_points" in predictions_dict
        and len(predictions_dict["cam_points"]) > 0
        and predictions_dict["cam_points"][0] != None
    ):
        for idx in range(num_ministar):
            idx_center = predictions_dict["indexes"][idx][0]
            predictions_dict["cam_points"][idx] = (
                predictions_dict["cam_points"][idx] / global_scales[idx]
            )

    # Rescale the points and extrinsics
    if (
        "points3d_virtual" in predictions_dict
        and len(predictions_dict["points3d_virtual"]) > 0
    ):
        # Rescale the virtual points
        for idx in range(num_ministar):
            predictions_dict["points3d_virtual"][idx] = (
                predictions_dict["points3d_virtual"][idx] / global_scales[idx]
            )

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        predictions_dict["extrinsics"][idx_star][:, :, :3, 3:] = (
            predictions_dict["extrinsics"][idx_star][:, :, :3, 3:]
            / global_scales[idx_star][0]
        )


def similarity_averaging(
    predictions_dict,
    global_rotations,
    global_centers=None,
    global_scales=None,
    max_num_iterations=50,
    fix_scales=False,
):
    logger.info("Performing similarity averaging...")
    num_ministar, global_centers, global_scales = initialize_parameters(
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
    )

    prob = pyceres.Problem()

    costs = []
    losses = []

    add_star_edge_error(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
        costs,
        losses,
    )

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        if (
            fix_scales
            or predictions_dict["median_tri_angle"][idx_star].max().item()
            < MIN_TRI_ANGLE
        ):
            prob.set_parameter_block_constant(global_scales[idx_star])
        else:
            prob.set_parameter_lower_bound(global_scales[idx_star], 0, 1e-5)
            prob.set_parameter_upper_bound(global_scales[idx_star], 0, 1e5)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.num_threads = 32
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = False

    logger.info("Solving the optimization problem...")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    logger.info(summary.BriefReport())

    update_points3d(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
    )

    return global_centers


def add_depth_error(
    prob,
    predictions_dict,
    global_rotations,
    global_centers,
    global_scales,
    num_ministar,
    costs,
    losses,
    points3D=None,
    keypoints_per_image=None,
    pts2d_idx_all=None,
    pts2d_idx_virtual_all=None,
    pts2d_idx_inv=None,
    pts2d_idx_virtual_inv=None,
    image_to_point3D=None,
):
    """
    Add depth consistency error terms to the optimization problem.

    Args:
        prob: pyceres.Problem instance
        predictions_dict: Dictionary containing tracks, extrinsics, etc.
        global_rotations: Dictionary of global rotation matrices
        global_centers: Dictionary of global camera centers
        global_scales: List of scale factors per mini-star
        num_ministar: Number of mini-stars
        costs: List to append cost functions to
        losses: List to append loss functions to
        points3D: Established 3D point tracks (from establish_tracks)
        keypoints_per_image: Keypoints per image (from establish_tracks)
        pts2d_idx_all: Point indices for real tracks (forward map)
        pts2d_idx_virtual_all: Point indices for virtual tracks (forward map)
        pts2d_idx_inv: Inverse map for real tracks (image_id -> list of (idx, i, j))
        pts2d_idx_virtual_inv: Inverse map for virtual tracks (image_id -> list of (idx, i, j))
        image_to_point3D: Dict[image_id, Dict[pt_idx, point3D_id]] mapping 2D to 3D points
    """
    if points3D is None or pts2d_idx_inv is None:
        return

    if (
        "cam_points" not in predictions_dict
        or len(predictions_dict["cam_points"]) == 0
        or predictions_dict["cam_points"][0] == None
    ):
        return

    num_constraints = 0
    num_same_cam_constraints = 0
    num_skipped_same_star = 0

    cam_points_rotated = {}
    for idx_star in range(num_ministar):
        cam_points_rotated[idx_star] = []
        for idx in range(predictions_dict["cam_points"][idx_star].shape[1]):
            idx_inner = predictions_dict["indexes"][idx_star][idx]
            # Pre-rotate camera points to world frame (R^T * X_cam)
            cam_points_rotated[idx_star].append(
                predictions_dict["cam_points"][idx_star][0, idx]
                @ global_rotations[idx_inner]
            )

    # Iterate through all established tracks
    for point3D_id, point3D in points3D.items():
        elements = list(point3D.track.elements)

        if len(elements) < 2:
            continue

        # For each pair of observations in the track
        for i in range(len(elements)):
            elem1 = elements[i]
            image_id1, pt_idx1 = elem1.image_id, elem1.point2D_idx

            # Get original position in tracks_dict via inverse map
            if image_id1 not in pts2d_idx_inv:
                continue
            if pt_idx1 >= len(pts2d_idx_inv[image_id1]):
                continue  # This might be a virtual point
            idx1, pos1, j1 = pts2d_idx_inv[image_id1][pt_idx1]

            # Get 3D camera point (rotated to world frame)
            cam_point1 = (
                cam_points_rotated[idx1][pos1][j1].numpy().astype(np.float64)
            )

            for k in range(i + 1, len(elements)):
                elem2 = elements[k]
                image_id2, pt_idx2 = elem2.image_id, elem2.point2D_idx

                if image_id2 not in pts2d_idx_inv:
                    continue
                if pt_idx2 >= len(pts2d_idx_inv[image_id2]):
                    continue
                idx2, pos2, j2 = pts2d_idx_inv[image_id2][pt_idx2]

                cam_point2 = (
                    cam_points_rotated[idx2][pos2][j2]
                    .numpy()
                    .astype(np.float64)
                )

                # Do not add constraint if same star
                if idx1 == idx2:
                    num_skipped_same_star += 1
                    continue

                # Check for the confidence of the two points
                pt_conf1 = predictions_dict["cam_points_conf"][idx1][
                    0, pos1, j1
                ]
                pt_conf2 = predictions_dict["cam_points_conf"][idx2][
                    0, pos2, j2
                ]

                if pt_conf1.item() < 0.01 or pt_conf2.item() < 0.01:
                    continue  # Skip low-confidence points

                weight = (
                    (pt_conf1.item() + 1e-6) * (pt_conf2.item() + 1e-6)
                ) * 0.01

                # if weight < 0.01:
                #     continue  # Skip low-confidence points

                loss = pyceres.LossFunction(
                    {"name": "cauchy", "params": [1.0], "magnitude": weight}
                )
                # Check if same camera
                if image_id1 == image_id2:
                    # Same camera, same star -> skip

                    # # Same camera, different stars -> use Point3DConsistencySameCamError
                    # cost = pygluemap.Point3DConsistencySameCamError(
                    #     cam_point1, cam_point2
                    # )
                    cost = pygluemap.DepthConsistencySameCamError(
                        predictions_dict["cam_points"][idx1][0, pos1, j1][
                            2
                        ].item(),
                        predictions_dict["cam_points"][idx2][0, pos2, j2][
                            2
                        ].item(),
                    )

                    prob.add_residual_block(
                        cost,
                        loss,
                        [
                            # global_centers[image_id1],  # center (same for both)
                            global_scales[idx1],  # scale_1
                            global_scales[idx2],  # scale_2
                        ],
                    )

                    num_same_cam_constraints += 1
                    pass
                else:
                    # Different cameras -> use Point3DConsistencyError
                    cost = pygluemap.Point3DConsistencyError(
                        cam_point1, cam_point2
                    )

                    prob.add_residual_block(
                        cost,
                        loss,
                        [
                            global_centers[image_id1],  # center_1
                            global_centers[image_id2],  # center_2
                            global_scales[idx1],  # scale_1
                            global_scales[idx2],  # scale_2
                        ],
                    )
                    num_constraints += 1
                    pass

                costs.append(cost)
                losses.append(loss)

    logger.info(f"Added {num_constraints} depth consistency constraints")
    logger.info(
        f"Added {num_same_cam_constraints} same-camera constraints (from tracks)"
    )
    logger.info(f"Skipped {num_skipped_same_star} same-star pairs")

    # Additionally, iterate through pts2d_idx_all to find points with same 2D index
    if pts2d_idx_all is not None:
        # Build mapping from (image_id, pt_idx) to list of (idx, i, j) sources
        pt_to_sources = defaultdict(list)

        for idx in pts2d_idx_all:
            for i in range(pts2d_idx_all[idx].shape[0]):
                image_id = predictions_dict["indexes"][idx][i]
                for j in range(pts2d_idx_all[idx].shape[1]):
                    pt_idx = pts2d_idx_all[idx][i, j]
                    if pt_idx >= 0:  # valid point (not -1)
                        pt_to_sources[(image_id, pt_idx)].append((idx, i, j))

        num_same_idx_constraints = 0

        # Add constraints for points with multiple sources (same 2D index from different stars)
        for (image_id, pt_idx), sources in pt_to_sources.items():
            if len(sources) < 2:
                continue

            # Add pairwise constraints between all sources
            for a in range(len(sources)):
                idx1, i1, j1 = sources[a]
                cam_point1 = (
                    cam_points_rotated[idx1][i1][j1].numpy().astype(np.float64)
                )

                for b in range(a + 1, len(sources)):
                    idx2, i2, j2 = sources[b]

                    if idx1 == idx2:
                        continue  # Skip same star

                    cam_point2 = (
                        cam_points_rotated[idx2][i2][j2]
                        .numpy()
                        .astype(np.float64)
                    )

                    # cost = pygluemap.Point3DConsistencySameCamError(
                    #     cam_point1, cam_point2
                    # )
                    cost = pygluemap.DepthConsistencySameCamError(
                        predictions_dict["cam_points"][idx1][0, i1, j1][
                            2
                        ].item(),
                        predictions_dict["cam_points"][idx2][0, i2, j2][
                            2
                        ].item(),
                    )
                    # loss = pyceres.LossFunction({"name": "huber", "params": [1.]})
                    # weight = np.log(
                    #     predictions_dict["cam_points_conf"][idx1][0, i1, j1].item()
                    #     + 1e-6
                    # ) * np.log(
                    #     predictions_dict["cam_points_conf"][idx2][0, i2, j2].item()
                    #     + 1e-6
                    # )
                    pt_conf1 = predictions_dict["cam_points_conf"][idx1][
                        0, i1, j1
                    ]
                    pt_conf2 = predictions_dict["cam_points_conf"][idx2][
                        0, i2, j2
                    ]
                    if pt_conf1.item() < 0.01 or pt_conf2.item() < 0.01:
                        continue  # Skip low-confidence points

                    weight = (
                        (pt_conf1.item() + 1e-6) * (pt_conf2.item() + 1e-6)
                    ) * 0.01

                    # if weight < 0.01:
                    #     continue  # Skip low-confidence points

                    loss = pyceres.LossFunction(
                        {"name": "cauchy", "params": [1.0], "magnitude": weight}
                    )

                    prob.add_residual_block(
                        cost,
                        loss,
                        [
                            # global_centers[image_id],  # center (same for both)
                            global_scales[idx1],  # scale_1
                            global_scales[idx2],  # scale_2
                        ],
                    )

                    costs.append(cost)
                    losses.append(loss)
                    num_same_idx_constraints += 1

        logger.info(
            f"Added {num_same_idx_constraints} same-2D-index constraints"
        )


def similarity_averaging_with_depth(
    predictions_dict,
    global_rotations,
    global_centers=None,
    global_scales=None,
    max_num_iterations=50,
    add_tracks=True,
    add_virtual_points=False,
    device="cuda",
    fix_scales=False,
):
    logger.info("Performing similarity averaging with depth...")
    num_ministar, global_centers, global_scales = initialize_parameters(
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
    )

    # Establish tracks as preprocessing
    num_images = len(global_rotations)
    options = TrackEstablishmentOptions(
        track_min_num_views_per_track=2,
    )
    (
        points3D,
        keypoints_per_image,
        pts2d_idx_all,
        pts2d_idx_virtual_all,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        image_to_point3D,
    ) = establish_tracks_from_tracks_dict(
        tracks_dict=predictions_dict,
        num_images=num_images,
        options=options,
        add_tracks=add_tracks,
        add_virtual_points=add_virtual_points,
        device=device,
    )
    logger.info(f"Established {len(points3D)} tracks")

    prob = pyceres.Problem()

    costs = []
    losses = []

    add_star_edge_error(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
        costs,
        losses,
    )

    add_depth_error(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
        costs,
        losses,
        points3D=points3D,
        keypoints_per_image=keypoints_per_image,
        pts2d_idx_all=pts2d_idx_all,
        pts2d_idx_virtual_all=pts2d_idx_virtual_all,
        pts2d_idx_inv=pts2d_idx_inv,
        pts2d_idx_virtual_inv=pts2d_idx_virtual_inv,
        image_to_point3D=image_to_point3D,
    )

    for idx_star in range(num_ministar):
        if not prob.has_parameter_block(global_scales[idx_star]):
            continue
        if fix_scales:
            prob.set_parameter_block_constant(global_scales[idx_star])
        else:
            prob.set_parameter_lower_bound(global_scales[idx_star], 0, 1e-5)
            prob.set_parameter_upper_bound(global_scales[idx_star], 0, 1e5)

    options = pyceres.SolverOptions()
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
    options.num_threads = 32
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = False

    logger.info("Solving the optimization problem...")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    logger.info(summary.BriefReport())

    update_points3d(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
    )

    return global_centers
