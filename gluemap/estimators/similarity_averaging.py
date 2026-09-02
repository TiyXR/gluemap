import logging

import numpy as np
import pyceres
import pygluemap
import torch

from gluemap.utils.runtime_capacity import resolve_native_thread_count

logger = logging.getLogger(__name__)
# Minimum angle (in degrees) for a triangle to be considered valid for
# scale estimation
MIN_TRI_ANGLE = 1


def _initialize_parameters(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None,
    global_scales: list[np.ndarray] | dict[int, float] | None,
    initialization_seed: int = 0,
) -> tuple[int, dict[int, np.ndarray], list[np.ndarray]]:
    """
    Allocate per-image centers and per-ministar scale parameter buffers.

    Args:
        predictions_dict: Star inference output; only ``"indexes"`` is read
            here to determine the number of ministars.
        global_rotations: Per-image rotation matrices ``{image_id: R}``.
        global_centers: Optional initial per-image camera centers; missing
            images get random initialization.
        global_scales: Optional initial per-ministar scales as a list, or as
            a ``{ministar_idx: scale}`` mapping; ``None`` defaults to ones.

    Returns:
        ``(num_ministar, global_centers, global_scales)`` where the latter
        two are normalized into the canonical containers used downstream.
    """
    num_ministar = len(predictions_dict["indexes"])
    def deterministic_center(image_id: int) -> np.ndarray:
        seed = np.random.SeedSequence(
            [int(initialization_seed) & 0xFFFFFFFF, int(image_id) & 0xFFFFFFFF]
        )
        return np.random.default_rng(seed).random(3, dtype=np.float64)

    if global_centers is None:
        global_centers = {
            idx: deterministic_center(idx) for idx in global_rotations
        }
    else:
        for idx in global_rotations:
            if idx not in global_centers:
                global_centers[idx] = deterministic_center(idx)
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


def _add_star_edge_error(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
    costs: list,
    losses: list,
    fixed_center_ids: set[int] | None = None,
) -> int:
    """
    Add pairwise direction residuals for every valid star edge.

    For each ministar's center image and its neighbours, attaches a
    ``PairwiseDirectionError`` cost block linking the two camera centers
    and the ministar scale. Edges with score <= 0 or median triangulation
    angle below :data:`MIN_TRI_ANGLE` are skipped. Pins the first chosen
    center and its ministar's scale to remove the gauge.

    Args:
        prob: Ceres problem to append residuals to.
        predictions_dict: Star inference output with ``"indexes"``,
            ``"pose_scores"``, ``"median_tri_angle"`` and ``"extrinsics"``.
        global_rotations: Per-image rotation matrices.
        global_centers: Per-image camera centers (parameter blocks).
        global_scales: Per-ministar scale parameter blocks.
        num_ministar: Number of ministars to iterate over.
        costs: Output list to which created cost objects are appended (kept
            alive by the caller).
        losses: Output list of created loss functions (kept alive too).
        fixed_center_ids: Existing canonical centers to keep constant. Two or
            more fixed centers also determine scale, so no arbitrary star
            scale is pinned in that case.

    Returns:
        The image id whose center was fixed for gauge, or ``-1`` if no edge
        was added.
    """
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

            cost = pygluemap.PairwiseDirectionError(t_ij_rotated)
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
                center = idx1
                scale_anchor_star = idx_star

    if center < 0:
        return center

    fixed_count = 0
    for image_id in sorted(fixed_center_ids or set()):
        parameter = global_centers.get(image_id)
        if parameter is not None and prob.has_parameter_block(parameter):
            prob.set_parameter_block_constant(parameter)
            fixed_count += 1
    if fixed_count == 0:
        prob.set_parameter_block_constant(global_centers[center])
    if fixed_count < 2:
        prob.set_parameter_block_constant(global_scales[scale_anchor_star])

    return center


def _update_points3d(
    prob: pyceres.Problem,
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_scales: list[np.ndarray],
    num_ministar: int,
) -> None:
    """
    Apply solved similarity transforms back to predicted 3D content.

    Rescales/rotates ``world_points``, ``cam_points`` and
    ``points3d_virtual`` (when present) and rescales ministar extrinsics so
    everything is expressed in the new global frame. Mutates
    ``predictions_dict`` in place.

    Args:
        prob: Ceres problem used to detect which ministar scales were
            actually parameter blocks (others are skipped).
        predictions_dict: Star inference dict; relevant tensors are updated
            in place.
        global_rotations: Per-image rotation matrices used to rotate world
            points back into the global frame.
        global_centers: Per-image solved centers used as translations.
        global_scales: Per-ministar solved scales (length-1 numpy arrays).
        num_ministar: Number of ministars to process.
    """
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
        and predictions_dict["cam_points"][0] is not None
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
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray] | None = None,
    global_scales: list[np.ndarray] | dict[int, float] | None = None,
    max_num_iterations: int = 50,
    fix_scales: bool = False,
    fixed_center_ids: set[int] | None = None,
    initialization_seed: int = 0,
) -> dict[int, np.ndarray]:
    """
    Solve for global per-image translations and per-ministar scales.

    Builds a Ceres problem from pairwise direction residuals over all valid
    ministar edges, fixes the gauge, solves, then back-substitutes the
    similarity transform into ``predictions_dict``.

    Args:
        predictions_dict: Star inference output; mutated in place by
            :func:`_update_points3d`.
        global_rotations: Per-image rotations (assumed already solved by
            rotation averaging).
        global_centers: Optional initial centers; missing entries are
            randomized via :func:`_initialize_parameters`.
        global_scales: Optional initial scales (list or dict).
        max_num_iterations: Max Ceres iterations.
        fix_scales: If True, freeze every ministar scale (useful when only
            translations should move).
        fixed_center_ids: Image ids whose supplied ``global_centers`` are
            canonical fixed anchors. With at least two, the fixed centers
            replace the legacy arbitrary center-and-scale gauge.

    Returns:
        The updated ``global_centers`` mapping (also written in place).
    """
    logger.info("Performing similarity averaging...")
    num_ministar, global_centers, global_scales = _initialize_parameters(
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        initialization_seed,
    )

    prob = pyceres.Problem()

    costs = []
    losses = []

    _add_star_edge_error(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
        costs,
        losses,
        fixed_center_ids,
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
    options.num_threads = resolve_native_thread_count()
    options.max_num_iterations = max_num_iterations
    options.minimizer_progress_to_stdout = False

    logger.info("Solving the optimization problem...")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)

    logger.info(summary.BriefReport())

    _update_points3d(
        prob,
        predictions_dict,
        global_rotations,
        global_centers,
        global_scales,
        num_ministar,
    )

    return global_centers
