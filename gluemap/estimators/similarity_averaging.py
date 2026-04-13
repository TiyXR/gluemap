import logging
from collections import defaultdict

import numpy as np
import pyceres
import pygluemap
import torch


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
