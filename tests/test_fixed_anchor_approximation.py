import numpy as np
import torch

from gluemap.estimators.fixed_anchor_approximation import (
    FixedAnchorApproximationSolver,
    slice_star_predictions,
)
from tests.helpers import (
    build_predictions_dict,
    build_star_topology_full,
    create_synthetic_reconstruction,
    extract_gt,
)


def _predictions() -> tuple[dict, list[int]]:
    reconstruction = create_synthetic_reconstruction(num_frames=8, seed=51)
    frame_ids, rotations, centers = extract_gt(reconstruction)
    raw = build_predictions_dict(
        rotations,
        centers,
        build_star_topology_full(frame_ids),
        np.ones(len(frame_ids)),
        generate_points3d_virtual=True,
    )
    predictions = {
        key: [values[index] for index in range(len(frame_ids))]
        for key, values in raw.items()
    }
    for extrinsics in predictions["extrinsics"]:
        extrinsics[0, 0, :3, :3] = torch.eye(3)
    predictions["intrinsics"] = []
    predictions["vis"] = []
    for indexes in predictions["indexes"]:
        intrinsics = torch.eye(3).reshape(1, 1, 3, 3).repeat(
            1, len(indexes), 1, 1
        )
        intrinsics[:, :, 0, 0] = 1000
        intrinsics[:, :, 1, 1] = 1000
        predictions["intrinsics"].append(intrinsics)
        predictions["vis"].append(torch.ones(1, len(indexes), 8))
    return predictions, frame_ids


def test_slice_remaps_one_window_without_mutating_source() -> None:
    predictions, frame_ids = _predictions()
    source_members = list(predictions["indexes"][1])
    sliced, global_to_local, local_to_global = slice_star_predictions(
        predictions, frame_ids[1:6]
    )
    assert global_to_local[frame_ids[1]] == 0
    assert local_to_global[4] == frame_ids[5]
    assert len(sliced["indexes"]) == 5
    assert predictions["indexes"][1] == source_members


def test_next_window_keeps_overlap_anchor_exact() -> None:
    predictions, frame_ids = _predictions()
    solver = FixedAnchorApproximationSolver(sequential_neighbor_distance=7)
    first = solver.solve(predictions, frame_ids[:7])
    fixed_ids = set(frame_ids[1:4])
    second = solver.solve(
        predictions,
        frame_ids[1:],
        initial_rotations=first.rotations,
        initial_centers=first.centers,
        fixed_pose_ids=fixed_ids,
    )
    assert second.report["status"] == "passed"
    assert second.report["fixedPoseCount"] == 3
    assert second.report["fixedAnchorMaximumRotationMatrixDelta"] < 1e-12
    assert second.report["fixedAnchorMaximumCenterDelta"] == 0


def test_solver_keeps_exact_declared_sequential_edges() -> None:
    solver = FixedAnchorApproximationSolver(
        sequential_neighbor_distance=1,
        sequential_edges={(0, 2), (1, 3)},
    )
    assert solver.sequential_edges == {(0, 2), (1, 3)}
