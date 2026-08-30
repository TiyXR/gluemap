from types import SimpleNamespace

import numpy as np

from gluemap.estimators.active_track_store import (
    SelectedTrackState,
    TrackObservation,
)
from gluemap.estimators.fixed_anchor_approximation import (
    FixedAnchorWindowSolution,
)
from gluemap.estimators.fixed_anchor_local_ba import (
    _ba_summary_acceptance,
    refine_fixed_anchor_window,
)
from gluemap.estimators.fixed_lag_triangulation import triangulate_selected_tracks


def _observation(track: int, frame: int, x: float, y: float) -> TrackObservation:
    return TrackObservation(
        observation_uid=f"track-{track}-frame-{frame}",
        geometry_ordinal=frame,
        frame_uid=f"frame-{frame}",
        pts_value=frame,
        time_base_numerator=1,
        time_base_denominator=10,
        x=x,
        y=y,
        image_width=640,
        image_height=480,
        score=1.0,
    )


def test_iteration_limit_with_finite_cost_decrease_remains_usable() -> None:
    summary = SimpleNamespace(
        termination_type=SimpleNamespace(name="NO_CONVERGENCE"),
        ceres_summary=SimpleNamespace(
            initial_cost=10.0,
            final_cost=4.0,
            num_successful_steps=100,
            message="Maximum number of iterations reached. Number of iterations: 100.",
        ),
    )
    assert _ba_summary_acceptance(summary) == (
        True,
        "iteration-limit-cost-decrease",
    )


def test_iteration_limit_without_improvement_is_rejected() -> None:
    summary = SimpleNamespace(
        termination_type=SimpleNamespace(name="NO_CONVERGENCE"),
        ceres_summary=SimpleNamespace(
            initial_cost=10.0,
            final_cost=11.0,
            num_successful_steps=100,
            message="Maximum number of iterations reached. Number of iterations: 100.",
        ),
    )
    assert _ba_summary_acceptance(summary) == (False, "unusable-termination")


def test_local_ba_keeps_fixed_overlap_poses() -> None:
    rotations = {frame: np.eye(3) for frame in range(3)}
    centers = {
        0: np.array((0.0, 0.0, 0.0)),
        1: np.array((1.0, 0.0, 0.0)),
        2: np.array((2.03, 0.01, 0.0)),
    }
    exact_centers = {
        0: np.array((0.0, 0.0, 0.0)),
        1: np.array((1.0, 0.0, 0.0)),
        2: np.array((2.0, 0.0, 0.0)),
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    tracks = []
    for track_index in range(32):
        point = np.array(
            (
                ((track_index % 8) - 4) * 0.12,
                ((track_index // 8) - 2) * 0.08,
                6.0 + (track_index % 5) * 0.3,
            )
        )
        observations = []
        for frame, center in exact_centers.items():
            camera = point - center
            x = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
            y = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
            observations.append(_observation(track_index, frame, x, y))
        tracks.append(
            SelectedTrackState(
                track_uid=f"track-{track_index}",
                observations=tuple(observations),
            )
        )

    triangulated, _ = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
    )
    coarse = FixedAnchorWindowSolution(
        frame_ids=(0, 1, 2),
        rotations=rotations,
        centers=centers,
        intrinsics=[intrinsics[None]],
        report={},
    )
    solution = refine_fixed_anchor_window(
        coarse,
        triangulated,
        intrinsics,
        fixed_pose_ids={0, 1},
        camera_model="PINHOLE",
        max_num_iterations=20,
        device_policy="cpu",
        ceres_cuda_available=False,
    )

    assert solution.report["status"] == "passed"
    assert solution.report["gpuUsed"] is False
    assert solution.report["trackCount"] == 32
    assert solution.report["observationCount"] == 96
    assert solution.report["maximumFixedRotationMatrixDelta"] == 0
    assert solution.report["maximumFixedCenterDelta"] == 0
    assert solution.report["ceresThreadsGiven"] >= 1
    assert solution.report["ceresThreadsUsed"] >= 1
    assert solution.next_prior is None
    assert np.allclose(solution.centers[0], centers[0])
    assert np.allclose(solution.centers[1], centers[1])
    assert solution.report["finalCost"] <= solution.report["initialCost"]


def test_local_ba_emits_gpu_schur_fej_prior() -> None:
    frame_count = 5
    rotations = {frame: np.eye(3) for frame in range(frame_count)}
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0))
        for frame in range(frame_count)
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    tracks = []
    for track_index in range(64):
        point = np.array(
            (
                ((track_index % 8) - 4) * 0.16,
                ((track_index // 8) - 4) * 0.1,
                6.0 + (track_index % 7) * 0.25,
            )
        )
        observations = []
        for frame, center in centers.items():
            camera = point - center
            observations.append(
                _observation(
                    track_index,
                    frame,
                    intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2],
                    intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2],
                )
            )
        tracks.append(
            SelectedTrackState(
                track_uid=f"track-{track_index}",
                observations=tuple(observations),
            )
        )
    triangulated, _ = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
    )
    coarse = FixedAnchorWindowSolution(
        frame_ids=tuple(range(frame_count)),
        rotations=rotations,
        centers=centers,
        intrinsics=[intrinsics[None]],
        report={},
    )

    solution = refine_fixed_anchor_window(
        coarse,
        triangulated,
        intrinsics,
        fixed_pose_ids={0},
        camera_model="PINHOLE",
        max_num_iterations=20,
        device_policy="cpu",
        ceres_cuda_available=False,
        marginalize_pose_id=1,
        prior_device_policy="cuda-required",
    )

    assert solution.report["status"] == "passed"
    assert solution.next_prior is not None
    assert solution.next_prior.camera_ids == (2, 3, 4)
    assert solution.next_prior.report["status"] == "passed"
    assert solution.next_prior.report["gpuUsed"] is True
    assert solution.next_prior.report["eliminatedCameraId"] == 1
    assert solution.next_prior.report["pointCount"] == 64
