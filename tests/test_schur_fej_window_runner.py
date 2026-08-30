import numpy as np

from gluemap.estimators.active_track_store import (
    SelectedTrackState,
    TrackObservation,
)
from gluemap.estimators.fixed_anchor_approximation import (
    FixedAnchorWindowSolution,
)
from gluemap.estimators.schur_fej_window_runner import SchurFejWindowRunner


class _CoarseSolver:
    def __init__(self, centers, intrinsics):
        self.centers = centers
        self.intrinsics = intrinsics
        self.calls = []

    def solve(
        self,
        _predictions,
        frame_ids,
        *,
        initial_rotations,
        initial_centers,
        fixed_pose_ids,
    ):
        self.calls.append((tuple(frame_ids), set(fixed_pose_ids)))
        rotations = {frame: np.eye(3) for frame in frame_ids}
        centers = {frame: self.centers[frame].copy() for frame in frame_ids}
        if initial_rotations is not None:
            for frame in fixed_pose_ids:
                rotations[frame] = initial_rotations[frame].copy()
                centers[frame] = initial_centers[frame].copy()
        return FixedAnchorWindowSolution(
            frame_ids=tuple(frame_ids),
            rotations=rotations,
            centers=centers,
            intrinsics=[self.intrinsics[None]],
            report={"status": "passed"},
        )


def _tracks(frame_ids, centers, intrinsics):
    result = []
    for track_index in range(64):
        point = np.array(
            (
                ((track_index % 8) - 4) * 0.16,
                ((track_index // 8) - 4) * 0.1,
                6.0 + (track_index % 7) * 0.25,
            )
        )
        observations = []
        for frame in frame_ids:
            camera = point - centers[frame]
            observations.append(
                TrackObservation(
                    observation_uid=f"track-{track_index}-frame-{frame}",
                    geometry_ordinal=frame,
                    frame_uid=f"frame-{frame}",
                    pts_value=frame,
                    time_base_numerator=1,
                    time_base_denominator=10,
                    x=intrinsics[0, 0] * camera[0] / camera[2]
                    + intrinsics[0, 2],
                    y=intrinsics[1, 1] * camera[1] / camera[2]
                    + intrinsics[1, 2],
                    image_width=640,
                    image_height=480,
                    score=1.0,
                )
            )
        result.append(
            SelectedTrackState(
                track_uid=f"track-{track_index}",
                observations=tuple(observations),
            )
        )
    return result


def test_true_window_keeps_only_gauge_fixed_and_marginalizes_one_body_pose():
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(6)
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    coarse = _CoarseSolver(centers, intrinsics)
    runner = SchurFejWindowRunner(
        fixed_gauge_frame_id=0,
        coarse_solver=coarse,
        camera_model="PINHOLE",
        triangulation_device_policy="cuda-required",
        ba_device_policy="cpu",
        ceres_cuda_available=False,
        prior_device_policy="cuda-required",
        prior_expected_nullity=1,
    )
    first_ids = [0, 1, 2, 3, 4]
    second_ids = [0, 2, 3, 4, 5]

    first = runner.advance({}, first_ids, _tracks(first_ids, centers, intrinsics))
    second = runner.advance(
        {}, second_ids, _tracks(second_ids, centers, intrinsics)
    )

    assert first.solved.finalized_frame_id == 1
    assert second.solved.finalized_frame_id == 2
    assert coarse.calls == [
        ((0, 1, 2, 3, 4), set()),
        ((0, 2, 3, 4, 5), {0, 2, 3, 4}),
    ]
    assert second.report["coarseFixedWarmStartCount"] == 4
    assert second.report["localBa"]["fixedPoseCount"] == 1
    assert second.report["prior"]["priorNullity"] == 1


def test_terminal_drain_finalizes_every_retained_body_pose_on_cuda():
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(6)
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    runner = SchurFejWindowRunner(
        fixed_gauge_frame_id=0,
        coarse_solver=_CoarseSolver(centers, intrinsics),
        camera_model="PINHOLE",
        triangulation_device_policy="cuda-required",
        ba_device_policy="cpu",
        ceres_cuda_available=False,
        prior_device_policy="cuda-required",
        prior_expected_nullity=1,
    )
    first_ids = [0, 1, 2, 3, 4]
    second_ids = [0, 2, 3, 4, 5]
    runner.advance({}, first_ids, _tracks(first_ids, centers, intrinsics))
    runner.advance({}, second_ids, _tracks(second_ids, centers, intrinsics))

    drained = []
    while runner.fixed_lag.prior is not None:
        frame_ids = sorted(
            {0, *runner.fixed_lag.prior.camera_ids}
        )
        drained.append(
            runner.drain_next(_tracks(frame_ids, centers, intrinsics))
        )

    assert [value.solved.finalized_frame_id for value in drained] == [3, 4, 5]
    assert [value.report["terminalDrain"] for value in drained] == [True] * 3
    assert [value.report["terminalSolveMode"] for value in drained] == [
        "prior-only-schur",
        "prior-only-schur",
        "sealed-pose-freeze",
    ]
    assert all(value.solved.triangulated_tracks == () for value in drained)
    assert all(value.report["actualBaCameraFrameUids"] == [] for value in drained)
    assert all(value.report["zeroConstraintFrameIds"] == [] for value in drained)
    assert all(value.report["localBa"]["ceresThreadsUsed"] == 0 for value in drained)
    assert drained[0].report["prior"]["gpuUsed"] is True
    assert drained[-1].report["terminalFinalized"] is True
    assert drained[-1].report["prior"] is None
    assert runner.fixed_lag.current_frame_ids == (0,)
    terminal = runner.snapshot_terminal()
    resumed = SchurFejWindowRunner(
        fixed_gauge_frame_id=0,
        coarse_solver=_CoarseSolver(centers, intrinsics),
        camera_model="PINHOLE",
        triangulation_device_policy="cuda-required",
        ba_device_policy="cpu",
        ceres_cuda_available=False,
        prior_device_policy="cuda-required",
        prior_expected_nullity=1,
    )
    resumed.restore(terminal)
    assert resumed.fixed_lag.terminal_finalized is True
    assert resumed.fixed_lag.prior is None
    assert resumed.fixed_lag.current_frame_ids == (0,)
