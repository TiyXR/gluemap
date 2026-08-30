import numpy as np

from gluemap.estimators.active_track_store import SelectedTrackState, TrackObservation
from gluemap.estimators.fixed_anchor_approximation import FixedAnchorWindowSolution
from gluemap.estimators.fixed_anchor_diagnostic_runner import (
    FixedAnchorDiagnosticRunner,
)


class _ExactCoarseSolver:
    def __init__(self, centers: dict[int, np.ndarray], intrinsics: np.ndarray) -> None:
        self.centers = centers
        self.intrinsics = intrinsics

    def solve(
        self,
        _predictions,
        frame_ids,
        *,
        initial_rotations=None,
        initial_centers=None,
        fixed_pose_ids=None,
    ):
        rotations = {value: np.eye(3) for value in frame_ids}
        centers = {value: self.centers[value].copy() for value in frame_ids}
        for frame_id in fixed_pose_ids or set():
            rotations[frame_id] = initial_rotations[frame_id].copy()
            centers[frame_id] = initial_centers[frame_id].copy()
        return FixedAnchorWindowSolution(
            frame_ids=tuple(frame_ids),
            rotations=rotations,
            centers=centers,
            intrinsics=[self.intrinsics[None]],
            report={"status": "passed", "backend": "exact-test"},
        )


def _tracks(
    frame_ids: list[int],
    centers: dict[int, np.ndarray],
    intrinsics: np.ndarray,
) -> list[SelectedTrackState]:
    result = []
    for track_index in range(32):
        point = np.array(
            (
                ((track_index % 8) - 4) * 0.12,
                ((track_index // 8) - 2) * 0.08,
                6.0 + (track_index % 5) * 0.3,
            )
        )
        observations = []
        for frame_id in frame_ids:
            camera = point - centers[frame_id]
            x = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
            y = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
            observations.append(
                TrackObservation(
                    observation_uid=f"track-{track_index}-frame-{frame_id}",
                    geometry_ordinal=frame_id,
                    frame_uid=f"frame-{frame_id}",
                    pts_value=frame_id,
                    time_base_numerator=1,
                    time_base_denominator=10,
                    x=float(x),
                    y=float(y),
                    image_width=640,
                    image_height=480,
                    score=1.0,
                )
            )
        result.append(
            SelectedTrackState(
                track_uid=f"track-{track_index}", observations=tuple(observations)
            )
        )
    return result


def _runner(centers, intrinsics) -> FixedAnchorDiagnosticRunner:
    return FixedAnchorDiagnosticRunner(
        initial_anchor_frame_ids={0, 1},
        coarse_solver=_ExactCoarseSolver(centers, intrinsics),
        camera_model="PINHOLE",
        triangulation_device_policy="cpu",
        triangulation_microbatch_tracks=16,
        ba_device_policy="cpu",
        ba_linear_solver_policy="auto",
        ba_max_iterations=20,
        ba_refinement_passes=1,
        ceres_cuda_available=False,
    )


def test_runner_advances_one_frame_and_restores_checkpoint() -> None:
    centers = {
        frame: np.array((float(frame), 0.0, 0.0)) for frame in range(4)
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    runner = _runner(centers, intrinsics)
    first = runner.advance({}, [0, 1, 2], _tracks([0, 1, 2], centers, intrinsics))
    second = runner.advance({}, [1, 2, 3], _tracks([1, 2, 3], centers, intrinsics))

    assert first.report["advanceStepKeyframes"] == 1
    assert second.report["overlapFrameCount"] == 2
    assert second.report["newFrameCount"] == 1
    assert second.report["maximumOverlapRotationMatrixDelta"] == 0
    assert second.report["maximumOverlapCenterDelta"] == 0
    assert second.report["zeroConstraintFrameIds"] == []
    assert second.report["actualBaCameraFrameUids"] == [
        "frame-1",
        "frame-2",
        "frame-3",
    ]

    checkpoint = runner.snapshot()
    restored = _runner(centers, intrinsics)
    restored.restore(checkpoint)
    assert restored.next_window_ordinal == 2
    assert restored.snapshot() == checkpoint
