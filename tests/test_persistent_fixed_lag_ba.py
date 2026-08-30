import numpy as np

from gluemap.estimators.active_track_store import TrackObservation
from gluemap.estimators.fixed_lag_triangulation import TriangulatedTrackState
from gluemap.estimators.persistent_fixed_lag_ba import (
    PersistentFixedLagBaProblem,
)
from gluemap.utils.colmap import camera_from_intrinsics_matrix


def _observation(track: int, frame: int, point, center, intrinsics):
    camera = point - center
    return TrackObservation(
        observation_uid=f"track-{track}-frame-{frame}",
        geometry_ordinal=frame,
        frame_uid=f"frame-{frame}",
        pts_value=frame,
        time_base_numerator=1,
        time_base_denominator=10,
        x=intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2],
        y=intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2],
        image_width=640,
        image_height=480,
        score=1.0,
    )


def _tracks(frame_ids, centers, intrinsics):
    values = []
    for track_index in range(32):
        point = np.array(
            (
                ((track_index % 8) - 4) * 0.16,
                ((track_index // 8) - 2) * 0.1,
                6.0 + (track_index % 7) * 0.25,
            )
        )
        values.append(
            TriangulatedTrackState(
                track_uid=f"track-{track_index}",
                xyz=tuple(point),
                observations=tuple(
                    _observation(
                        track_index,
                        frame,
                        point,
                        centers[frame],
                        intrinsics,
                    )
                    for frame in frame_ids
                ),
                positive_depth_fraction=1.0,
                maximum_reprojection_error_pixels=0.0,
            )
        )
    return values


def test_persistent_problem_applies_native_visual_enter_leave_delta() -> None:
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(6)
    }
    rotations = {frame: np.eye(3) for frame in centers}
    camera = camera_from_intrinsics_matrix(
        intrinsics,
        camera_model="PINHOLE",
        width=640,
        height=480,
        camera_id=1,
    )
    problem = PersistentFixedLagBaProblem(
        camera_model_id=camera.model,
        camera_params=camera.params,
    )

    first = problem.synchronize(
        frame_ids=(0, 1, 2, 3, 4),
        rotations={value: rotations[value] for value in (0, 1, 2, 3, 4)},
        centers={value: centers[value] for value in (0, 1, 2, 3, 4)},
        fixed_pose_ids={0},
        tracks=_tracks((0, 1, 2, 3, 4), centers, intrinsics),
        camera_model_id=camera.model,
        camera_params=camera.params,
    )
    summary, solve = problem.solve(
        max_num_iterations=20,
        linear_solver_policy="auto",
        device_policy="cpu",
        ceres_cuda_available=False,
    )
    second = problem.synchronize(
        frame_ids=(0, 2, 3, 4, 5),
        rotations={value: rotations[value] for value in (0, 2, 3, 4, 5)},
        centers={value: centers[value] for value in (0, 2, 3, 4, 5)},
        fixed_pose_ids={0},
        tracks=_tracks((0, 2, 3, 4, 5), centers, intrinsics),
        camera_model_id=camera.model,
        camera_params=camera.params,
    )
    unchanged = problem.synchronize(
        frame_ids=(0, 2, 3, 4, 5),
        rotations={value: rotations[value] for value in (0, 2, 3, 4, 5)},
        centers={value: centers[value] for value in (0, 2, 3, 4, 5)},
        fixed_pose_ids={0},
        tracks=_tracks((0, 2, 3, 4, 5), centers, intrinsics),
        camera_model_id=camera.model,
        camera_params=camera.params,
    )

    assert first["createdPoseCount"] == 5
    assert first["createdPointCount"] == 32
    assert first["createdObservationCount"] == 160
    assert first["visualResidualBindingMode"] == "native-enter-leave-delta"
    assert first["problemResidualCount"] == 320
    assert solve["gpuRequested"] is False
    assert summary.final_cost <= summary.initial_cost
    assert second["createdPoseCount"] == 1
    assert second["removedPoseCount"] == 1
    assert second["createdPointCount"] == 0
    assert second["reusedPointCount"] == 32
    assert second["createdObservationCount"] == 32
    assert second["reusedObservationCount"] == 128
    assert second["removedObservationCount"] == 32
    assert second["residentObservationCount"] == 160
    assert second["visualResidualBindingMode"] == "native-enter-leave-delta"
    assert second["problemResidualCount"] == 320
    assert unchanged["createdObservationCount"] == 0
    assert unchanged["removedObservationCount"] == 0
    assert unchanged["reusedObservationCount"] == 160
    assert unchanged["problemResidualCount"] == 320
