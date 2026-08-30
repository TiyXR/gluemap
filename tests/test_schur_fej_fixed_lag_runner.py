import numpy as np

from gluemap.estimators.active_track_store import (
    SelectedTrackState,
    TrackObservation,
)
from gluemap.estimators.fixed_anchor_approximation import (
    FixedAnchorWindowSolution,
)
from gluemap.estimators.schur_fej_fixed_lag_runner import (
    SchurFejFixedLagRunner,
)


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


def _window(frame_ids, centers, intrinsics):
    tracks = []
    for track_index in range(64):
        point = np.array(
            (
                ((track_index % 8) - 4) * 0.16,
                ((track_index // 8) - 4) * 0.1,
                6.0 + (track_index % 7) * 0.25,
            )
        )
        tracks.append(
            SelectedTrackState(
                track_uid=f"track-{track_index}",
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
            )
        )
    coarse = FixedAnchorWindowSolution(
        frame_ids=tuple(frame_ids),
        rotations={frame: np.eye(3) for frame in frame_ids},
        centers={frame: centers[frame].copy() for frame in frame_ids},
        intrinsics=[intrinsics[None]],
        report={},
    )
    return coarse, tracks


def test_two_advances_consume_previous_prior_on_real_cuda_backend():
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(6)
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    first_coarse, first_tracks = _window(
        (0, 1, 2, 3, 4), centers, intrinsics
    )
    second_coarse, second_tracks = _window(
        (0, 2, 3, 4, 5), centers, intrinsics
    )
    runner = SchurFejFixedLagRunner(
        fixed_gauge_frame_ids={0},
        camera_model="PINHOLE",
        triangulation_device_policy="cuda-required",
        ba_device_policy="cpu",
        ceres_cuda_available=False,
        prior_device_policy="cuda-required",
        prior_expected_nullity=1,
    )

    first = runner.advance(
        first_coarse, first_tracks, marginalize_frame_id=1
    )
    checkpoint = runner.snapshot()
    second = runner.advance(
        second_coarse, second_tracks, marginalize_frame_id=2
    )
    resumed_runner = SchurFejFixedLagRunner(
        fixed_gauge_frame_ids={0},
        camera_model="PINHOLE",
        triangulation_device_policy="cuda-required",
        ba_device_policy="cpu",
        ceres_cuda_available=False,
        prior_device_policy="cuda-required",
        prior_expected_nullity=1,
    )
    resumed_runner.restore(checkpoint)
    resumed_second = resumed_runner.advance(
        second_coarse, second_tracks, marginalize_frame_id=2
    )

    assert first.report["status"] == "passed"
    assert first.report["previousPriorCameraCount"] == 0
    assert first.prior.camera_ids == (2, 3, 4)
    assert first.prior.report["gpuUsed"] is True
    assert first.prior.report["priorNullity"] == 1
    assert second.report["status"] == "passed"
    assert second.report["previousPriorCameraCount"] == 3
    assert second.prior.camera_ids == (3, 4, 5)
    assert second.prior.report["gpuUsed"] is True
    assert second.prior.report["priorNullity"] == 1
    assert runner.next_window_ordinal == 2
    assert resumed_second.report["previousPriorCameraCount"] == 3
    assert resumed_second.prior.camera_ids == second.prior.camera_ids
    np.testing.assert_allclose(
        resumed_second.finalized_rotation,
        second.finalized_rotation,
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        resumed_second.finalized_center,
        second.finalized_center,
        rtol=1e-9,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        resumed_second.prior.hessian.cpu(),
        second.prior.hessian.cpu(),
        rtol=1e-9,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        resumed_second.prior.gradient.cpu(),
        second.prior.gradient.cpu(),
        rtol=1e-9,
        atol=1e-9,
    )
