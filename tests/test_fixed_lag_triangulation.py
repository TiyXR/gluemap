import numpy as np

from gluemap.estimators.active_track_store import (
    SelectedTrackState,
    TrackObservation,
)
from gluemap.estimators.fixed_lag_triangulation import (
    triangulate_selected_tracks,
)


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


def test_gpu_style_batch_dlt_recovers_known_points() -> None:
    rotations = {frame: np.eye(3) for frame in range(3)}
    centers = {
        0: np.array((0.0, 0.0, 0.0)),
        1: np.array((1.0, 0.0, 0.0)),
        2: np.array((2.0, 0.0, 0.0)),
    }
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    points = [(0.25, 0.1, 5.0), (1.0, -0.2, 8.0)]
    tracks = []
    for track_index, point in enumerate(points):
        observations = []
        for frame, center in centers.items():
            camera = np.asarray(point) - center
            x = intrinsics[0, 0] * camera[0] / camera[2] + intrinsics[0, 2]
            y = intrinsics[1, 1] * camera[1] / camera[2] + intrinsics[1, 2]
            observations.append(_observation(track_index, frame, x, y))
        tracks.append(
            SelectedTrackState(
                track_uid=f"track-{track_index}",
                observations=tuple(observations),
            )
        )

    triangulated, report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=2,
    )

    assert report["status"] == "passed"
    assert report["triangulatedTrackCount"] == 2
    assert report["reprojectionErrorP95Pixels"] < 1e-9
    for actual, expected in zip(triangulated, points, strict=True):
        assert np.allclose(actual.xyz, expected, atol=1e-9)
        assert actual.positive_depth_fraction == 1.0
