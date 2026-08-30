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


def test_reduced_gpu_solver_candidates_match_homogeneous_svd() -> None:
    rotations = {frame: np.eye(3) for frame in range(6)}
    centers = {
        frame: np.array((float(frame) * 0.3, 0.02 * frame, 0.0))
        for frame in rotations
    }
    intrinsics = np.array(
        ((900.0, 0.0, 640.0), (0.0, 900.0, 360.0), (0.0, 0.0, 1.0))
    )
    tracks = []
    for track_index in range(32):
        point = np.array(
            (
                0.1 + 0.07 * track_index,
                -0.3 + 0.02 * track_index,
                8.0 + 0.15 * track_index,
            )
        )
        observations = []
        for frame, center in centers.items():
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

    svd, svd_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="homogeneous-svd",
    )
    gram, gram_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="homogeneous-gram-eigh",
    )
    qr, qr_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="homogeneous-qr-svd",
    )
    least_squares, least_squares_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="inhomogeneous-lstsq",
    )
    hybrid, hybrid_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="homogeneous-gram-eigh-fallback-svd",
        solver_fallback_relative_eigenvalue=1e-30,
    )
    cpu_lapack, cpu_lapack_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="homogeneous-svd-cpu-lapack",
    )
    automatic, automatic_report = triangulate_selected_tracks(
        tracks,
        rotations,
        centers,
        intrinsics,
        device_policy="cpu",
        microbatch_tracks=7,
        solver_policy="homogeneous-svd-auto-benchmark",
    )

    assert svd_report["solverPolicy"] == "homogeneous-svd"
    assert gram_report["solverPolicy"] == "homogeneous-gram-eigh"
    assert qr_report["solverPolicy"] == "homogeneous-qr-svd"
    assert least_squares_report["solverPolicy"] == "inhomogeneous-lstsq"
    assert hybrid_report["solverPolicy"] == (
        "homogeneous-gram-eigh-fallback-svd"
    )
    assert (
        hybrid_report["solverFastTrackCount"]
        + hybrid_report["solverFallbackTrackCount"]
        == len(tracks)
    )
    assert cpu_lapack_report["solverComputeBackend"] == "cpu-lapack"
    assert automatic_report["resolvedSolverPolicy"] == (
        "homogeneous-svd-cpu-lapack"
    )
    assert automatic_report["solverBenchmarkTrackCount"] == 7
    assert [value.track_uid for value in gram] == [value.track_uid for value in svd]
    assert np.allclose(
        np.asarray([value.xyz for value in gram]),
        np.asarray([value.xyz for value in svd]),
        rtol=1e-8,
        atol=1e-8,
    )
    assert np.allclose(
        np.asarray([value.xyz for value in qr]),
        np.asarray([value.xyz for value in svd]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray([value.xyz for value in least_squares]),
        np.asarray([value.xyz for value in svd]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray([value.xyz for value in hybrid]),
        np.asarray([value.xyz for value in svd]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray([value.xyz for value in cpu_lapack]),
        np.asarray([value.xyz for value in svd]),
        rtol=1e-10,
        atol=1e-10,
    )
    assert np.allclose(
        np.asarray([value.xyz for value in automatic]),
        np.asarray([value.xyz for value in svd]),
        rtol=1e-10,
        atol=1e-10,
    )
