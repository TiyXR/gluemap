import numpy as np
import pygluemap

from gluemap.estimators.active_track_store import TrackObservation
from gluemap.estimators.fixed_lag_ceres_linearization import (
    capture_explicit_ceres_problem_linearization,
)
from gluemap.estimators.fixed_lag_prior import marginalize_ceres_linearization
from gluemap.estimators.fixed_lag_triangulation import TriangulatedTrackState
from gluemap.estimators.persistent_fixed_lag_ba import (
    AUTO_DENSE_SCHUR_MAXIMUM_CAMERAS,
    PersistentFixedLagBaProblem,
    _solver_configuration,
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


def test_auto_solver_tracks_the_active_camera_frontier() -> None:
    import pyceres

    problem = pyceres.Problem()
    parameter = np.zeros((3,), dtype=np.float64)
    problem.add_parameter_block(parameter, 3)

    dense_options, requested_threads, use_gpu = _solver_configuration(
        problem,
        frame_count=AUTO_DENSE_SCHUR_MAXIMUM_CAMERAS,
        max_num_iterations=100,
        linear_solver_policy="auto",
        dense_linear_algebra_policy="auto",
        device_policy="cpu",
        ceres_cuda_available=False,
    )
    sparse_options, _, _ = _solver_configuration(
        problem,
        frame_count=AUTO_DENSE_SCHUR_MAXIMUM_CAMERAS + 1,
        max_num_iterations=100,
        linear_solver_policy="auto",
        dense_linear_algebra_policy="lapack",
        device_policy="cpu",
        ceres_cuda_available=False,
    )

    assert dense_options.linear_solver_type == pyceres.LinearSolverType.DENSE_SCHUR
    assert sparse_options.linear_solver_type == pyceres.LinearSolverType.SPARSE_SCHUR
    assert (
        sparse_options.dense_linear_algebra_library_type
        == pyceres.DenseLinearAlgebraLibraryType.LAPACK
    )
    assert requested_threads >= 1
    assert use_gpu is False


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
        linear_solver_ordering_policy="point-first",
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
    assert solve["linearSolverOrdering"] == "point-first"
    assert solve["orderedPointCount"] == 32
    assert solve["orderedPoseCount"] == 5
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


def test_native_rebuild_skips_discarded_observation_bookkeeping() -> None:
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(5)
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
        policy="native-rebuild-every-window",
    )

    report = problem.synchronize(
        frame_ids=tuple(centers),
        rotations=rotations,
        centers=centers,
        fixed_pose_ids={0},
        tracks=_tracks(tuple(centers), centers, intrinsics),
        camera_model_id=camera.model,
        camera_params=camera.params,
    )
    summary, _ = problem.solve(
        max_num_iterations=20,
        linear_solver_policy="auto",
        linear_solver_ordering_policy="point-first",
        device_policy="cpu",
        ceres_cuda_available=False,
    )

    assert report["createdObservationCount"] == 160
    assert report["residentObservationCount"] == 160
    assert report["phaseWallSeconds"]["observationBookkeeping"] >= 0.0
    assert all(not point.observations for point in problem.points.values())
    assert len(problem._native_batches) == 1
    assert summary.final_cost <= summary.initial_cost


def test_native_normal_blocks_match_crs_prior() -> None:
    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(5)
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
        policy="native-rebuild-every-window",
    )
    tracks = _tracks(tuple(centers), centers, intrinsics)
    problem.synchronize(
        frame_ids=tuple(centers),
        rotations=rotations,
        centers=centers,
        fixed_pose_ids={0},
        tracks=tracks,
        camera_model_id=camera.model,
        camera_params=camera.params,
    )
    problem.solve(
        max_num_iterations=20,
        linear_solver_policy="auto",
        linear_solver_ordering_policy="point-first",
        device_policy="cpu",
        ceres_cuda_available=False,
    )
    camera_ids = tuple(frame for frame in centers if frame != 0)
    point_uids = tuple(track.track_uid for track in tracks)
    point_parameters = problem.point_parameter_blocks(point_uids)
    capture_arguments = {
        "camera_ids": camera_ids,
        "image_ids": camera_ids,
        "point3d_ids": tuple(problem.point_id(value) for value in point_uids),
        "pose_parameters": problem.pose_parameter_blocks(camera_ids),
        "point_parameters": point_parameters,
        "residual_seed_parameters": point_parameters,
    }
    crs = capture_explicit_ceres_problem_linearization(
        problem.problem, **capture_arguments
    )
    normal_blocks = capture_explicit_ceres_problem_linearization(
        problem.problem, build_normal_blocks=True, **capture_arguments
    )

    crs_prior = marginalize_ceres_linearization(
        crs, eliminate_camera_id=camera_ids[0], device_policy="cpu"
    )
    normal_prior = marginalize_ceres_linearization(
        normal_blocks,
        eliminate_camera_id=camera_ids[0],
        device_policy="cpu",
    )

    assert normal_blocks.report["representation"] == "native-normal-blocks"
    assert normal_blocks.report["nativeNormalBuildWallSeconds"] >= 0.0
    np.testing.assert_allclose(
        normal_prior.hessian.numpy(),
        crs_prior.hessian.numpy(),
        rtol=1e-10,
        atol=2e-9,
    )
    np.testing.assert_allclose(
        normal_prior.gradient.numpy(),
        crs_prior.gradient.numpy(),
        rtol=1e-10,
        atol=1e-10,
    )


def test_native_csr_matches_image_major_residual_solution() -> None:
    if not hasattr(
        pygluemap,
        "add_reprojection_residual_csr_implicit_parameters",
    ):
        return

    intrinsics = np.array(
        ((500.0, 0.0, 320.0), (0.0, 500.0, 240.0), (0.0, 0.0, 1.0))
    )
    true_centers = {
        frame: np.array((frame * 0.5, 0.0, 0.0)) for frame in range(16)
    }
    initial_centers = {
        frame: value
        + np.array((0.0, 0.01 * ((frame % 3) - 1), 0.002 * (frame % 5)))
        for frame, value in true_centers.items()
    }
    rotations = {frame: np.eye(3) for frame in true_centers}
    camera = camera_from_intrinsics_matrix(
        intrinsics,
        camera_model="PINHOLE",
        width=640,
        height=480,
        camera_id=1,
    )
    tracks = _tracks(tuple(true_centers), true_centers, intrinsics)
    csr_binding = pygluemap.add_reprojection_residual_csr_implicit_parameters

    def solve(use_csr: bool):
        if not use_csr:
            delattr(
                pygluemap,
                "add_reprojection_residual_csr_implicit_parameters",
            )
        try:
            problem = PersistentFixedLagBaProblem(
                camera_model_id=camera.model,
                camera_params=camera.params,
                policy="native-rebuild-every-window",
            )
            report = problem.synchronize(
                frame_ids=tuple(true_centers),
                rotations=rotations,
                centers=initial_centers,
                fixed_pose_ids={0},
                tracks=tracks,
                camera_model_id=camera.model,
                camera_params=camera.params,
            )
            summary, _ = problem.solve(
                max_num_iterations=20,
                linear_solver_policy="auto",
                linear_solver_ordering_policy="point-first",
                device_policy="cpu",
                ceres_cuda_available=False,
            )
            poses = np.stack(
                [problem.pose_values(frame).copy() for frame in true_centers]
            )
            points = np.stack(
                [problem.point_values(track.track_uid).copy() for track in tracks]
            )
            return report, summary, poses, points
        finally:
            if not hasattr(
                pygluemap,
                "add_reprojection_residual_csr_implicit_parameters",
            ):
                setattr(
                    pygluemap,
                    "add_reprojection_residual_csr_implicit_parameters",
                    csr_binding,
                )

    old_report, old_summary, old_poses, old_points = solve(False)
    csr_report, csr_summary, csr_poses, csr_points = solve(True)

    assert old_report["visualResidualBindingMode"] == (
        "native-image-major-implicit-parameters"
    )
    assert csr_report["visualResidualBindingMode"] == (
        "native-image-major-csr-implicit-parameters"
    )
    assert np.isclose(old_summary.initial_cost, csr_summary.initial_cost, atol=1e-12)
    assert np.isclose(old_summary.final_cost, csr_summary.final_cost, atol=1e-12)
    assert np.allclose(old_poses, csr_poses, rtol=0.0, atol=1e-12)
    assert np.allclose(old_points, csr_points, rtol=0.0, atol=1e-12)
