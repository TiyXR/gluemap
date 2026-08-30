"""Build and refine one bounded fixed-anchor window without image IO."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pycolmap

from gluemap.estimators.augmented_bundle_adjustment import bundle_adjustment
from gluemap.estimators.fixed_anchor_approximation import (
    FixedAnchorWindowSolution,
)
from gluemap.estimators.fixed_lag_triangulation import TriangulatedTrackState
from gluemap.estimators.fixed_lag_ceres_linearization import (
    CeresProblemLinearization,
    capture_ceres_problem_linearization,
)
from gluemap.estimators.fixed_lag_prior import (
    FejPriorState,
    marginalize_ceres_linearization,
)
from gluemap.utils.colmap import camera_from_intrinsics_matrix
from gluemap.utils.runtime_capacity import resolve_native_thread_count


class FixedAnchorLocalBaError(ValueError):
    """Raised when a bounded local BA window violates its identity."""


@dataclass(frozen=True)
class FixedAnchorLocalBaSolution:
    frame_ids: tuple[int, ...]
    rotations: dict[int, np.ndarray]
    centers: dict[int, np.ndarray]
    intrinsics: np.ndarray
    track_point3d_ids: dict[str, int]
    reconstruction: pycolmap.Reconstruction
    next_prior: FejPriorState | None
    report: dict[str, Any]


def _shared_intrinsics_matrix(value: Any) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float64).squeeze()
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise FixedAnchorLocalBaError("shared intrinsics matrix is invalid")
    return matrix


def _resolved_backend(summary: object) -> tuple[str, str, bool]:
    ceres = getattr(summary, "ceres_summary", summary)
    dense = str(ceres.dense_linear_algebra_library_type)
    sparse = str(ceres.sparse_linear_algebra_library_type)
    gpu_used = dense.endswith("CUDA") or sparse.endswith("CUDA_SPARSE")
    return dense, sparse, gpu_used


def _ba_summary_acceptance(summary: object) -> tuple[bool, str]:
    """Keep a finite improving solve that only exhausted its iteration budget."""
    termination = getattr(summary, "termination_type", None)
    termination_name = getattr(termination, "name", str(termination))
    if termination_name in {"CONVERGENCE", "USER_SUCCESS"}:
        return True, "converged"
    ceres = getattr(summary, "ceres_summary", summary)
    initial_cost = float(getattr(ceres, "initial_cost", float("nan")))
    final_cost = float(getattr(ceres, "final_cost", float("nan")))
    successful_steps = int(getattr(ceres, "num_successful_steps", 0))
    message = str(getattr(ceres, "message", ""))
    if (
        termination_name == "NO_CONVERGENCE"
        and "Maximum number of iterations reached" in message
        and np.isfinite(initial_cost)
        and np.isfinite(final_cost)
        and final_cost <= initial_cost
        and successful_steps > 0
    ):
        return True, "iteration-limit-cost-decrease"
    return False, "unusable-termination"


def refine_fixed_anchor_window(
    coarse: FixedAnchorWindowSolution,
    tracks: list[TriangulatedTrackState],
    intrinsics: Any,
    *,
    fixed_pose_ids: set[int],
    camera_model: str = "SIMPLE_PINHOLE",
    max_num_iterations: int = 50,
    refinement_passes: int = 1,
    linear_solver_policy: str = "auto",
    device_policy: str = "cuda-preferred",
    ceres_cuda_available: bool | None = None,
    previous_prior: FejPriorState | None = None,
    marginalize_pose_id: int | None = None,
    marginalization_residual_policy: str = "all-active",
    prior_device_policy: str = "cuda-preferred",
    prior_relative_rank_threshold: float = 1e-10,
    prior_maximum_condition_estimate: float | None = None,
    prior_expected_nullity: int | None = None,
) -> FixedAnchorLocalBaSolution:
    """Run local BA over one fixed-lag window with canonical overlap poses."""
    frame_ids = tuple(coarse.frame_ids)
    frame_id_set = set(frame_ids)
    if not frame_ids or set(coarse.rotations) != frame_id_set:
        raise FixedAnchorLocalBaError("coarse rotation identity is incomplete")
    if set(coarse.centers) != frame_id_set:
        raise FixedAnchorLocalBaError("coarse center identity is incomplete")
    if not fixed_pose_ids or fixed_pose_ids - frame_id_set:
        raise FixedAnchorLocalBaError("fixed pose identity is invalid")
    if max_num_iterations < 1:
        raise FixedAnchorLocalBaError("BA iteration limit is invalid")
    if not 1 <= refinement_passes <= 3:
        raise FixedAnchorLocalBaError("BA refinement pass count is invalid")
    if not tracks:
        raise FixedAnchorLocalBaError("triangulated track set is empty")
    if previous_prior is not None and set(previous_prior.camera_ids) & fixed_pose_ids:
        raise FixedAnchorLocalBaError("FEJ prior pose cannot also be fixed")
    if previous_prior is not None and set(previous_prior.camera_ids) - frame_id_set:
        raise FixedAnchorLocalBaError("FEJ prior pose is outside the window")
    if marginalize_pose_id is not None and (
        marginalize_pose_id not in frame_id_set
        or marginalize_pose_id in fixed_pose_ids
    ):
        raise FixedAnchorLocalBaError("marginalized pose identity is invalid")
    if marginalization_residual_policy not in {
        "all-active",
        "retiring-track-closure",
    }:
        raise FixedAnchorLocalBaError(
            "marginalization residual policy is invalid"
        )

    matrix_k = _shared_intrinsics_matrix(intrinsics)
    frame_uid_by_id: dict[int, str] = {}
    image_size_by_id: dict[int, tuple[int, int]] = {}
    for track in tracks:
        for observation in track.observations:
            frame_id = observation.geometry_ordinal
            if frame_id not in frame_id_set:
                raise FixedAnchorLocalBaError("track observation is outside the window")
            previous_uid = frame_uid_by_id.setdefault(frame_id, observation.frame_uid)
            if previous_uid != observation.frame_uid:
                raise FixedAnchorLocalBaError("frame UID is inconsistent")
            size = (observation.image_width, observation.image_height)
            previous_size = image_size_by_id.setdefault(frame_id, size)
            if previous_size != size:
                raise FixedAnchorLocalBaError("frame image size is inconsistent")
    if not image_size_by_id:
        raise FixedAnchorLocalBaError("track observations have no image size")
    fallback_size = next(iter(image_size_by_id.values()))

    build_started = time.perf_counter()
    reconstruction = pycolmap.Reconstruction()
    width, height = fallback_size
    camera = camera_from_intrinsics_matrix(
        matrix_k,
        camera_model=camera_model,
        width=width,
        height=height,
        camera_id=1,
    )
    reconstruction.add_camera_with_trivial_rig(camera)

    image_id_by_frame = {
        frame_id: index + 1 for index, frame_id in enumerate(frame_ids)
    }
    images: dict[int, pycolmap.Image] = {}
    for frame_id in frame_ids:
        image = pycolmap.Image()
        image.image_id = image_id_by_frame[frame_id]
        image.camera_id = 1
        image.name = frame_uid_by_id.get(frame_id, f"geometry-{frame_id}")
        images[frame_id] = image

    point_rows: list[tuple[str, np.ndarray, pycolmap.Track]] = []
    marginalize_track_uids: set[str] = set()
    observation_count = 0
    for track in tracks:
        colmap_track = pycolmap.Track()
        seen_frames: set[int] = set()
        for observation in track.observations:
            frame_id = observation.geometry_ordinal
            if frame_id in seen_frames:
                raise FixedAnchorLocalBaError(
                    "one track contains duplicate frame observations"
                )
            seen_frames.add(frame_id)
            image = images[frame_id]
            point2d_index = len(image.points2D)
            image.points2D.append(
                pycolmap.Point2D(
                    np.asarray((observation.x, observation.y), dtype=np.float64)
                )
            )
            colmap_track.add_element(image.image_id, point2d_index)
            observation_count += 1
        if len(seen_frames) < 2:
            continue
        if marginalize_pose_id is not None and marginalize_pose_id in seen_frames:
            marginalize_track_uids.add(track.track_uid)
        point_rows.append(
            (
                track.track_uid,
                np.asarray(track.xyz, dtype=np.float64).reshape(3, 1),
                colmap_track,
            )
        )

    for frame_id in frame_ids:
        rotation = np.asarray(coarse.rotations[frame_id], dtype=np.float64)
        center = np.asarray(coarse.centers[frame_id], dtype=np.float64)
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(rotation), -rotation @ center
        )
        reconstruction.add_image_with_trivial_frame(
            images[frame_id], cam_from_world
        )

    track_point3d_ids: dict[str, int] = {}
    for track_uid, xyz, colmap_track in point_rows:
        point3d_id = reconstruction.add_point3D(xyz, colmap_track)
        track_point3d_ids[track_uid] = int(point3d_id)
    marginalize_point3d_ids = tuple(
        sorted(track_point3d_ids[value] for value in marginalize_track_uids)
    )
    capture_point3d_ids = (
        marginalize_point3d_ids
        if marginalization_residual_policy == "retiring-track-closure"
        else None
    )
    build_wall = time.perf_counter() - build_started

    before_fixed_rotations = {
        frame_id: np.asarray(coarse.rotations[frame_id]).copy()
        for frame_id in fixed_pose_ids
    }
    before_fixed_centers = {
        frame_id: np.asarray(coarse.centers[frame_id]).copy()
        for frame_id in fixed_pose_ids
    }
    solve_started = time.perf_counter()
    summaries = []
    captured_linearization: list[CeresProblemLinearization] = []
    variable_image_ids = {
        frame_id: image_id_by_frame[frame_id]
        for frame_id in frame_ids
        if frame_id not in fixed_pose_ids
    }
    for pass_ordinal in range(refinement_passes):
        reconstruction, _, summary = bundle_adjustment(
            reconstruction,
            virtual_reconstruction=None,
            negative_depth_observations={},
            max_num_iterations=max_num_iterations,
            loss_type_normal="huber",
            linear_solver_type=linear_solver_policy,
            fixed_pose_ids={image_id_by_frame[value] for value in fixed_pose_ids},
            fix_intrinsics=True,
            device_policy=device_policy,
            ceres_cuda_available=ceres_cuda_available,
            fej_prior=previous_prior,
            fej_prior_image_ids=(
                {
                    frame_id: image_id_by_frame[frame_id]
                    for frame_id in previous_prior.camera_ids
                }
                if previous_prior is not None
                else None
            ),
            post_solve_problem_callback=(
                lambda problem, current: captured_linearization.append(
                    capture_ceres_problem_linearization(
                        problem,
                        current,
                        variable_image_ids,
                        point3d_ids=capture_point3d_ids,
                        residual_seed_point3d_ids=capture_point3d_ids,
                    )
                )
                if marginalize_pose_id is not None
                and pass_ordinal == refinement_passes - 1
                else None
            ),
        )
        summaries.append(summary)
    solve_wall = time.perf_counter() - solve_started
    next_prior = None
    if marginalize_pose_id is not None:
        if len(captured_linearization) != 1:
            raise FixedAnchorLocalBaError("Ceres linearization capture is incomplete")
        next_prior = marginalize_ceres_linearization(
            captured_linearization[0],
            eliminate_camera_id=marginalize_pose_id,
            previous_prior=previous_prior,
            device_policy=prior_device_policy,
            relative_rank_threshold=prior_relative_rank_threshold,
            maximum_condition_estimate=prior_maximum_condition_estimate,
            expected_nullity=prior_expected_nullity,
        )

    rotations: dict[int, np.ndarray] = {}
    centers: dict[int, np.ndarray] = {}
    for frame_id, image_id in image_id_by_frame.items():
        pose = reconstruction.images[image_id].cam_from_world()
        rotation = np.asarray(pose.rotation.matrix(), dtype=np.float64)
        translation = np.asarray(pose.translation, dtype=np.float64)
        rotations[frame_id] = rotation
        centers[frame_id] = -rotation.T @ translation

    maximum_fixed_rotation_delta = max(
        float(np.max(np.abs(rotations[key] - before_fixed_rotations[key])))
        for key in fixed_pose_ids
    )
    maximum_fixed_center_delta = max(
        float(np.max(np.abs(centers[key] - before_fixed_centers[key])))
        for key in fixed_pose_ids
    )
    resolved_backends = [_resolved_backend(summary) for summary in summaries]
    dense_backend, sparse_backend, gpu_used = resolved_backends[-1]
    if any(value != resolved_backends[0] for value in resolved_backends[1:]):
        raise FixedAnchorLocalBaError("BA backend changed across refinement passes")
    ceres = getattr(summaries[-1], "ceres_summary", summaries[-1])
    termination = getattr(summaries[-1], "termination_type", None)
    termination_names = [
        getattr(
            getattr(summary, "termination_type", None),
            "name",
            str(getattr(summary, "termination_type", None)),
        )
        for summary in summaries
    ]
    pass_acceptance = [_ba_summary_acceptance(summary) for summary in summaries]
    solve_passed = all(value[0] for value in pass_acceptance)
    ceres_solve_wall = sum(
        float(
            getattr(summary, "ceres_summary", summary).total_time_in_seconds
        )
        for summary in summaries
    )
    linearization_capture_wall = sum(
        float(value.report["captureWallSeconds"])
        for value in captured_linearization
    )
    report = {
        "contractId": "jarailsense.gluemap-fixed-anchor-local-ba/v1",
        "status": "passed" if solve_passed else "failed",
        "publishable": False,
        "diagnosticMode": "fixed-anchor-approximation",
        "requestedDevicePolicy": device_policy,
        "gpuUsed": gpu_used,
        "denseLinearAlgebraBackend": dense_backend,
        "sparseLinearAlgebraBackend": sparse_backend,
        "nativeThreadCount": resolve_native_thread_count(),
        "ceresThreadsGiven": int(ceres.num_threads_given),
        "ceresThreadsUsed": int(ceres.num_threads_used),
        "linearSolverPolicy": linear_solver_policy,
        "refinementPassCount": refinement_passes,
        "marginalizationResidualPolicy": marginalization_residual_policy,
        "frameCount": len(frame_ids),
        "fixedPoseCount": len(fixed_pose_ids),
        "trackCount": len(point_rows),
        "observationCount": observation_count,
        "buildWallSeconds": build_wall,
        "solveWallSeconds": solve_wall,
        "ceresSolveWallSeconds": ceres_solve_wall,
        "linearizationCaptureWallSeconds": linearization_capture_wall,
        "solveOrchestrationWallSeconds": max(
            0.0,
            solve_wall - ceres_solve_wall - linearization_capture_wall,
        ),
        "totalWallSeconds": build_wall + solve_wall,
        "termination": getattr(termination, "name", str(termination)),
        "solveAcceptance": pass_acceptance[-1][1],
        "degradedPassCount": sum(
            value[1] == "iteration-limit-cost-decrease"
            for value in pass_acceptance
        ),
        "initialCost": float(
            getattr(summaries[0], "ceres_summary", summaries[0]).initial_cost
        ),
        "finalCost": float(ceres.final_cost),
        "refinementPasses": [
            {
                "passOrdinal": index,
                "termination": termination_names[index],
                "solveAcceptance": pass_acceptance[index][1],
                "initialCost": float(
                    getattr(summary, "ceres_summary", summary).initial_cost
                ),
                "finalCost": float(
                    getattr(summary, "ceres_summary", summary).final_cost
                ),
                "ceresThreadsGiven": int(
                    getattr(summary, "ceres_summary", summary).num_threads_given
                ),
                "ceresThreadsUsed": int(
                    getattr(summary, "ceres_summary", summary).num_threads_used
                ),
            }
            for index, summary in enumerate(summaries)
        ],
        "maximumFixedRotationMatrixDelta": maximum_fixed_rotation_delta,
        "maximumFixedCenterDelta": maximum_fixed_center_delta,
        "prior": None if next_prior is None else next_prior.report,
    }
    return FixedAnchorLocalBaSolution(
        frame_ids=frame_ids,
        rotations=rotations,
        centers=centers,
        intrinsics=np.asarray(reconstruction.cameras[1].calibration_matrix()),
        track_point3d_ids=track_point3d_ids,
        reconstruction=reconstruction,
        next_prior=next_prior,
        report=report,
    )
