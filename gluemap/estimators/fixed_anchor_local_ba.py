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


def refine_fixed_anchor_window(
    coarse: FixedAnchorWindowSolution,
    tracks: list[TriangulatedTrackState],
    intrinsics: Any,
    *,
    fixed_pose_ids: set[int],
    camera_model: str = "SIMPLE_PINHOLE",
    max_num_iterations: int = 50,
    device_policy: str = "cuda-preferred",
    ceres_cuda_available: bool | None = None,
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
    if not tracks:
        raise FixedAnchorLocalBaError("triangulated track set is empty")

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
    reconstruction, _, summary = bundle_adjustment(
        reconstruction,
        virtual_reconstruction=None,
        negative_depth_observations={},
        max_num_iterations=max_num_iterations,
        loss_type_normal="huber",
        linear_solver_type="auto",
        fixed_pose_ids={image_id_by_frame[value] for value in fixed_pose_ids},
        fix_intrinsics=True,
        device_policy=device_policy,
        ceres_cuda_available=ceres_cuda_available,
    )
    solve_wall = time.perf_counter() - solve_started

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
    dense_backend, sparse_backend, gpu_used = _resolved_backend(summary)
    ceres = getattr(summary, "ceres_summary", summary)
    termination = getattr(summary, "termination_type", None)
    report = {
        "contractId": "jarailsense.gluemap-fixed-anchor-local-ba/v1",
        "status": "passed",
        "publishable": False,
        "diagnosticMode": "fixed-anchor-approximation",
        "requestedDevicePolicy": device_policy,
        "gpuUsed": gpu_used,
        "denseLinearAlgebraBackend": dense_backend,
        "sparseLinearAlgebraBackend": sparse_backend,
        "nativeThreadCount": resolve_native_thread_count(),
        "frameCount": len(frame_ids),
        "fixedPoseCount": len(fixed_pose_ids),
        "trackCount": len(point_rows),
        "observationCount": observation_count,
        "buildWallSeconds": build_wall,
        "solveWallSeconds": solve_wall,
        "totalWallSeconds": build_wall + solve_wall,
        "termination": getattr(termination, "name", str(termination)),
        "initialCost": float(ceres.initial_cost),
        "finalCost": float(ceres.final_cost),
        "maximumFixedRotationMatrixDelta": maximum_fixed_rotation_delta,
        "maximumFixedCenterDelta": maximum_fixed_center_delta,
    }
    return FixedAnchorLocalBaSolution(
        frame_ids=frame_ids,
        rotations=rotations,
        centers=centers,
        intrinsics=np.asarray(reconstruction.cameras[1].calibration_matrix()),
        track_point3d_ids=track_point3d_ids,
        reconstruction=reconstruction,
        report=report,
    )
