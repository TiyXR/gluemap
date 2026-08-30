"""Persistent Ceres problem for forward-only fixed-lag bundle adjustment."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyceres
import pycolmap
import pygluemap

from gluemap.estimators.augmented_bundle_adjustment import (
    _configure_ceres_cpu_concurrency,
    _pyceres_loss_function,
    _validate_resolved_ba_backend,
)
from gluemap.estimators.fixed_lag_prior import FejPriorState
from gluemap.estimators.fixed_lag_triangulation import TriangulatedTrackState


class PersistentFixedLagBaError(ValueError):
    """Raised when a persistent BA delta violates the active-window identity."""


@dataclass
class PersistentFixedLagBaSession:
    """Runner-owned, non-durable holder rebuilt after process recovery."""

    problem: "PersistentFixedLagBaProblem | None" = None


@dataclass
class _PoseBlock:
    values: np.ndarray
    manifold: object


@dataclass
class _ObservationBlock:
    frame_id: int
    xy: tuple[float, float]
    native_batch: object
    native_batch_index: int


@dataclass
class _PointBlock:
    point_id: int
    values: np.ndarray
    observations: dict[str, _ObservationBlock]


def _pose_ambient(rotation: Any, center: Any) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64)
    position = np.asarray(center, dtype=np.float64)
    if matrix.shape != (3, 3) or position.shape != (3,):
        raise PersistentFixedLagBaError("persistent BA pose shape is invalid")
    rigid = pycolmap.Rigid3d(
        pycolmap.Rotation3d(matrix), -matrix @ position
    )
    return np.asarray(rigid.params, dtype=np.float64).copy()


def _solver_configuration(
    problem: pyceres.Problem,
    *,
    frame_count: int,
    max_num_iterations: int,
    linear_solver_policy: str,
    device_policy: str,
    ceres_cuda_available: bool | None,
) -> tuple[pyceres.SolverOptions, int, bool]:
    options = pycolmap.BundleAdjustmentOptions()
    options.ceres.solver_options = pyceres.SolverOptions()
    options.ceres.solver_options.max_num_iterations = max_num_iterations
    requested_threads = _configure_ceres_cpu_concurrency(options.ceres)
    options.ceres.auto_select_solver_type = True
    solver_types = {
        "dense-schur": pyceres.LinearSolverType.DENSE_SCHUR,
        "sparse-schur": pyceres.LinearSolverType.SPARSE_SCHUR,
        "iterative-schur": pyceres.LinearSolverType.ITERATIVE_SCHUR,
    }
    if linear_solver_policy in solver_types:
        options.ceres.auto_select_solver_type = False
        options.ceres.solver_options.linear_solver_type = solver_types[
            linear_solver_policy
        ]
    elif linear_solver_policy != "auto":
        raise PersistentFixedLagBaError("persistent BA solver policy is invalid")
    if device_policy not in {"cuda-required", "cuda-preferred", "cpu"}:
        raise PersistentFixedLagBaError("persistent BA device policy is invalid")
    cuda_available = ceres_cuda_available is True
    if device_policy == "cuda-required" and not cuda_available:
        raise PersistentFixedLagBaError("persistent CUDA BA is unavailable")
    use_gpu = device_policy != "cpu" and cuda_available
    options.ceres.use_gpu = use_gpu

    config = pycolmap.BundleAdjustmentConfig()
    for image_id in range(1, frame_count + 1):
        config.add_image(image_id)
    solver_options = options.ceres.create_solver_options(config, problem)
    return solver_options, requested_threads, use_gpu


class PersistentFixedLagBaProblem:
    """Keep Ceres parameter/residual objects alive across sliding windows.

    Pixel measurements are immutable and keyed by observation UID. Pose,
    point, and visual residual blocks apply entering/leaving deltas. Each
    window submits its entering and leaving visual factors as native batches
    without per-row Python calls.
    """

    def __init__(self, *, camera_model_id: int, camera_params: Any) -> None:
        params = np.asarray(camera_params, dtype=np.float64).copy()
        if params.ndim != 1 or not np.isfinite(params).all():
            raise PersistentFixedLagBaError(
                "persistent BA camera parameters are invalid"
            )
        self.problem = pyceres.Problem()
        self.camera_model_id = int(camera_model_id)
        self.camera_params = params
        self.problem.add_parameter_block(self.camera_params, len(params))
        self.problem.set_parameter_block_constant(self.camera_params)
        self.loss_function = _pyceres_loss_function("huber")
        self.poses: dict[int, _PoseBlock] = {}
        self.points: dict[str, _PointBlock] = {}
        self._next_point_id = 1
        self._prior_residual_block: object | None = None
        self._prior_cost: object | None = None
        self._ordered_frame_ids: tuple[int, ...] = ()
        self._ordered_track_uids: tuple[str, ...] = ()

    def _remove_observation(
        self, track_uid: str, observation_uid: str
    ) -> _ObservationBlock:
        point = self.points[track_uid]
        return point.observations.pop(observation_uid)

    @staticmethod
    def _remove_native_observations(
        observations: list[_ObservationBlock],
    ) -> None:
        removals: dict[int, tuple[object, list[int]]] = {}
        for observation in observations:
            key = id(observation.native_batch)
            if key not in removals:
                removals[key] = (observation.native_batch, [])
            removals[key][1].append(observation.native_batch_index)
        for native_batch, indices in removals.values():
            native_batch.remove_indices(np.asarray(indices, dtype=np.int64))

    def _validate_observation(self, observation: Any) -> None:
        frame_id = int(observation.geometry_ordinal)
        if frame_id not in self.poses:
            raise PersistentFixedLagBaError(
                "persistent BA observation pose is absent"
            )

    def synchronize(
        self,
        *,
        frame_ids: tuple[int, ...],
        rotations: dict[int, np.ndarray],
        centers: dict[int, np.ndarray],
        fixed_pose_ids: set[int],
        tracks: list[TriangulatedTrackState],
        camera_model_id: int,
        camera_params: Any,
    ) -> dict[str, Any]:
        """Apply one active-window delta without changing factor semantics."""
        started = time.perf_counter()
        if self._prior_residual_block is not None:
            raise PersistentFixedLagBaError(
                "persistent BA prior must be removed before synchronization"
            )
        if int(camera_model_id) != self.camera_model_id:
            raise PersistentFixedLagBaError("persistent BA camera model changed")
        incoming_camera = np.asarray(camera_params, dtype=np.float64)
        if incoming_camera.shape != self.camera_params.shape or not np.array_equal(
            incoming_camera, self.camera_params
        ):
            raise PersistentFixedLagBaError(
                "persistent BA frozen intrinsics changed"
            )
        target_frames = set(int(value) for value in frame_ids)
        if (
            not target_frames
            or target_frames != set(rotations)
            or target_frames != set(centers)
            or not fixed_pose_ids
            or fixed_pose_ids - target_frames
        ):
            raise PersistentFixedLagBaError(
                "persistent BA active pose identity is invalid"
            )
        track_by_uid = {str(value.track_uid): value for value in tracks}
        if len(track_by_uid) != len(tracks) or not track_by_uid:
            raise PersistentFixedLagBaError(
                "persistent BA track identity is invalid"
            )
        target_observations: dict[tuple[str, str], Any] = {}
        for track_uid, track in track_by_uid.items():
            for observation in track.observations:
                observation_uid = str(observation.observation_uid)
                key = (track_uid, observation_uid)
                if (
                    key in target_observations
                    or int(observation.geometry_ordinal) not in target_frames
                ):
                    raise PersistentFixedLagBaError(
                        "persistent BA observation identity is invalid"
                    )
                target_observations[key] = observation

        removed_observations = 0
        native_removals: list[_ObservationBlock] = []
        for track_uid, point in list(self.points.items()):
            for observation_uid in list(point.observations):
                if (track_uid, observation_uid) not in target_observations:
                    native_removals.append(
                        self._remove_observation(track_uid, observation_uid)
                    )
                    removed_observations += 1
        native_batch_started = time.perf_counter()
        self._remove_native_observations(native_removals)

        removed_points = 0
        for track_uid in list(self.points):
            if track_uid not in track_by_uid:
                if self.points[track_uid].observations:
                    raise PersistentFixedLagBaError(
                        "persistent BA point still owns observations"
                    )
                self.problem.remove_parameter_block(self.points[track_uid].values)
                del self.points[track_uid]
                removed_points += 1

        removed_poses = 0
        for frame_id in list(self.poses):
            if frame_id not in target_frames:
                self.problem.remove_parameter_block(self.poses[frame_id].values)
                del self.poses[frame_id]
                removed_poses += 1

        created_poses = 0
        for frame_id in frame_ids:
            values = _pose_ambient(rotations[frame_id], centers[frame_id])
            pose = self.poses.get(frame_id)
            if pose is None:
                manifold = pygluemap.CreatePoseManifold()
                self.problem.add_parameter_block(values, 7, manifold)
                pose = _PoseBlock(values=values, manifold=manifold)
                self.poses[frame_id] = pose
                created_poses += 1
            else:
                pose.values[:] = values
            if frame_id in fixed_pose_ids:
                self.problem.set_parameter_block_constant(pose.values)
            else:
                self.problem.set_parameter_block_variable(pose.values)

        created_points = 0
        reused_points = 0
        for track_uid, track in track_by_uid.items():
            values = np.asarray(track.xyz, dtype=np.float64)
            if values.shape != (3,) or not np.isfinite(values).all():
                raise PersistentFixedLagBaError(
                    "persistent BA point value is invalid"
                )
            point = self.points.get(track_uid)
            if point is None:
                point = _PointBlock(
                    point_id=self._next_point_id,
                    values=values.copy(),
                    observations={},
                )
                self._next_point_id += 1
                self.problem.add_parameter_block(point.values, 3)
                self.points[track_uid] = point
                created_points += 1
            else:
                point.values[:] = values
                reused_points += 1

        created_observations = 0
        reused_observations = 0
        entering_observations: list[tuple[str, str, Any]] = []
        for (track_uid, observation_uid), observation in target_observations.items():
            existing = self.points[track_uid].observations.get(observation_uid)
            xy = (float(observation.x), float(observation.y))
            frame_id = int(observation.geometry_ordinal)
            if existing is not None and (
                existing.frame_id != frame_id or existing.xy != xy
            ):
                removed = self._remove_observation(track_uid, observation_uid)
                self._remove_native_observations([removed])
                existing = None
                removed_observations += 1
            if existing is None:
                self._validate_observation(observation)
                entering_observations.append(
                    (track_uid, observation_uid, observation)
                )
                created_observations += 1
            else:
                reused_observations += 1

        point_addresses = np.fromiter(
            (
                self.points[track_uid].values.ctypes.data
                for track_uid, _, _ in entering_observations
            ),
            dtype=np.uint64,
            count=len(entering_observations),
        )
        pose_addresses = np.fromiter(
            (
                self.poses[int(observation.geometry_ordinal)].values.ctypes.data
                for _, _, observation in entering_observations
            ),
            dtype=np.uint64,
            count=len(entering_observations),
        )
        observation_xy = np.asarray(
            [
                (float(observation.x), float(observation.y))
                for _, _, observation in entering_observations
            ],
            dtype=np.float64,
        )
        if entering_observations:
            native_batch = pygluemap.add_reprojection_residual_batch(
                self.problem,
                self.camera_model_id,
                point_addresses,
                pose_addresses,
                int(self.camera_params.ctypes.data),
                observation_xy,
                self.loss_function,
            )
            if native_batch.size != len(entering_observations):
                raise PersistentFixedLagBaError(
                    "persistent BA native visual batch size differs"
                )
            for batch_index, (
                track_uid,
                observation_uid,
                observation,
            ) in enumerate(entering_observations):
                self.points[track_uid].observations[observation_uid] = (
                    _ObservationBlock(
                        frame_id=int(observation.geometry_ordinal),
                        xy=(float(observation.x), float(observation.y)),
                        native_batch=native_batch,
                        native_batch_index=batch_index,
                    )
                )
        native_batch_wall_seconds = time.perf_counter() - native_batch_started
        self._ordered_frame_ids = tuple(int(value) for value in frame_ids)
        self._ordered_track_uids = tuple(track_by_uid)

        return {
            "status": "passed",
            "mode": "persistent-delta",
            "visualResidualBindingMode": "native-enter-leave-delta",
            "createdPoseCount": created_poses,
            "removedPoseCount": removed_poses,
            "createdPointCount": created_points,
            "reusedPointCount": reused_points,
            "removedPointCount": removed_points,
            "createdObservationCount": created_observations,
            "reusedObservationCount": reused_observations,
            "removedObservationCount": removed_observations,
            "residentPoseCount": len(self.poses),
            "residentPointCount": len(self.points),
            "residentObservationCount": sum(
                len(value.observations) for value in self.points.values()
            ),
            "problemParameterBlockCount": self.problem.num_parameter_blocks(),
            "problemResidualBlockCount": self.problem.num_residual_blocks(),
            "problemResidualCount": self.problem.num_residuals(),
            "nativeVisualBatchWallSeconds": native_batch_wall_seconds,
            "wallSeconds": time.perf_counter() - started,
        }

    def add_prior(self, prior: FejPriorState | None) -> None:
        if prior is None:
            return
        if self._prior_residual_block is not None:
            raise PersistentFixedLagBaError("persistent BA prior is already active")
        if any(camera_id not in self.poses for camera_id in prior.camera_ids):
            raise PersistentFixedLagBaError(
                "persistent BA prior pose is absent"
            )
        cost = pygluemap.CreateFejPosePriorCost(
            prior.factor.detach().cpu().numpy(),
            prior.factor_residual.detach().cpu().numpy(),
            prior.linearization_points.detach().cpu().numpy(),
        )
        self._prior_residual_block = self.problem.add_residual_block(
            cost,
            None,
            [self.poses[value].values for value in prior.camera_ids],
        )
        self._prior_cost = cost

    def remove_prior(self) -> None:
        if self._prior_residual_block is None:
            return
        self.problem.remove_residual_block(self._prior_residual_block)
        self._prior_residual_block = None
        self._prior_cost = None

    def solve(
        self,
        *,
        max_num_iterations: int,
        linear_solver_policy: str,
        device_policy: str,
        ceres_cuda_available: bool | None,
    ) -> tuple[pyceres.SolverSummary, dict[str, Any]]:
        started = time.perf_counter()
        options, requested_threads, use_gpu = _solver_configuration(
            self.problem,
            frame_count=len(self.poses),
            max_num_iterations=max_num_iterations,
            linear_solver_policy=linear_solver_policy,
            device_policy=device_policy,
            ceres_cuda_available=ceres_cuda_available,
        )
        summary = pyceres.SolverSummary()
        point_addresses = np.fromiter(
            (
                self.points[value].values.ctypes.data
                for value in self._ordered_track_uids
            ),
            dtype=np.uint64,
            count=len(self._ordered_track_uids),
        )
        pose_addresses = np.fromiter(
            (
                self.poses[value].values.ctypes.data
                for value in self._ordered_frame_ids
            ),
            dtype=np.uint64,
            count=len(self._ordered_frame_ids),
        )
        pygluemap.solve_with_ba_ordering(
            options,
            self.problem,
            summary,
            point_addresses,
            pose_addresses,
            int(self.camera_params.ctypes.data),
            use_gpu,
        )
        _validate_resolved_ba_backend(summary, use_gpu)
        return summary, {
            "status": "passed",
            "requestedThreadCount": requested_threads,
            "gpuRequested": use_gpu,
            "linearSolverOrdering": "points-group-0-poses-group-1",
            "orderedPointCount": len(self._ordered_track_uids),
            "orderedPoseCount": len(self._ordered_frame_ids),
            "wallSeconds": time.perf_counter() - started,
        }

    def pose_values(self, frame_id: int) -> np.ndarray:
        return self.poses[frame_id].values

    def point_values(self, track_uid: str) -> np.ndarray:
        return self.points[track_uid].values

    def point_id(self, track_uid: str) -> int:
        return self.points[track_uid].point_id

    def pose_parameter_blocks(self, frame_ids: tuple[int, ...]) -> list[np.ndarray]:
        return [self.poses[value].values for value in frame_ids]

    def point_parameter_blocks(
        self, track_uids: tuple[str, ...]
    ) -> list[np.ndarray]:
        return [self.points[value].values for value in track_uids]
