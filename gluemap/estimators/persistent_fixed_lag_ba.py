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
    _pyceres_loss_function,
    _validate_resolved_ba_backend,
)
from gluemap.estimators.fixed_lag_prior import FejPriorState
from gluemap.estimators.fixed_lag_triangulation import TriangulatedTrackState
from gluemap.utils.runtime_capacity import resolve_native_thread_count


class PersistentFixedLagBaError(ValueError):
    """Raised when a persistent BA delta violates the active-window identity."""


AUTO_DENSE_SCHUR_MAXIMUM_CAMERAS = 100


@dataclass
class PersistentFixedLagBaSession:
    """Runner-owned, non-durable holder rebuilt after process recovery."""

    policy: str = "persistent-delta"
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
    if frame_count < 1 or problem.num_parameter_blocks() < 1:
        raise PersistentFixedLagBaError("persistent BA problem is empty")
    solver_options = pyceres.SolverOptions()
    solver_options.max_num_iterations = max_num_iterations
    requested_threads = resolve_native_thread_count()
    solver_options.num_threads = requested_threads
    solver_types = {
        "dense-schur": pyceres.LinearSolverType.DENSE_SCHUR,
        "sparse-schur": pyceres.LinearSolverType.SPARSE_SCHUR,
        "iterative-schur": pyceres.LinearSolverType.ITERATIVE_SCHUR,
    }
    if linear_solver_policy in solver_types:
        solver_options.linear_solver_type = solver_types[linear_solver_policy]
    elif linear_solver_policy == "auto":
        # Ceres recommends dense Schur for BA frontiers of roughly one hundred
        # cameras or fewer.  The forward-only railway window normally stays in
        # that range even though the complete video contains thousands of
        # cameras.  Switch on the active frontier, not the full-video size.
        solver_options.linear_solver_type = (
            pyceres.LinearSolverType.DENSE_SCHUR
            if frame_count <= AUTO_DENSE_SCHUR_MAXIMUM_CAMERAS
            else pyceres.LinearSolverType.SPARSE_SCHUR
        )
    elif linear_solver_policy != "auto":
        raise PersistentFixedLagBaError("persistent BA solver policy is invalid")
    if device_policy not in {"cuda-required", "cuda-preferred", "cpu"}:
        raise PersistentFixedLagBaError("persistent BA device policy is invalid")
    cuda_available = ceres_cuda_available is True
    if device_policy == "cuda-required" and not cuda_available:
        raise PersistentFixedLagBaError("persistent CUDA BA is unavailable")
    use_gpu = device_policy != "cpu" and cuda_available
    return solver_options, requested_threads, use_gpu


class PersistentFixedLagBaProblem:
    """Keep Ceres parameter/residual objects alive across sliding windows.

    Pixel measurements are immutable and keyed by observation UID. Pose,
    point, and visual residual blocks apply entering/leaving deltas. Each
    window submits its entering and leaving visual factors as native batches
    without per-row Python calls.
    """

    def __init__(
        self,
        *,
        camera_model_id: int,
        camera_params: Any,
        policy: str = "persistent-delta",
    ) -> None:
        params = np.asarray(camera_params, dtype=np.float64).copy()
        if params.ndim != 1 or not np.isfinite(params).all():
            raise PersistentFixedLagBaError(
                "persistent BA camera parameters are invalid"
            )
        self.problem = pyceres.Problem()
        if policy not in {"persistent-delta", "native-rebuild-every-window"}:
            raise PersistentFixedLagBaError("persistent BA policy is invalid")
        self.policy = policy
        self.camera_model_id = int(camera_model_id)
        self.camera_params = params
        if policy == "persistent-delta":
            self.problem.add_parameter_block(self.camera_params, len(params))
            self.problem.set_parameter_block_constant(self.camera_params)
        self.loss_function = _pyceres_loss_function("huber")
        self.poses: dict[int, _PoseBlock] = {}
        self.points: dict[str, _PointBlock] = {}
        self._next_point_id = 1
        self._prior_residual_block: object | None = None
        self._prior_cost: object | None = None
        self._native_batches: list[object] = []
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
        phase_wall_seconds: dict[str, float] = {}
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
        native_rebuild = self.policy == "native-rebuild-every-window"
        if native_rebuild and (
            self.poses
            or self.points
            or self.problem.num_parameter_blocks()
            or self.problem.num_residual_blocks()
        ):
            raise PersistentFixedLagBaError(
                "native rebuild BA problem must start empty"
            )
        phase_wall_seconds["identityValidation"] = time.perf_counter() - started
        phase_started = time.perf_counter()
        target_observations: dict[tuple[str, str], Any] = {}
        if not native_rebuild:
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
        phase_wall_seconds["targetObservationBuild"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        removed_observations = 0
        native_removals: list[_ObservationBlock] = []
        if not native_rebuild:
            for track_uid, point in list(self.points.items()):
                for observation_uid in list(point.observations):
                    if (track_uid, observation_uid) not in target_observations:
                        native_removals.append(
                            self._remove_observation(track_uid, observation_uid)
                        )
                        removed_observations += 1
        phase_wall_seconds["removalScan"] = time.perf_counter() - phase_started
        native_batch_started = time.perf_counter()
        phase_started = time.perf_counter()
        self._remove_native_observations(native_removals)
        phase_wall_seconds["nativeRemoval"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
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
                if self.policy == "persistent-delta":
                    self.problem.add_parameter_block(values, 7, manifold)
                pose = _PoseBlock(values=values, manifold=manifold)
                self.poses[frame_id] = pose
                created_poses += 1
            else:
                pose.values[:] = values
            if self.policy == "persistent-delta":
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
                if self.policy == "persistent-delta":
                    self.problem.add_parameter_block(point.values, 3)
                self.points[track_uid] = point
                created_points += 1
            else:
                point.values[:] = values
                reused_points += 1
        phase_wall_seconds["posePointBuild"] = time.perf_counter() - phase_started

        phase_started = time.perf_counter()
        created_observations = 0
        reused_observations = 0
        entering_observations: list[tuple[str, str, Any]] = []
        native_csr_rebuild = native_rebuild and hasattr(
            pygluemap,
            "add_reprojection_residual_csr_implicit_parameters",
        )
        if native_csr_rebuild:
            observation_count = sum(
                len(track.observations) for track in track_by_uid.values()
            )
            if observation_count < 1:
                raise PersistentFixedLagBaError(
                    "persistent BA active observations are empty"
                )
            created_observations = observation_count
        elif native_rebuild:
            frame_order = {
                int(frame_id): ordinal
                for ordinal, frame_id in enumerate(frame_ids)
            }
            frame_buckets: list[list[tuple[str, str, Any]]] = [
                [] for _ in frame_ids
            ]
            for track_uid, track in track_by_uid.items():
                seen_uids: set[str] = set()
                for observation in track.observations:
                    observation_uid = str(observation.observation_uid)
                    frame_id = int(observation.geometry_ordinal)
                    if (
                        observation_uid in seen_uids
                        or frame_id not in frame_order
                    ):
                        raise PersistentFixedLagBaError(
                            "persistent BA observation identity is invalid"
                        )
                    seen_uids.add(observation_uid)
                    self._validate_observation(observation)
                    frame_buckets[frame_order[frame_id]].append(
                        (track_uid, observation_uid, observation)
                    )
            entering_observations = [
                value for bucket in frame_buckets for value in bucket
            ]
            created_observations = len(entering_observations)
        else:
            for (
                track_uid,
                observation_uid,
            ), observation in target_observations.items():
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
        phase_wall_seconds["enteringObservationBuild"] = (
            time.perf_counter() - phase_started
        )

        phase_started = time.perf_counter()
        if native_csr_rebuild:
            frame_order = {
                int(frame_id): ordinal
                for ordinal, frame_id in enumerate(frame_ids)
            }
            point_addresses = np.fromiter(
                (
                    self.points[track_uid].values.ctypes.data
                    for track_uid in track_by_uid
                ),
                dtype=np.uint64,
                count=len(track_by_uid),
            )
            pose_addresses = np.fromiter(
                (
                    self.poses[int(frame_id)].values.ctypes.data
                    for frame_id in frame_ids
                ),
                dtype=np.uint64,
                count=len(frame_ids),
            )
            fixed_pose_flags = np.fromiter(
                (int(frame_id) in fixed_pose_ids for frame_id in frame_ids),
                dtype=np.uint8,
                count=len(frame_ids),
            )
            track_offsets = np.empty((len(track_by_uid) + 1,), dtype=np.uint64)
            observation_frame_indices = np.empty(
                (observation_count,), dtype=np.int64
            )
            observation_xy = np.empty(
                (observation_count, 2), dtype=np.float64
            )
            observation_index = 0
            for track_index, track in enumerate(track_by_uid.values()):
                track_offsets[track_index] = observation_index
                seen_uids: set[str] = set()
                for observation in track.observations:
                    observation_uid = str(observation.observation_uid)
                    frame_id = int(observation.geometry_ordinal)
                    if observation_uid in seen_uids or frame_id not in frame_order:
                        raise PersistentFixedLagBaError(
                            "persistent BA observation identity is invalid"
                        )
                    seen_uids.add(observation_uid)
                    self._validate_observation(observation)
                    observation_frame_indices[observation_index] = frame_order[
                        frame_id
                    ]
                    observation_xy[observation_index, 0] = float(observation.x)
                    observation_xy[observation_index, 1] = float(observation.y)
                    observation_index += 1
            track_offsets[len(track_by_uid)] = observation_index
            if observation_index != observation_count:
                raise PersistentFixedLagBaError(
                    "persistent BA CSR observation count changed"
                )
        else:
            observation_count = len(entering_observations)
            point_addresses = np.empty((observation_count,), dtype=np.uint64)
            pose_addresses = np.empty((observation_count,), dtype=np.uint64)
            observation_xy = np.empty((observation_count, 2), dtype=np.float64)
            fixed_pose_flags = np.empty((observation_count,), dtype=np.uint8)
            for index, (track_uid, _, observation) in enumerate(
                entering_observations
            ):
                frame_id = int(observation.geometry_ordinal)
                point_addresses[index] = self.points[
                    track_uid
                ].values.ctypes.data
                pose_addresses[index] = self.poses[frame_id].values.ctypes.data
                observation_xy[index, 0] = float(observation.x)
                observation_xy[index, 1] = float(observation.y)
                fixed_pose_flags[index] = frame_id in fixed_pose_ids
        phase_wall_seconds["nativeArrayBuild"] = time.perf_counter() - phase_started
        if observation_count:
            phase_started = time.perf_counter()
            if native_csr_rebuild:
                native_batch = (
                    pygluemap.add_reprojection_residual_csr_implicit_parameters(
                        self.problem,
                        self.camera_model_id,
                        point_addresses,
                        pose_addresses,
                        fixed_pose_flags,
                        track_offsets,
                        observation_frame_indices,
                        int(self.camera_params.ctypes.data),
                        observation_xy,
                        self.loss_function,
                    )
                )
            elif self.policy == "native-rebuild-every-window":
                native_batch = (
                    pygluemap.add_reprojection_residual_batch_implicit_parameters(
                        self.problem,
                        self.camera_model_id,
                        point_addresses,
                        pose_addresses,
                        fixed_pose_flags,
                        int(self.camera_params.ctypes.data),
                        observation_xy,
                        self.loss_function,
                    )
                )
            else:
                native_batch = pygluemap.add_reprojection_residual_batch(
                    self.problem,
                    self.camera_model_id,
                    point_addresses,
                    pose_addresses,
                    int(self.camera_params.ctypes.data),
                    observation_xy,
                    self.loss_function,
                )
            if native_batch.size != observation_count:
                raise PersistentFixedLagBaError(
                    "persistent BA native visual batch size differs"
                )
            phase_wall_seconds["nativeResidualBatch"] = (
                time.perf_counter() - phase_started
            )
            phase_started = time.perf_counter()
            if native_rebuild:
                self._native_batches.append(native_batch)
            else:
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
            phase_wall_seconds["observationBookkeeping"] = (
                time.perf_counter() - phase_started
            )
        else:
            phase_wall_seconds["nativeResidualBatch"] = 0.0
            phase_wall_seconds["observationBookkeeping"] = 0.0
        phase_started = time.perf_counter()
        if self.policy == "native-rebuild-every-window":
            if not self.problem.has_parameter_block(self.camera_params):
                raise PersistentFixedLagBaError(
                    "native rebuild camera parameter is absent"
                )
            self.problem.set_parameter_block_constant(self.camera_params)
            for frame_id in frame_ids:
                pose = self.poses[frame_id]
                if not self.problem.has_parameter_block(pose.values):
                    if frame_id in fixed_pose_ids:
                        continue
                    raise PersistentFixedLagBaError(
                        "native rebuild variable pose parameter is absent"
                    )
                self.problem.set_manifold(pose.values, pose.manifold)
                self.problem.set_parameter_block_variable(pose.values)
        phase_wall_seconds["parameterFinalize"] = time.perf_counter() - phase_started
        native_batch_wall_seconds = time.perf_counter() - native_batch_started
        self._ordered_frame_ids = tuple(int(value) for value in frame_ids)
        self._ordered_track_uids = tuple(track_by_uid)
        wall_seconds = time.perf_counter() - started
        phase_wall_seconds["other"] = max(
            0.0, wall_seconds - sum(phase_wall_seconds.values())
        )

        return {
            "status": "passed",
            "mode": self.policy,
            "visualResidualBindingMode": (
                "native-image-major-csr-implicit-parameters"
                if native_csr_rebuild
                else (
                    "native-image-major-implicit-parameters"
                    if native_rebuild
                    else "native-enter-leave-delta"
                )
            ),
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
            ) if not native_rebuild else self.problem.num_residual_blocks(),
            "problemParameterBlockCount": self.problem.num_parameter_blocks(),
            "problemResidualBlockCount": self.problem.num_residual_blocks(),
            "problemResidualCount": self.problem.num_residuals(),
            "nativeVisualBatchWallSeconds": native_batch_wall_seconds,
            "phaseWallSeconds": phase_wall_seconds,
            "wallSeconds": wall_seconds,
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
        linear_solver_ordering_policy: str,
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
        if linear_solver_ordering_policy == "auto":
            if use_gpu:
                pygluemap.solve_cuda(options, self.problem, summary)
            else:
                pyceres.solve(options, self.problem, summary)
        elif linear_solver_ordering_policy == "point-first":
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
                    if self.problem.has_parameter_block(
                        self.poses[value].values
                    )
                ),
                dtype=np.uint64,
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
        else:
            raise PersistentFixedLagBaError(
                "persistent BA ordering policy is invalid"
            )
        _validate_resolved_ba_backend(summary, use_gpu)
        return summary, {
            "status": "passed",
            "requestedThreadCount": requested_threads,
            "gpuRequested": use_gpu,
            "linearSolverOrdering": linear_solver_ordering_policy,
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
