"""Advance the A11 fixed-anchor approximation one keyframe at a time."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from gluemap.estimators.active_track_store import SelectedTrackState
from gluemap.estimators.fixed_anchor_approximation import (
    FixedAnchorApproximationSolver,
    FixedAnchorWindowSolution,
)
from gluemap.estimators.fixed_anchor_local_ba import (
    FixedAnchorLocalBaSolution,
    refine_fixed_anchor_window,
)
from gluemap.estimators.fixed_lag_triangulation import (
    TriangulatedTrackState,
    triangulate_selected_tracks,
)


class FixedAnchorDiagnosticRunnerError(ValueError):
    """Raised when the A11 diagnostic window cannot advance continuously."""


@dataclass(frozen=True)
class FixedAnchorDiagnosticStep:
    window_ordinal: int
    coarse: FixedAnchorWindowSolution
    triangulated_tracks: tuple[TriangulatedTrackState, ...]
    refined: FixedAnchorLocalBaSolution
    report: dict[str, Any]


@dataclass(frozen=True)
class _RunnerPoseState:
    frame_ids: tuple[int, ...]
    rotations: dict[int, np.ndarray]
    centers: dict[int, np.ndarray]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _resolve_shared_intrinsics(values: list[Any]) -> np.ndarray:
    matrices = [
        np.asarray(value, dtype=np.float64).squeeze()
        for value in values
        if value is not None
    ]
    if not matrices or any(matrix.shape != (3, 3) for matrix in matrices):
        raise FixedAnchorDiagnosticRunnerError(
            "coarse shared intrinsics are incomplete"
        )
    first = matrices[0]
    if any(not np.allclose(value, first, rtol=0, atol=1e-9) for value in matrices[1:]):
        raise FixedAnchorDiagnosticRunnerError(
            "coarse window contains multiple intrinsics groups"
        )
    return first


class FixedAnchorDiagnosticRunner:
    """Own A11 warm-start state while preserving one canonical gauge."""

    def __init__(
        self,
        *,
        initial_anchor_frame_ids: set[int],
        coarse_solver: FixedAnchorApproximationSolver | Any | None = None,
        camera_model: str = "SIMPLE_PINHOLE",
        triangulation_device_policy: str = "cuda-required",
        triangulation_microbatch_tracks: int = 4096,
        triangulation_solver_policy: str = "homogeneous-svd",
        ba_device_policy: str = "cuda-preferred",
        ba_linear_solver_policy: str = "auto",
        ba_max_iterations: int = 20,
        ba_refinement_passes: int = 1,
        ceres_cuda_available: bool | None = None,
    ) -> None:
        if len(initial_anchor_frame_ids) < 2:
            raise FixedAnchorDiagnosticRunnerError(
                "initial anchor must contain at least two frames"
            )
        self.initial_anchor_frame_ids = set(initial_anchor_frame_ids)
        self.coarse_solver = coarse_solver or FixedAnchorApproximationSolver()
        self.camera_model = camera_model
        self.triangulation_device_policy = triangulation_device_policy
        self.triangulation_microbatch_tracks = triangulation_microbatch_tracks
        self.triangulation_solver_policy = triangulation_solver_policy
        self.ba_device_policy = ba_device_policy
        self.ba_linear_solver_policy = ba_linear_solver_policy
        self.ba_max_iterations = ba_max_iterations
        self.ba_refinement_passes = ba_refinement_passes
        self.ceres_cuda_available = ceres_cuda_available
        self._previous: _RunnerPoseState | None = None
        self._frozen_intrinsics: np.ndarray | None = None
        self._next_window_ordinal = 0

    @property
    def next_window_ordinal(self) -> int:
        return self._next_window_ordinal

    def advance(
        self,
        predictions: dict[str, list[Any]],
        frame_ids: list[int],
        selected_tracks: list[SelectedTrackState],
    ) -> FixedAnchorDiagnosticStep:
        if len(frame_ids) < 3 or frame_ids != sorted(set(frame_ids)):
            raise FixedAnchorDiagnosticRunnerError(
                "diagnostic frame ids must be sorted and unique"
            )
        frame_id_set = set(frame_ids)
        started = time.perf_counter()
        previous = self._previous
        if previous is None:
            fixed_pose_ids: set[int] = set()
            local_ba_fixed_ids = self.initial_anchor_frame_ids & frame_id_set
            if local_ba_fixed_ids != self.initial_anchor_frame_ids:
                raise FixedAnchorDiagnosticRunnerError(
                    "first window does not contain the frozen initial anchor"
                )
            initial_rotations = None
            initial_centers = None
        else:
            fixed_pose_ids = set(previous.frame_ids) & frame_id_set
            if len(fixed_pose_ids) < 2:
                raise FixedAnchorDiagnosticRunnerError(
                    "successive windows do not retain two fixed poses"
                )
            local_ba_fixed_ids = fixed_pose_ids
            initial_rotations = previous.rotations
            initial_centers = previous.centers

        coarse_started = time.perf_counter()
        coarse = self.coarse_solver.solve(
            predictions,
            frame_ids,
            initial_rotations=initial_rotations,
            initial_centers=initial_centers,
            fixed_pose_ids=fixed_pose_ids,
        )
        coarse_wall = time.perf_counter() - coarse_started
        if self._frozen_intrinsics is None:
            self._frozen_intrinsics = _resolve_shared_intrinsics(coarse.intrinsics)
        matrix_k = self._frozen_intrinsics

        triangulation_started = time.perf_counter()
        triangulated, triangulation_report = triangulate_selected_tracks(
            selected_tracks,
            coarse.rotations,
            coarse.centers,
            matrix_k,
            device_policy=self.triangulation_device_policy,
            microbatch_tracks=self.triangulation_microbatch_tracks,
            solver_policy=self.triangulation_solver_policy,
        )
        triangulation_wall = time.perf_counter() - triangulation_started

        ba_started = time.perf_counter()
        refined = refine_fixed_anchor_window(
            coarse,
            triangulated,
            matrix_k,
            fixed_pose_ids=local_ba_fixed_ids,
            camera_model=self.camera_model,
            max_num_iterations=self.ba_max_iterations,
            refinement_passes=self.ba_refinement_passes,
            linear_solver_policy=self.ba_linear_solver_policy,
            device_policy=self.ba_device_policy,
            ceres_cuda_available=self.ceres_cuda_available,
        )
        if refined.report["status"] != "passed":
            raise FixedAnchorDiagnosticRunnerError(
                "local BA did not reach an accepted termination"
            )
        ba_wall = time.perf_counter() - ba_started

        overlap_rotation_delta = 0.0
        overlap_center_delta = 0.0
        if previous is not None:
            for frame_id in fixed_pose_ids:
                overlap_rotation_delta = max(
                    overlap_rotation_delta,
                    float(
                        np.max(
                            np.abs(
                                refined.rotations[frame_id]
                                - previous.rotations[frame_id]
                            )
                        )
                    ),
                )
                overlap_center_delta = max(
                    overlap_center_delta,
                    float(
                        np.max(
                            np.abs(
                                refined.centers[frame_id]
                                - previous.centers[frame_id]
                            )
                        )
                    ),
                )

        constraint_counts = {frame_id: 0 for frame_id in frame_ids}
        frame_uid_by_id: dict[int, str] = {}
        for track in selected_tracks:
            for observation in track.observations:
                if observation.geometry_ordinal in constraint_counts:
                    constraint_counts[observation.geometry_ordinal] += 1
                    frame_uid_by_id.setdefault(
                        observation.geometry_ordinal, observation.frame_uid
                    )
        zero_constraint_frames = [
            frame_id for frame_id, count in constraint_counts.items() if count == 0
        ]
        actual_frame_uids = [
            frame_uid_by_id.get(frame_id, f"geometry-{frame_id}")
            for frame_id in frame_ids
        ]
        report = {
            "contractId": "jarailsense.gluemap-fixed-anchor-diagnostic-step/v1",
            "status": "passed" if not zero_constraint_frames else "failed",
            "publishable": False,
            "diagnosticMode": "fixed-anchor-approximation",
            "windowOrdinal": self._next_window_ordinal,
            "firstFrameId": frame_ids[0],
            "lastFrameId": frame_ids[-1],
            "frameCount": len(frame_ids),
            "advanceStepKeyframes": 1,
            "fixedPoseCount": len(local_ba_fixed_ids),
            "overlapFrameCount": len(fixed_pose_ids),
            "newFrameCount": len(frame_id_set - fixed_pose_ids),
            "actualBaCameraFrameUids": actual_frame_uids,
            "actualBaCameraFrameUidsSha256": _canonical_sha256(actual_frame_uids),
            "nonKeyframeBaCameraCount": 0,
            "zeroConstraintFrameIds": zero_constraint_frames,
            "minimumConstraintCount": min(constraint_counts.values()),
            "maximumOverlapRotationMatrixDelta": overlap_rotation_delta,
            "maximumOverlapCenterDelta": overlap_center_delta,
            "coarseWallSeconds": coarse_wall,
            "triangulationWallSeconds": triangulation_wall,
            "baWallSeconds": ba_wall,
            "totalWallSeconds": time.perf_counter() - started,
            "coarse": coarse.report,
            "triangulation": triangulation_report,
            "localBa": refined.report,
        }
        if zero_constraint_frames:
            raise FixedAnchorDiagnosticRunnerError(
                "diagnostic window contains zero-constraint frames"
            )
        self._previous = _RunnerPoseState(
            frame_ids=refined.frame_ids,
            rotations=refined.rotations,
            centers=refined.centers,
        )
        self._next_window_ordinal += 1
        return FixedAnchorDiagnosticStep(
            window_ordinal=self._next_window_ordinal - 1,
            coarse=coarse,
            triangulated_tracks=tuple(triangulated),
            refined=refined,
            report=report,
        )

    def snapshot(self) -> dict[str, Any]:
        if self._previous is None or self._frozen_intrinsics is None:
            raise FixedAnchorDiagnosticRunnerError(
                "diagnostic runner has no solved state"
            )
        state = {
            "contractId": "jarailsense.gluemap-fixed-anchor-checkpoint/v1",
            "status": "passed",
            "publishable": False,
            "diagnosticMode": "fixed-anchor-approximation",
            "nextWindowOrdinal": self._next_window_ordinal,
            "frameIds": list(self._previous.frame_ids),
            "rotations": {
                str(key): value.tolist()
                for key, value in sorted(self._previous.rotations.items())
            },
            "centers": {
                str(key): value.tolist()
                for key, value in sorted(self._previous.centers.items())
            },
            "frozenIntrinsics": self._frozen_intrinsics.tolist(),
        }
        return {**state, "stateSha256": _canonical_sha256(state)}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        state = {key: value for key, value in checkpoint.items() if key != "stateSha256"}
        if (
            checkpoint.get("contractId")
            != "jarailsense.gluemap-fixed-anchor-checkpoint/v1"
            or checkpoint.get("status") != "passed"
            or checkpoint.get("publishable") is not False
            or checkpoint.get("diagnosticMode") != "fixed-anchor-approximation"
            or checkpoint.get("stateSha256") != _canonical_sha256(state)
        ):
            raise FixedAnchorDiagnosticRunnerError(
                "diagnostic checkpoint identity differs"
            )
        frame_ids = tuple(int(value) for value in checkpoint.get("frameIds", []))
        rotations = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in checkpoint.get("rotations", {}).items()
        }
        centers = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in checkpoint.get("centers", {}).items()
        }
        intrinsics = np.asarray(checkpoint.get("frozenIntrinsics"), dtype=np.float64)
        next_ordinal = checkpoint.get("nextWindowOrdinal")
        if (
            len(frame_ids) < 3
            or set(rotations) != set(frame_ids)
            or set(centers) != set(frame_ids)
            or intrinsics.shape != (3, 3)
            or isinstance(next_ordinal, bool)
            or not isinstance(next_ordinal, int)
            or next_ordinal < 1
        ):
            raise FixedAnchorDiagnosticRunnerError(
                "diagnostic checkpoint state is invalid"
            )
        self._previous = _RunnerPoseState(
            frame_ids=frame_ids,
            rotations=rotations,
            centers=centers,
        )
        self._frozen_intrinsics = intrinsics
        self._next_window_ordinal = next_ordinal
