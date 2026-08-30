"""Persistent Schur/FEJ fixed-lag solve state over caller-initialized windows."""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gluemap.estimators.active_track_store import SelectedTrackState
from gluemap.estimators.fixed_anchor_approximation import FixedAnchorWindowSolution
from gluemap.estimators.fixed_anchor_diagnostic_runner import (
    _resolve_shared_intrinsics,
)
from gluemap.estimators.fixed_anchor_local_ba import (
    FixedAnchorLocalBaSolution,
    refine_fixed_anchor_window,
)
from gluemap.estimators.fixed_lag_prior import (
    FejPriorState,
    marginalize_pose_prior,
    marginalize_pose_prior_batch,
)
from gluemap.estimators.persistent_fixed_lag_ba import (
    PersistentFixedLagBaSession,
)
from gluemap.estimators.fixed_lag_triangulation import (
    TriangulatedTrackState,
    triangulate_selected_tracks,
)


class SchurFejFixedLagRunnerError(ValueError):
    """Raised when the persistent Schur/FEJ state cannot advance exactly."""


@dataclass(frozen=True)
class SchurFejFixedLagStep:
    window_ordinal: int
    frame_ids: tuple[int, ...]
    finalized_frame_id: int
    finalized_rotation: np.ndarray
    finalized_center: np.ndarray
    triangulated_tracks: tuple[TriangulatedTrackState, ...]
    refined: FixedAnchorLocalBaSolution | None
    prior: FejPriorState
    report: dict[str, Any]


@dataclass(frozen=True)
class SchurFejFixedLagBatch:
    window_ordinal: int
    frame_ids: tuple[int, ...]
    finalized_frame_ids: tuple[int, ...]
    finalized_rotations: dict[int, np.ndarray]
    finalized_centers: dict[int, np.ndarray]
    triangulated_tracks: tuple[TriangulatedTrackState, ...]
    refined: FixedAnchorLocalBaSolution
    prior: FejPriorState
    report: dict[str, Any]


@dataclass(frozen=True)
class SchurFejTerminalStep:
    window_ordinal: int
    frame_ids: tuple[int, ...]
    finalized_frame_id: int
    finalized_rotation: np.ndarray
    finalized_center: np.ndarray
    triangulated_tracks: tuple[TriangulatedTrackState, ...]
    refined: FixedAnchorLocalBaSolution | None
    prior: None
    report: dict[str, Any]


@dataclass(frozen=True)
class _PoseState:
    rotations: dict[int, np.ndarray]
    centers: dict[int, np.ndarray]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SchurFejFixedLagRunner:
    """Own one canonical gauge and a persistent pose-only FEJ prior."""

    def __init__(
        self,
        *,
        fixed_gauge_frame_ids: set[int],
        camera_model: str = "SIMPLE_PINHOLE",
        triangulation_device_policy: str = "cuda-required",
        triangulation_microbatch_tracks: int = 4096,
        ba_device_policy: str = "cuda-preferred",
        ba_linear_solver_policy: str = "auto",
        ba_linear_solver_ordering_policy: str = "auto",
        ba_max_iterations: int = 100,
        ba_refinement_passes: int = 1,
        ceres_cuda_available: bool | None = None,
        prior_device_policy: str = "cuda-required",
        marginalization_residual_policy: str = "all-active",
        ba_problem_policy: str = "rebuild-every-window",
        prior_relative_rank_threshold: float = 1e-10,
        prior_maximum_condition_estimate: float | None = None,
        prior_expected_nullity: int | None = 1,
    ) -> None:
        if not fixed_gauge_frame_ids:
            raise SchurFejFixedLagRunnerError("fixed gauge pose set is empty")
        if marginalization_residual_policy not in {
            "all-active",
            "retiring-track-closure",
        }:
            raise SchurFejFixedLagRunnerError(
                "marginalization residual policy is invalid"
            )
        if ba_problem_policy not in {
            "rebuild-every-window",
            "persistent-delta",
            "native-rebuild-every-window",
        }:
            raise SchurFejFixedLagRunnerError("BA problem policy is invalid")
        if ba_linear_solver_ordering_policy not in {"auto", "point-first"}:
            raise SchurFejFixedLagRunnerError(
                "BA linear solver ordering policy is invalid"
            )
        self.fixed_gauge_frame_ids = set(fixed_gauge_frame_ids)
        self.camera_model = camera_model
        self.triangulation_device_policy = triangulation_device_policy
        self.triangulation_microbatch_tracks = triangulation_microbatch_tracks
        self.ba_device_policy = ba_device_policy
        self.ba_linear_solver_policy = ba_linear_solver_policy
        self.ba_linear_solver_ordering_policy = (
            ba_linear_solver_ordering_policy
        )
        self.ba_max_iterations = ba_max_iterations
        self.ba_refinement_passes = ba_refinement_passes
        self.ceres_cuda_available = ceres_cuda_available
        self.prior_device_policy = prior_device_policy
        self.marginalization_residual_policy = marginalization_residual_policy
        self.ba_problem_policy = ba_problem_policy
        self.prior_relative_rank_threshold = prior_relative_rank_threshold
        self.prior_maximum_condition_estimate = prior_maximum_condition_estimate
        self.prior_expected_nullity = prior_expected_nullity
        self._prior: FejPriorState | None = None
        self._poses: _PoseState | None = None
        self._frozen_intrinsics: np.ndarray | None = None
        self._next_window_ordinal = 0
        self._terminal_finalized = False
        self._persistent_ba_session = (
            PersistentFixedLagBaSession(policy=ba_problem_policy)
            if ba_problem_policy
            in {"persistent-delta", "native-rebuild-every-window"}
            else None
        )

    @property
    def prior(self) -> FejPriorState | None:
        return self._prior

    @property
    def next_window_ordinal(self) -> int:
        return self._next_window_ordinal

    @property
    def terminal_finalized(self) -> bool:
        return self._terminal_finalized

    @property
    def current_frame_ids(self) -> tuple[int, ...]:
        if self._poses is None:
            return ()
        return tuple(sorted(self._poses.rotations))

    @property
    def frozen_intrinsics_copy(self) -> np.ndarray:
        if self._frozen_intrinsics is None:
            raise SchurFejFixedLagRunnerError(
                "fixed-lag intrinsics are unavailable"
            )
        return self._frozen_intrinsics.copy()

    def current_pose_copies(
        self,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        if self._poses is None:
            return {}, {}
        return (
            {key: value.copy() for key, value in self._poses.rotations.items()},
            {key: value.copy() for key, value in self._poses.centers.items()},
        )

    def advance(
        self,
        coarse: FixedAnchorWindowSolution,
        selected_tracks: list[SelectedTrackState],
        *,
        marginalize_frame_id: int,
    ) -> SchurFejFixedLagStep:
        """Advance one pose while preserving the original public contract."""
        batch = self.advance_batch(
            coarse,
            selected_tracks,
            marginalize_frame_ids=(marginalize_frame_id,),
        )
        return SchurFejFixedLagStep(
            window_ordinal=batch.window_ordinal,
            frame_ids=batch.frame_ids,
            finalized_frame_id=marginalize_frame_id,
            finalized_rotation=batch.finalized_rotations[
                marginalize_frame_id
            ].copy(),
            finalized_center=batch.finalized_centers[
                marginalize_frame_id
            ].copy(),
            triangulated_tracks=batch.triangulated_tracks,
            refined=batch.refined,
            prior=batch.prior,
            report=batch.report,
        )

    def advance_batch(
        self,
        coarse: FixedAnchorWindowSolution,
        selected_tracks: list[SelectedTrackState],
        *,
        marginalize_frame_ids: tuple[int, ...] | list[int],
    ) -> SchurFejFixedLagBatch:
        """Run one BA and finalize a bounded oldest-pose batch."""
        started = time.perf_counter()
        frame_ids = tuple(coarse.frame_ids)
        frame_id_set = set(frame_ids)
        marginalize_ids = tuple(int(value) for value in marginalize_frame_ids)
        if (
            len(frame_ids) < 3
            or tuple(sorted(frame_id_set)) != frame_ids
            or self.fixed_gauge_frame_ids - frame_id_set
        ):
            raise SchurFejFixedLagRunnerError("fixed-lag frame identity is invalid")
        body_frame_ids = tuple(
            value for value in frame_ids if value not in self.fixed_gauge_frame_ids
        )
        if (
            not marginalize_ids
            or len(set(marginalize_ids)) != len(marginalize_ids)
            or marginalize_ids != body_frame_ids[: len(marginalize_ids)]
            or len(marginalize_ids) >= len(body_frame_ids)
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag marginal pose identity is invalid"
            )
        if self._prior is not None and set(self._prior.camera_ids) - frame_id_set:
            raise SchurFejFixedLagRunnerError(
                "fixed-lag window dropped a retained prior pose"
            )
        if not selected_tracks:
            raise SchurFejFixedLagRunnerError("fixed-lag selected tracks are empty")
        if any(
            observation.geometry_ordinal not in frame_id_set
            for track in selected_tracks
            for observation in track.observations
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag track observation is outside the window"
            )

        rotations = {
            frame_id: np.asarray(coarse.rotations[frame_id], dtype=np.float64).copy()
            for frame_id in frame_ids
        }
        centers = {
            frame_id: np.asarray(coarse.centers[frame_id], dtype=np.float64).copy()
            for frame_id in frame_ids
        }
        overlap_ids: set[int] = set()
        if self._poses is not None:
            overlap_ids = set(self._poses.rotations) & frame_id_set
            for frame_id in overlap_ids:
                rotations[frame_id] = self._poses.rotations[frame_id].copy()
                centers[frame_id] = self._poses.centers[frame_id].copy()
        warm_coarse = FixedAnchorWindowSolution(
            frame_ids=frame_ids,
            rotations=rotations,
            centers=centers,
            intrinsics=coarse.intrinsics,
            report=coarse.report,
        )
        if self._frozen_intrinsics is None:
            self._frozen_intrinsics = _resolve_shared_intrinsics(coarse.intrinsics)
        matrix_k = self._frozen_intrinsics

        triangulation_started = time.perf_counter()
        triangulated, triangulation_report = triangulate_selected_tracks(
            selected_tracks,
            rotations,
            centers,
            matrix_k,
            device_policy=self.triangulation_device_policy,
            microbatch_tracks=self.triangulation_microbatch_tracks,
        )
        triangulation_wall = time.perf_counter() - triangulation_started
        solve_started = time.perf_counter()
        if (
            self._persistent_ba_session is not None
            and self.ba_problem_policy == "native-rebuild-every-window"
        ):
            self._persistent_ba_session.problem = None
        refined = refine_fixed_anchor_window(
            warm_coarse,
            triangulated,
            matrix_k,
            fixed_pose_ids=self.fixed_gauge_frame_ids,
            camera_model=self.camera_model,
            max_num_iterations=self.ba_max_iterations,
            refinement_passes=self.ba_refinement_passes,
            linear_solver_policy=self.ba_linear_solver_policy,
            linear_solver_ordering_policy=(
                self.ba_linear_solver_ordering_policy
            ),
            device_policy=self.ba_device_policy,
            ceres_cuda_available=self.ceres_cuda_available,
            previous_prior=self._prior,
            marginalize_pose_id=marginalize_ids[0],
            marginalization_residual_policy=(
                self.marginalization_residual_policy
            ),
            prior_device_policy=self.prior_device_policy,
            prior_relative_rank_threshold=self.prior_relative_rank_threshold,
            prior_maximum_condition_estimate=(
                self.prior_maximum_condition_estimate
            ),
            prior_expected_nullity=self.prior_expected_nullity,
            persistent_ba_session=self._persistent_ba_session,
        )
        if refined.report["status"] != "passed" or refined.next_prior is None:
            raise SchurFejFixedLagRunnerError("fixed-lag BA/prior did not pass")
        next_prior = refined.next_prior
        batch_prior_wall = 0.0
        if len(marginalize_ids) > 1:
            batch_prior_started = time.perf_counter()
            next_prior = marginalize_pose_prior_batch(
                next_prior,
                eliminate_camera_ids=marginalize_ids[1:],
                device_policy=self.prior_device_policy,
                relative_rank_threshold=self.prior_relative_rank_threshold,
                maximum_condition_estimate=(
                    self.prior_maximum_condition_estimate
                ),
                expected_nullity=self.prior_expected_nullity,
            )
            batch_prior_wall = time.perf_counter() - batch_prior_started
        solve_wall = time.perf_counter() - solve_started
        if next_prior.report["status"] != "passed":
            raise SchurFejFixedLagRunnerError(
                "fixed-lag prior gate did not pass: "
                f"{next_prior.report}"
            )

        overlap_rotation_delta = 0.0
        overlap_center_delta = 0.0
        if self._poses is not None:
            for frame_id in overlap_ids:
                if frame_id in self.fixed_gauge_frame_ids:
                    continue
                overlap_rotation_delta = max(
                    overlap_rotation_delta,
                    float(
                        np.max(
                            np.abs(
                                refined.rotations[frame_id]
                                - self._poses.rotations[frame_id]
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
                                - self._poses.centers[frame_id]
                            )
                        )
                    ),
                )
        report = {
            "contractId": "jarailsense.gluemap-schur-fej-step/v1",
            "status": "passed",
            "publishable": False,
            "marginalizationMode": "schur-fej",
            "windowOrdinal": self._next_window_ordinal,
            "advanceStepKeyframes": 1,
            "baSolveEveryAdvances": len(marginalize_ids),
            "logicalAdvanceCount": len(marginalize_ids),
            "frameCount": len(frame_ids),
            "fixedGaugeFrameIds": sorted(self.fixed_gauge_frame_ids),
            "previousPriorCameraCount": (
                0 if self._prior is None else len(self._prior.camera_ids)
            ),
            "nextPriorCameraCount": len(next_prior.camera_ids),
            "marginalizedFrameId": marginalize_ids[0],
            "marginalizedFrameIds": list(marginalize_ids),
            "overlapFrameCount": len(overlap_ids),
            "maximumOverlapRotationMatrixDelta": overlap_rotation_delta,
            "maximumOverlapCenterDelta": overlap_center_delta,
            "triangulationWallSeconds": triangulation_wall,
            "solveAndPriorWallSeconds": solve_wall,
            "batchPriorWallSeconds": batch_prior_wall,
            "totalWallSeconds": time.perf_counter() - started,
            "triangulation": triangulation_report,
            "localBa": refined.report,
            "prior": next_prior.report,
        }
        self._prior = next_prior
        self._poses = _PoseState(
            rotations=refined.rotations,
            centers=refined.centers,
        )
        batch = SchurFejFixedLagBatch(
            window_ordinal=self._next_window_ordinal,
            frame_ids=frame_ids,
            finalized_frame_ids=marginalize_ids,
            finalized_rotations={
                value: refined.rotations[value].copy()
                for value in marginalize_ids
            },
            finalized_centers={
                value: refined.centers[value].copy()
                for value in marginalize_ids
            },
            triangulated_tracks=tuple(triangulated),
            refined=refined,
            prior=next_prior,
            report=report,
        )
        self._next_window_ordinal += len(marginalize_ids)
        return batch

    def drain_prior_only(
        self,
        *,
        finalized_frame_id: int,
    ) -> SchurFejFixedLagStep | SchurFejTerminalStep:
        """Release one sealed tail pose from the existing FEJ prior only.

        Railway video is forward-only, so the fixed gauge at the beginning of
        the sequence normally has no visual overlap with the retained tail.
        Re-triangulating and running BA during drain both wastes throughput and
        creates a false long-range visual requirement.  Retained poses are
        therefore frozen from the last optimized state and the existing dense
        FEJ prior is Schur-eliminated directly on its resolved device.
        """
        started = time.perf_counter()
        if self._prior is None or self._poses is None:
            raise SchurFejFixedLagRunnerError(
                "terminal fixed-lag state is unavailable"
            )
        frame_ids = tuple(
            sorted(self.fixed_gauge_frame_ids | set(self._prior.camera_ids))
        )
        if (
            finalized_frame_id != min(self._prior.camera_ids)
            or finalized_frame_id in self.fixed_gauge_frame_ids
            or finalized_frame_id not in self._poses.rotations
            or finalized_frame_id not in self._poses.centers
        ):
            raise SchurFejFixedLagRunnerError(
                "terminal fixed-lag frame identity is invalid"
            )
        final_rotation = self._poses.rotations[finalized_frame_id].copy()
        final_center = self._poses.centers[finalized_frame_id].copy()
        previous_prior = self._prior
        terminal = len(previous_prior.camera_ids) == 1
        solve_started = time.perf_counter()
        next_prior = None
        if not terminal:
            next_prior = marginalize_pose_prior(
                previous_prior,
                eliminate_camera_id=finalized_frame_id,
                device_policy=self.prior_device_policy,
                relative_rank_threshold=self.prior_relative_rank_threshold,
                maximum_condition_estimate=(
                    self.prior_maximum_condition_estimate
                ),
                expected_nullity=self.prior_expected_nullity,
            )
            if next_prior.report["status"] != "passed":
                raise SchurFejFixedLagRunnerError(
                    "terminal pose-only prior gate did not pass: "
                    f"{next_prior.report}"
                )
        solve_wall = time.perf_counter() - solve_started
        retained_ids = self.fixed_gauge_frame_ids | (
            set() if next_prior is None else set(next_prior.camera_ids)
        )
        retained_rotations = {
            frame_id: self._poses.rotations[frame_id].copy()
            for frame_id in retained_ids
        }
        retained_centers = {
            frame_id: self._poses.centers[frame_id].copy()
            for frame_id in retained_ids
        }
        solve_mode = "sealed-pose-freeze" if terminal else "prior-only-schur"
        local_report = {
            "contractId": "jarailsense.gluemap-prior-only-terminal-solve/v1",
            "status": "passed",
            "solverMode": solve_mode,
            "gpuUsed": bool(
                next_prior is not None and next_prior.report["gpuUsed"]
            ),
            "denseLinearAlgebraBackend": "NONE",
            "sparseLinearAlgebraBackend": "NONE",
            "ceresThreadsGiven": 0,
            "ceresThreadsUsed": 0,
            "trackCount": 0,
            "observationCount": 0,
            "termination": (
                "SEALED_POSE_FREEZE" if terminal else "PRIOR_ONLY_SCHUR"
            ),
            "initialCost": 0.0,
            "finalCost": 0.0,
            "maximumFixedRotationMatrixDelta": 0.0,
            "maximumFixedCenterDelta": 0.0,
            "fixedPoseCount": len(self.fixed_gauge_frame_ids),
            "wallSeconds": solve_wall,
        }
        triangulation_report = {
            "contractId": "jarailsense.gluemap-terminal-triangulation/v1",
            "status": "passed",
            "mode": "not-run-prior-only",
            "gpuUsed": False,
            "trackCount": 0,
            "wallSeconds": 0.0,
        }
        report = {
            "contractId": (
                "jarailsense.gluemap-schur-fej-terminal-step/v1"
                if terminal
                else "jarailsense.gluemap-schur-fej-step/v1"
            ),
            "status": "passed",
            "publishable": False,
            "marginalizationMode": "schur-fej",
            "terminalSolveMode": solve_mode,
            "windowOrdinal": self._next_window_ordinal,
            "frameCount": len(frame_ids),
            "fixedGaugeFrameIds": sorted(self.fixed_gauge_frame_ids),
            "previousPriorCameraCount": len(previous_prior.camera_ids),
            "nextPriorCameraCount": (
                0 if next_prior is None else len(next_prior.camera_ids)
            ),
            "marginalizedFrameId": finalized_frame_id,
            "terminalFinalized": terminal,
            "triangulationWallSeconds": 0.0,
            "solveAndPriorWallSeconds": solve_wall,
            "totalWallSeconds": time.perf_counter() - started,
            "triangulation": triangulation_report,
            "localBa": local_report,
            "prior": None if next_prior is None else next_prior.report,
        }
        self._prior = next_prior
        self._poses = _PoseState(
            rotations=retained_rotations,
            centers=retained_centers,
        )
        step_type = SchurFejTerminalStep if terminal else SchurFejFixedLagStep
        step = step_type(
            window_ordinal=self._next_window_ordinal,
            frame_ids=frame_ids,
            finalized_frame_id=finalized_frame_id,
            finalized_rotation=final_rotation,
            finalized_center=final_center,
            triangulated_tracks=(),
            refined=None,
            prior=next_prior,
            report=report,
        )
        self._next_window_ordinal += 1
        self._terminal_finalized = terminal
        return step

    def snapshot_terminal(self) -> dict[str, Any]:
        """Return the terminal gauge state after every body pose is frozen."""
        if (
            not self._terminal_finalized
            or self._prior is not None
            or self._poses is None
            or self._frozen_intrinsics is None
            or set(self._poses.rotations) != self.fixed_gauge_frame_ids
            or set(self._poses.centers) != self.fixed_gauge_frame_ids
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag terminal state is unavailable"
            )
        state = {
            "contractId": "jarailsense.gluemap-schur-fej-terminal-state/v1",
            "status": "passed",
            "publishable": False,
            "marginalizationMode": "schur-fej",
            "nextWindowOrdinal": self._next_window_ordinal,
            "fixedGaugeFrameIds": sorted(self.fixed_gauge_frame_ids),
            "activeBodyFrameIds": [],
            "rotations": {
                str(key): value.tolist()
                for key, value in sorted(self._poses.rotations.items())
            },
            "centers": {
                str(key): value.tolist()
                for key, value in sorted(self._poses.centers.items())
            },
            "frozenIntrinsics": self._frozen_intrinsics.tolist(),
            "prior": None,
        }
        return {**state, "stateSha256": _canonical_sha256(state)}

    def snapshot(self) -> dict[str, Any]:
        """Return one JSON-compatible state containing the complete FEJ prior."""
        if (
            self._prior is None
            or self._poses is None
            or self._frozen_intrinsics is None
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag runner has no solved state"
            )
        prior = self._prior.cpu()
        state = {
            "contractId": "jarailsense.gluemap-schur-fej-checkpoint/v1",
            "status": "passed",
            "publishable": False,
            "marginalizationMode": "schur-fej",
            "nextWindowOrdinal": self._next_window_ordinal,
            "fixedGaugeFrameIds": sorted(self.fixed_gauge_frame_ids),
            "frameIds": sorted(self._poses.rotations),
            "rotations": {
                str(key): value.tolist()
                for key, value in sorted(self._poses.rotations.items())
            },
            "centers": {
                str(key): value.tolist()
                for key, value in sorted(self._poses.centers.items())
            },
            "frozenIntrinsics": self._frozen_intrinsics.tolist(),
            "prior": {
                "cameraIds": list(prior.camera_ids),
                "linearizationPoints": prior.linearization_points.tolist(),
                "hessian": prior.hessian.tolist(),
                "gradient": prior.gradient.tolist(),
                "factor": prior.factor.tolist(),
                "factorResidual": prior.factor_residual.tolist(),
                "report": prior.report,
            },
        }
        return {**state, "stateSha256": _canonical_sha256(state)}

    def restore(self, checkpoint: dict[str, Any]) -> None:
        """Restore an exact prior/pose state without recomputing old windows."""
        state = {
            key: value for key, value in checkpoint.items() if key != "stateSha256"
        }
        if checkpoint.get("contractId") == (
            "jarailsense.gluemap-schur-fej-terminal-state/v1"
        ):
            self._restore_terminal(checkpoint, state)
            return
        if (
            checkpoint.get("contractId")
            != "jarailsense.gluemap-schur-fej-checkpoint/v1"
            or checkpoint.get("status") != "passed"
            or checkpoint.get("publishable") is not False
            or checkpoint.get("marginalizationMode") != "schur-fej"
            or checkpoint.get("stateSha256") != _canonical_sha256(state)
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag checkpoint identity differs"
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
        intrinsics = np.asarray(
            checkpoint.get("frozenIntrinsics"), dtype=np.float64
        )
        next_ordinal = checkpoint.get("nextWindowOrdinal")
        fixed_gauge = set(checkpoint.get("fixedGaugeFrameIds", []))
        prior_value = checkpoint.get("prior")
        if (
            len(frame_ids) < 3
            or frame_ids != tuple(sorted(set(frame_ids)))
            or set(rotations) != set(frame_ids)
            or set(centers) != set(frame_ids)
            or fixed_gauge != self.fixed_gauge_frame_ids
            or fixed_gauge - set(frame_ids)
            or intrinsics.shape != (3, 3)
            or isinstance(next_ordinal, bool)
            or not isinstance(next_ordinal, int)
            or next_ordinal < 1
            or not isinstance(prior_value, dict)
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag checkpoint state is invalid"
            )
        prior_camera_ids = tuple(
            int(value) for value in prior_value.get("cameraIds", [])
        )
        camera_count = len(prior_camera_ids)
        linearization = torch.as_tensor(
            prior_value.get("linearizationPoints"), dtype=torch.float64
        )
        hessian = torch.as_tensor(
            prior_value.get("hessian"), dtype=torch.float64
        )
        gradient = torch.as_tensor(
            prior_value.get("gradient"), dtype=torch.float64
        )
        factor = torch.as_tensor(
            prior_value.get("factor"), dtype=torch.float64
        )
        factor_residual = torch.as_tensor(
            prior_value.get("factorResidual"), dtype=torch.float64
        )
        if (
            not prior_camera_ids
            or len(set(prior_camera_ids)) != camera_count
            or set(prior_camera_ids) - set(frame_ids)
            or linearization.shape != (camera_count, 7)
            or hessian.shape != (camera_count * 6, camera_count * 6)
            or gradient.shape != (camera_count * 6,)
            or factor.ndim != 2
            or factor.shape[1] != camera_count * 6
            or factor_residual.shape != (factor.shape[0],)
            or not all(
                bool(torch.isfinite(value).all())
                for value in (
                    linearization,
                    hessian,
                    gradient,
                    factor,
                    factor_residual,
                )
            )
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag checkpoint prior is invalid"
            )
        self._poses = _PoseState(rotations=rotations, centers=centers)
        self._frozen_intrinsics = intrinsics
        self._prior = FejPriorState(
            camera_ids=prior_camera_ids,
            linearization_points=linearization,
            hessian=hessian,
            gradient=gradient,
            factor=factor,
            factor_residual=factor_residual,
            report=dict(prior_value.get("report", {})),
        )
        self._next_window_ordinal = next_ordinal
        self._terminal_finalized = False
        if self._persistent_ba_session is not None:
            self._persistent_ba_session.problem = None

    def _restore_terminal(
        self, checkpoint: dict[str, Any], state: dict[str, Any]
    ) -> None:
        rotations = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in checkpoint.get("rotations", {}).items()
        }
        centers = {
            int(key): np.asarray(value, dtype=np.float64)
            for key, value in checkpoint.get("centers", {}).items()
        }
        intrinsics = np.asarray(
            checkpoint.get("frozenIntrinsics"), dtype=np.float64
        )
        next_ordinal = checkpoint.get("nextWindowOrdinal")
        fixed_gauge = set(checkpoint.get("fixedGaugeFrameIds", []))
        if (
            checkpoint.get("status") != "passed"
            or checkpoint.get("publishable") is not False
            or checkpoint.get("marginalizationMode") != "schur-fej"
            or checkpoint.get("stateSha256") != _canonical_sha256(state)
            or checkpoint.get("activeBodyFrameIds") != []
            or checkpoint.get("prior") is not None
            or fixed_gauge != self.fixed_gauge_frame_ids
            or set(rotations) != fixed_gauge
            or set(centers) != fixed_gauge
            or any(value.shape != (3, 3) for value in rotations.values())
            or any(value.shape != (3,) for value in centers.values())
            or intrinsics.shape != (3, 3)
            or isinstance(next_ordinal, bool)
            or not isinstance(next_ordinal, int)
            or next_ordinal < 1
        ):
            raise SchurFejFixedLagRunnerError(
                "fixed-lag terminal checkpoint state is invalid"
            )
        self._poses = _PoseState(rotations=rotations, centers=centers)
        self._frozen_intrinsics = intrinsics
        self._prior = None
        self._next_window_ordinal = next_ordinal
        self._terminal_finalized = True
        if self._persistent_ba_session is not None:
            self._persistent_ba_session.problem = None
