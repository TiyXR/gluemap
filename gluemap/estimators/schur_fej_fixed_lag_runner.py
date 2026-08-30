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
from gluemap.estimators.fixed_lag_prior import FejPriorState
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
    refined: FixedAnchorLocalBaSolution
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
        ba_max_iterations: int = 100,
        ba_refinement_passes: int = 1,
        ceres_cuda_available: bool | None = None,
        prior_device_policy: str = "cuda-required",
        prior_relative_rank_threshold: float = 1e-10,
        prior_maximum_condition_estimate: float | None = None,
        prior_expected_nullity: int | None = 1,
    ) -> None:
        if not fixed_gauge_frame_ids:
            raise SchurFejFixedLagRunnerError("fixed gauge pose set is empty")
        self.fixed_gauge_frame_ids = set(fixed_gauge_frame_ids)
        self.camera_model = camera_model
        self.triangulation_device_policy = triangulation_device_policy
        self.triangulation_microbatch_tracks = triangulation_microbatch_tracks
        self.ba_device_policy = ba_device_policy
        self.ba_linear_solver_policy = ba_linear_solver_policy
        self.ba_max_iterations = ba_max_iterations
        self.ba_refinement_passes = ba_refinement_passes
        self.ceres_cuda_available = ceres_cuda_available
        self.prior_device_policy = prior_device_policy
        self.prior_relative_rank_threshold = prior_relative_rank_threshold
        self.prior_maximum_condition_estimate = prior_maximum_condition_estimate
        self.prior_expected_nullity = prior_expected_nullity
        self._prior: FejPriorState | None = None
        self._poses: _PoseState | None = None
        self._frozen_intrinsics: np.ndarray | None = None
        self._next_window_ordinal = 0

    @property
    def prior(self) -> FejPriorState | None:
        return self._prior

    @property
    def next_window_ordinal(self) -> int:
        return self._next_window_ordinal

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
        started = time.perf_counter()
        frame_ids = tuple(coarse.frame_ids)
        frame_id_set = set(frame_ids)
        if (
            len(frame_ids) < 3
            or tuple(sorted(frame_id_set)) != frame_ids
            or self.fixed_gauge_frame_ids - frame_id_set
        ):
            raise SchurFejFixedLagRunnerError("fixed-lag frame identity is invalid")
        if (
            marginalize_frame_id not in frame_id_set
            or marginalize_frame_id in self.fixed_gauge_frame_ids
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
        refined = refine_fixed_anchor_window(
            warm_coarse,
            triangulated,
            matrix_k,
            fixed_pose_ids=self.fixed_gauge_frame_ids,
            camera_model=self.camera_model,
            max_num_iterations=self.ba_max_iterations,
            refinement_passes=self.ba_refinement_passes,
            linear_solver_policy=self.ba_linear_solver_policy,
            device_policy=self.ba_device_policy,
            ceres_cuda_available=self.ceres_cuda_available,
            previous_prior=self._prior,
            marginalize_pose_id=marginalize_frame_id,
            prior_device_policy=self.prior_device_policy,
            prior_relative_rank_threshold=self.prior_relative_rank_threshold,
            prior_maximum_condition_estimate=(
                self.prior_maximum_condition_estimate
            ),
            prior_expected_nullity=self.prior_expected_nullity,
        )
        solve_wall = time.perf_counter() - solve_started
        if refined.report["status"] != "passed" or refined.next_prior is None:
            raise SchurFejFixedLagRunnerError("fixed-lag BA/prior did not pass")
        if refined.next_prior.report["status"] != "passed":
            raise SchurFejFixedLagRunnerError("fixed-lag prior gate did not pass")

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
            "frameCount": len(frame_ids),
            "fixedGaugeFrameIds": sorted(self.fixed_gauge_frame_ids),
            "previousPriorCameraCount": (
                0 if self._prior is None else len(self._prior.camera_ids)
            ),
            "nextPriorCameraCount": len(refined.next_prior.camera_ids),
            "marginalizedFrameId": marginalize_frame_id,
            "overlapFrameCount": len(overlap_ids),
            "maximumOverlapRotationMatrixDelta": overlap_rotation_delta,
            "maximumOverlapCenterDelta": overlap_center_delta,
            "triangulationWallSeconds": triangulation_wall,
            "solveAndPriorWallSeconds": solve_wall,
            "totalWallSeconds": time.perf_counter() - started,
            "triangulation": triangulation_report,
            "localBa": refined.report,
            "prior": refined.next_prior.report,
        }
        self._prior = refined.next_prior
        self._poses = _PoseState(
            rotations=refined.rotations,
            centers=refined.centers,
        )
        step = SchurFejFixedLagStep(
            window_ordinal=self._next_window_ordinal,
            frame_ids=frame_ids,
            finalized_frame_id=marginalize_frame_id,
            finalized_rotation=refined.rotations[marginalize_frame_id].copy(),
            finalized_center=refined.centers[marginalize_frame_id].copy(),
            triangulated_tracks=tuple(triangulated),
            refined=refined,
            prior=refined.next_prior,
            report=report,
        )
        self._next_window_ordinal += 1
        return step

    def finalize_terminal(
        self,
        coarse: FixedAnchorWindowSolution,
        selected_tracks: list[SelectedTrackState],
        *,
        final_frame_id: int,
    ) -> SchurFejTerminalStep:
        """Solve and freeze the final body pose without creating an empty prior."""
        started = time.perf_counter()
        if self._prior is None or self._poses is None:
            raise SchurFejFixedLagRunnerError(
                "terminal fixed-lag state is unavailable"
            )
        frame_ids = tuple(coarse.frame_ids)
        expected_ids = tuple(
            sorted(self.fixed_gauge_frame_ids | set(self._prior.camera_ids))
        )
        if (
            frame_ids != expected_ids
            or self._prior.camera_ids != (final_frame_id,)
            or len(frame_ids) != len(self.fixed_gauge_frame_ids) + 1
            or not selected_tracks
        ):
            raise SchurFejFixedLagRunnerError(
                "terminal fixed-lag frame identity is invalid"
            )
        rotations = {
            frame_id: self._poses.rotations[frame_id].copy()
            for frame_id in frame_ids
        }
        centers = {
            frame_id: self._poses.centers[frame_id].copy()
            for frame_id in frame_ids
        }
        warm = FixedAnchorWindowSolution(
            frame_ids=frame_ids,
            rotations=rotations,
            centers=centers,
            intrinsics=coarse.intrinsics,
            report=coarse.report,
        )
        triangulation_started = time.perf_counter()
        triangulated, triangulation_report = triangulate_selected_tracks(
            selected_tracks,
            rotations,
            centers,
            self._frozen_intrinsics,
            device_policy=self.triangulation_device_policy,
            microbatch_tracks=self.triangulation_microbatch_tracks,
        )
        triangulation_wall = time.perf_counter() - triangulation_started
        solve_started = time.perf_counter()
        refined = refine_fixed_anchor_window(
            warm,
            triangulated,
            self._frozen_intrinsics,
            fixed_pose_ids=self.fixed_gauge_frame_ids,
            camera_model=self.camera_model,
            max_num_iterations=self.ba_max_iterations,
            refinement_passes=self.ba_refinement_passes,
            linear_solver_policy=self.ba_linear_solver_policy,
            device_policy=self.ba_device_policy,
            ceres_cuda_available=self.ceres_cuda_available,
            previous_prior=self._prior,
            marginalize_pose_id=None,
            prior_device_policy=self.prior_device_policy,
            prior_relative_rank_threshold=self.prior_relative_rank_threshold,
            prior_maximum_condition_estimate=(
                self.prior_maximum_condition_estimate
            ),
            prior_expected_nullity=self.prior_expected_nullity,
        )
        solve_wall = time.perf_counter() - solve_started
        if refined.report["status"] != "passed" or refined.next_prior is not None:
            raise SchurFejFixedLagRunnerError(
                "terminal fixed-lag BA did not pass"
            )
        final_rotation = refined.rotations[final_frame_id].copy()
        final_center = refined.centers[final_frame_id].copy()
        gauge_rotations = {
            frame_id: refined.rotations[frame_id].copy()
            for frame_id in self.fixed_gauge_frame_ids
        }
        gauge_centers = {
            frame_id: refined.centers[frame_id].copy()
            for frame_id in self.fixed_gauge_frame_ids
        }
        report = {
            "contractId": "jarailsense.gluemap-schur-fej-terminal-step/v1",
            "status": "passed",
            "publishable": False,
            "marginalizationMode": "schur-fej",
            "windowOrdinal": self._next_window_ordinal,
            "frameCount": len(frame_ids),
            "fixedGaugeFrameIds": sorted(self.fixed_gauge_frame_ids),
            "previousPriorCameraCount": len(self._prior.camera_ids),
            "nextPriorCameraCount": 0,
            "marginalizedFrameId": final_frame_id,
            "terminalFinalized": True,
            "triangulationWallSeconds": triangulation_wall,
            "solveAndPriorWallSeconds": solve_wall,
            "totalWallSeconds": time.perf_counter() - started,
            "triangulation": triangulation_report,
            "localBa": refined.report,
            "prior": None,
        }
        self._prior = None
        self._poses = _PoseState(
            rotations=gauge_rotations,
            centers=gauge_centers,
        )
        step = SchurFejTerminalStep(
            window_ordinal=self._next_window_ordinal,
            frame_ids=frame_ids,
            finalized_frame_id=final_frame_id,
            finalized_rotation=final_rotation,
            finalized_center=final_center,
            triangulated_tracks=tuple(triangulated),
            refined=refined,
            prior=None,
            report=report,
        )
        self._next_window_ordinal += 1
        return step

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
