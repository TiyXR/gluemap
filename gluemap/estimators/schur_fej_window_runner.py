"""Initialize and advance one true Schur/FEJ fixed-lag geometry window."""

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
from gluemap.estimators.schur_fej_fixed_lag_runner import (
    SchurFejFixedLagBatch,
    SchurFejFixedLagRunner,
    SchurFejFixedLagStep,
    SchurFejTerminalStep,
)


class SchurFejWindowRunnerError(ValueError):
    """Raised when the initialized Schur/FEJ window identity is invalid."""


@dataclass(frozen=True)
class SchurFejWindowStep:
    window_ordinal: int
    coarse: FixedAnchorWindowSolution
    solved: SchurFejFixedLagStep | SchurFejTerminalStep
    report: dict[str, Any]


@dataclass(frozen=True)
class SchurFejWindowBatch:
    window_ordinal: int
    coarse: FixedAnchorWindowSolution
    solved: SchurFejFixedLagBatch
    report: dict[str, Any]


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SchurFejWindowRunner:
    """Use overlap poses only as coarse warm starts, never as fixed anchors."""

    def __init__(
        self,
        *,
        fixed_gauge_frame_id: int,
        coarse_solver: FixedAnchorApproximationSolver | Any | None = None,
        coarse_warm_start_enabled: bool = True,
        **fixed_lag_options: Any,
    ) -> None:
        self.fixed_gauge_frame_id = int(fixed_gauge_frame_id)
        self.coarse_solver = coarse_solver or FixedAnchorApproximationSolver()
        if not isinstance(coarse_warm_start_enabled, bool):
            raise SchurFejWindowRunnerError(
                "coarse warm start policy is invalid"
            )
        self.coarse_warm_start_enabled = coarse_warm_start_enabled
        self.fixed_lag = SchurFejFixedLagRunner(
            fixed_gauge_frame_ids={self.fixed_gauge_frame_id},
            **fixed_lag_options,
        )

    @property
    def next_window_ordinal(self) -> int:
        return self.fixed_lag.next_window_ordinal

    def advance(
        self,
        predictions: dict[str, list[Any]],
        frame_ids: list[int],
        selected_tracks: list[SelectedTrackState],
    ) -> SchurFejWindowStep:
        """Advance one keyframe while preserving the original API."""
        batch = self.advance_batch(
            predictions,
            frame_ids,
            selected_tracks,
            solve_every_advances=1,
        )
        solved = batch.solved
        finalized_frame_id = solved.finalized_frame_ids[0]
        return SchurFejWindowStep(
            window_ordinal=batch.window_ordinal,
            coarse=batch.coarse,
            solved=SchurFejFixedLagStep(
                window_ordinal=solved.window_ordinal,
                frame_ids=solved.frame_ids,
                finalized_frame_id=finalized_frame_id,
                finalized_rotation=solved.finalized_rotations[
                    finalized_frame_id
                ].copy(),
                finalized_center=solved.finalized_centers[
                    finalized_frame_id
                ].copy(),
                triangulated_tracks=solved.triangulated_tracks,
                refined=solved.refined,
                prior=solved.prior,
                report=solved.report,
            ),
            report=batch.report,
        )

    def advance_batch(
        self,
        predictions: dict[str, list[Any]],
        frame_ids: list[int],
        selected_tracks: list[SelectedTrackState],
        *,
        solve_every_advances: int,
    ) -> SchurFejWindowBatch:
        """Advance a bounded keyframe batch with one coarse and BA solve."""
        started = time.perf_counter()
        if (
            len(frame_ids) < 3
            or frame_ids != sorted(set(frame_ids))
            or self.fixed_gauge_frame_id not in frame_ids
            or isinstance(solve_every_advances, bool)
            or solve_every_advances < 1
        ):
            raise SchurFejWindowRunnerError(
                "Schur/FEJ frame ordering or gauge is invalid"
            )
        previous_frame_ids = set(self.fixed_lag.current_frame_ids)
        previous_rotations, previous_centers = (
            self.fixed_lag.current_pose_copies()
        )
        overlap = previous_frame_ids & set(frame_ids)
        if previous_frame_ids and len(overlap) < 2:
            raise SchurFejWindowRunnerError(
                "Schur/FEJ successive windows have insufficient overlap"
            )
        if previous_frame_ids:
            removed = (
                previous_frame_ids
                - set(frame_ids)
                - {self.fixed_gauge_frame_id}
            )
            prior_ids = set(self.fixed_lag.prior.camera_ids)
            already_marginalized = (
                previous_frame_ids
                - prior_ids
                - {self.fixed_gauge_frame_id}
            )
            if removed != already_marginalized:
                raise SchurFejWindowRunnerError(
                    "Schur/FEJ advance did not drop the finalized pose"
                )
        body_frame_ids = tuple(
            value
            for value in frame_ids
            if value != self.fixed_gauge_frame_id
        )
        if solve_every_advances >= len(body_frame_ids):
            raise SchurFejWindowRunnerError(
                "Schur/FEJ advance would remove every body pose"
            )
        marginalize_frame_ids = body_frame_ids[:solve_every_advances]

        coarse_started = time.perf_counter()
        coarse = self.coarse_solver.solve(
            predictions,
            frame_ids,
            initial_rotations=(
                previous_rotations
                if self.coarse_warm_start_enabled and previous_rotations
                else None
            ),
            initial_centers=(
                previous_centers
                if self.coarse_warm_start_enabled and previous_centers
                else None
            ),
            fixed_pose_ids=(overlap if self.coarse_warm_start_enabled else set()),
        )
        coarse_wall = time.perf_counter() - coarse_started
        solved = self.fixed_lag.advance_batch(
            coarse,
            selected_tracks,
            marginalize_frame_ids=marginalize_frame_ids,
        )
        constraint_counts = {frame_id: 0 for frame_id in frame_ids}
        frame_uid_by_id: dict[int, str] = {}
        for track in selected_tracks:
            for observation in track.observations:
                frame_id = observation.geometry_ordinal
                if frame_id in constraint_counts:
                    constraint_counts[frame_id] += 1
                    frame_uid_by_id.setdefault(frame_id, observation.frame_uid)
        zero_constraint_frames = [
            frame_id
            for frame_id, count in constraint_counts.items()
            if frame_id != self.fixed_gauge_frame_id and count == 0
        ]
        if zero_constraint_frames:
            raise SchurFejWindowRunnerError(
                "Schur/FEJ window contains zero-constraint frames"
            )
        frame_uids = [
            frame_uid_by_id.get(frame_id, f"geometry-{frame_id}")
            for frame_id in frame_ids
        ]
        report = {
            **solved.report,
            "contractId": "jarailsense.gluemap-schur-fej-window-step/v1",
            "firstFrameId": frame_ids[0],
            "lastFrameId": frame_ids[-1],
            "advanceStepKeyframes": 1,
            "baSolveEveryAdvances": solve_every_advances,
            "logicalAdvanceCount": solve_every_advances,
            "marginalizedFrameIds": list(marginalize_frame_ids),
            "fixedGaugeFrameId": self.fixed_gauge_frame_id,
            "coarseWarmStartEnabled": self.coarse_warm_start_enabled,
            "coarseFixedWarmStartCount": (
                len(overlap) if self.coarse_warm_start_enabled else 0
            ),
            "actualBaCameraFrameUids": frame_uids,
            "actualBaCameraFrameUidsSha256": _canonical_sha256(frame_uids),
            "nonKeyframeBaCameraCount": 0,
            "zeroConstraintFrameIds": zero_constraint_frames,
            "constraintExemptFrameIds": [self.fixed_gauge_frame_id],
            "minimumConstraintCount": min(
                count
                for frame_id, count in constraint_counts.items()
                if frame_id != self.fixed_gauge_frame_id
            ),
            "coarseWallSeconds": coarse_wall,
            "coarse": coarse.report,
            "totalWallSeconds": time.perf_counter() - started,
        }
        return SchurFejWindowBatch(
            window_ordinal=solved.window_ordinal,
            coarse=coarse,
            solved=solved,
            report=report,
        )

    def snapshot(self) -> dict[str, Any]:
        return self.fixed_lag.snapshot()

    def snapshot_terminal(self) -> dict[str, Any]:
        return self.fixed_lag.snapshot_terminal()

    def drain_next(
        self, selected_tracks: list[SelectedTrackState]
    ) -> SchurFejWindowStep:
        """Finalize one retained tail pose after the source timeline is sealed."""
        started = time.perf_counter()
        prior = self.fixed_lag.prior
        if prior is None or not prior.camera_ids:
            raise SchurFejWindowRunnerError(
                "Schur/FEJ terminal drain has no retained body pose"
            )
        frame_ids = sorted(
            {self.fixed_gauge_frame_id, *prior.camera_ids}
        )
        rotations, centers = self.fixed_lag.current_pose_copies()
        coarse = FixedAnchorWindowSolution(
            frame_ids=tuple(frame_ids),
            rotations={value: rotations[value] for value in frame_ids},
            centers={value: centers[value] for value in frame_ids},
            intrinsics=[self.fixed_lag.frozen_intrinsics_copy[None]],
            report={
                "contractId": "jarailsense.gluemap-terminal-warm-start/v1",
                "status": "passed",
            },
        )
        finalized_frame_id = min(prior.camera_ids)
        solved = self.fixed_lag.drain_prior_only(
            finalized_frame_id=finalized_frame_id,
        )
        constraint_counts = {frame_id: 0 for frame_id in frame_ids}
        frame_uid_by_id: dict[int, str] = {}
        for track in selected_tracks:
            for observation in track.observations:
                frame_id = observation.geometry_ordinal
                if frame_id in constraint_counts:
                    constraint_counts[frame_id] += 1
                    frame_uid_by_id.setdefault(frame_id, observation.frame_uid)
        zero_constraint_frames = [
            frame_id
            for frame_id, count in constraint_counts.items()
            if count == 0
        ]
        frame_uids = [
            frame_uid_by_id.get(frame_id, f"geometry-{frame_id}")
            for frame_id in frame_ids
        ]
        report = {
            **solved.report,
            "contractId": "jarailsense.gluemap-schur-fej-drain-step/v1",
            "terminalDrain": True,
            "firstFrameId": frame_ids[0],
            "lastFrameId": frame_ids[-1],
            "advanceStepKeyframes": 1,
            "fixedGaugeFrameId": self.fixed_gauge_frame_id,
            "coarseFixedWarmStartCount": 0,
            "actualBaCameraFrameUids": [],
            "actualBaCameraFrameUidsSha256": _canonical_sha256([]),
            "nonKeyframeBaCameraCount": 0,
            "zeroConstraintFrameIds": [],
            "minimumConstraintCount": 0,
            "terminalContextFrameUids": frame_uids,
            "terminalContextFrameUidsSha256": _canonical_sha256(frame_uids),
            "terminalGateZeroConstraintFrameIds": zero_constraint_frames,
            "terminalGateMinimumConstraintCount": min(
                constraint_counts.values(), default=0
            ),
            "coarseWallSeconds": 0.0,
            "totalWallSeconds": time.perf_counter() - started,
        }
        return SchurFejWindowStep(
            window_ordinal=solved.window_ordinal,
            coarse=coarse,
            solved=solved,
            report=report,
        )

    def restore(self, checkpoint: dict[str, Any]) -> None:
        self.fixed_lag.restore(checkpoint)
