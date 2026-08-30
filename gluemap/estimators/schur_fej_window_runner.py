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
        **fixed_lag_options: Any,
    ) -> None:
        self.fixed_gauge_frame_id = int(fixed_gauge_frame_id)
        self.coarse_solver = coarse_solver or FixedAnchorApproximationSolver()
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
        started = time.perf_counter()
        if (
            len(frame_ids) < 3
            or frame_ids != sorted(set(frame_ids))
            or self.fixed_gauge_frame_id not in frame_ids
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
            if len(removed) != 1 or removed != already_marginalized:
                raise SchurFejWindowRunnerError(
                    "Schur/FEJ advance did not drop the finalized pose"
                )
        marginalize_frame_id = min(
            value
            for value in frame_ids
            if value != self.fixed_gauge_frame_id
        )

        coarse_started = time.perf_counter()
        coarse = self.coarse_solver.solve(
            predictions,
            frame_ids,
            initial_rotations=(previous_rotations or None),
            initial_centers=(previous_centers or None),
            fixed_pose_ids=overlap,
        )
        coarse_wall = time.perf_counter() - coarse_started
        solved = self.fixed_lag.advance(
            coarse,
            selected_tracks,
            marginalize_frame_id=marginalize_frame_id,
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
            "fixedGaugeFrameId": self.fixed_gauge_frame_id,
            "coarseFixedWarmStartCount": len(overlap),
            "actualBaCameraFrameUids": frame_uids,
            "actualBaCameraFrameUidsSha256": _canonical_sha256(frame_uids),
            "nonKeyframeBaCameraCount": 0,
            "zeroConstraintFrameIds": zero_constraint_frames,
            "minimumConstraintCount": min(constraint_counts.values()),
            "coarseWallSeconds": coarse_wall,
            "coarse": coarse.report,
            "totalWallSeconds": time.perf_counter() - started,
        }
        return SchurFejWindowStep(
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
