"""Bounded active-window track ownership and coverage gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from math import floor, hypot, isfinite
from typing import Any, Iterable

from gluemap.math.union_find import UnionFind


class ActiveTrackStoreError(ValueError):
    """Raised when active track state violates its deterministic contract."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: str, scope: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ActiveTrackStoreError(f"{scope} must be a lowercase SHA-256")


@dataclass(frozen=True)
class TrackBudget:
    window_size_keyframes: int
    query_tracks_per_keyframe: int
    active_track_budget_per_keyframe: int
    minimum_constraints_per_keyframe: int
    minimum_bridge_tracks: int
    minimum_track_views: int
    coverage_grid_columns: int
    coverage_grid_rows: int
    selection_time_bin_seconds: float
    maximum_tracks_per_grid_cell: int
    minimum_parallax_diagonals: float
    maximum_match_error_pixels: float

    @property
    def maximum_active_tracks(self) -> int:
        return self.window_size_keyframes * self.active_track_budget_per_keyframe


@dataclass(frozen=True)
class TrackObservation:
    observation_uid: str
    geometry_ordinal: int
    frame_uid: str
    pts_value: int
    time_base_numerator: int
    time_base_denominator: int
    x: float
    y: float
    image_width: int
    image_height: int
    score: float

    @property
    def seconds(self) -> float:
        return self.pts_value * self.time_base_numerator / self.time_base_denominator


@dataclass(frozen=True)
class TrackCorrespondence:
    first_observation_uid: str
    second_observation_uid: str
    match_error_pixels: float


class ActiveTrackStore:
    """Own candidate tracks only while their observations are in the lag.

    The store is deliberately independent from ``pycolmap``.  It receives the
    frontend's stable observation identities, builds connected components, and
    emits deterministic selection/gate reports.  Destructive release is a
    two-step operation: a proposal is immutable, and the actual purge requires
    the SHA-256 of an already accepted durable journal head.
    """

    def __init__(self, budget: TrackBudget) -> None:
        self.budget = budget
        self._validate_budget()
        self._observations: dict[str, TrackObservation] = {}
        self._observations_by_frame: dict[int, set[str]] = defaultdict(set)
        self._edges: dict[tuple[str, str], TrackCorrespondence] = {}
        self._union_find = UnionFind()
        self._component_uid_by_root: dict[Any, str] = {}
        self._last_accepted_journal_head: str | None = None
        self._pending_release: dict[str, Any] | None = None
        self.store_identity_sha256 = _canonical_sha256(asdict(budget))

    def _validate_budget(self) -> None:
        integer_bounds = {
            "window_size_keyframes": (self.budget.window_size_keyframes, 2),
            "query_tracks_per_keyframe": (self.budget.query_tracks_per_keyframe, 1),
            "active_track_budget_per_keyframe": (
                self.budget.active_track_budget_per_keyframe,
                1,
            ),
            "minimum_constraints_per_keyframe": (
                self.budget.minimum_constraints_per_keyframe,
                1,
            ),
            "minimum_bridge_tracks": (self.budget.minimum_bridge_tracks, 1),
            "minimum_track_views": (self.budget.minimum_track_views, 2),
            "coverage_grid_columns": (self.budget.coverage_grid_columns, 2),
            "coverage_grid_rows": (self.budget.coverage_grid_rows, 2),
            "maximum_tracks_per_grid_cell": (
                self.budget.maximum_tracks_per_grid_cell,
                1,
            ),
        }
        for name, (value, minimum) in integer_bounds.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ActiveTrackStoreError(f"{name} is below its minimum")
        if (
            not isfinite(self.budget.selection_time_bin_seconds)
            or self.budget.selection_time_bin_seconds <= 0
            or not isfinite(self.budget.minimum_parallax_diagonals)
            or self.budget.minimum_parallax_diagonals < 0
            or not isfinite(self.budget.maximum_match_error_pixels)
            or self.budget.maximum_match_error_pixels <= 0
        ):
            raise ActiveTrackStoreError("track metric thresholds are invalid")
        if (
            self.budget.minimum_constraints_per_keyframe
            > self.budget.maximum_active_tracks
        ):
            raise ActiveTrackStoreError("minimum constraints exceed the active budget")

    @property
    def observation_count(self) -> int:
        return len(self._observations)

    @property
    def edge_count(self) -> int:
        return len(self._edges)

    @property
    def last_accepted_journal_head(self) -> str | None:
        return self._last_accepted_journal_head

    def add_observations(self, values: Iterable[TrackObservation]) -> None:
        if self._pending_release is not None:
            raise ActiveTrackStoreError("cannot ingest while release is pending")
        for value in values:
            self._validate_observation(value)
            existing = self._observations.get(value.observation_uid)
            if existing is not None:
                if existing != value:
                    raise ActiveTrackStoreError("observation identity was reused")
                continue
            frame_values = self._observations_by_frame[value.geometry_ordinal]
            if len(frame_values) >= self.budget.query_tracks_per_keyframe:
                raise ActiveTrackStoreError("per-keyframe observation bound exceeded")
            self._observations[value.observation_uid] = value
            frame_values.add(value.observation_uid)
            root = self._union_find.find(value.observation_uid)
            self._component_uid_by_root[root] = value.observation_uid

    def _validate_observation(self, value: TrackObservation) -> None:
        if (
            not value.observation_uid
            or not value.frame_uid
            or value.geometry_ordinal < 0
            or value.time_base_numerator <= 0
            or value.time_base_denominator <= 0
            or value.image_width <= 0
            or value.image_height <= 0
            or not all(isfinite(number) for number in (value.x, value.y, value.score))
            or value.x < 0
            or value.x >= value.image_width
            or value.y < 0
            or value.y >= value.image_height
            or value.score < 0
        ):
            raise ActiveTrackStoreError("track observation is invalid")

    def add_correspondences(self, values: Iterable[TrackCorrespondence]) -> None:
        if self._pending_release is not None:
            raise ActiveTrackStoreError("cannot ingest while release is pending")
        for value in values:
            first = value.first_observation_uid
            second = value.second_observation_uid
            if (
                first == second
                or first not in self._observations
                or second not in self._observations
                or not isfinite(value.match_error_pixels)
                or value.match_error_pixels < 0
            ):
                raise ActiveTrackStoreError("track correspondence is invalid")
            key = tuple(sorted((first, second)))
            existing = self._edges.get(key)
            if existing is not None:
                if existing != value and existing != TrackCorrespondence(
                    second, first, value.match_error_pixels
                ):
                    raise ActiveTrackStoreError("correspondence identity was reused")
                continue
            first_root = self._union_find.find(first)
            second_root = self._union_find.find(second)
            component_uid = min(
                self._component_uid_by_root[first_root],
                self._component_uid_by_root[second_root],
            )
            self._union_find.union(first_root, second_root)
            new_root = self._union_find.find(first)
            self._component_uid_by_root.pop(first_root, None)
            self._component_uid_by_root.pop(second_root, None)
            self._component_uid_by_root[new_root] = component_uid
            self._edges[key] = value

    def _components(self) -> dict[str, list[TrackObservation]]:
        values: dict[str, list[TrackObservation]] = defaultdict(list)
        for observation_uid, observation in self._observations.items():
            root = self._union_find.find(observation_uid)
            values[self._component_uid_by_root[root]].append(observation)
        for observations in values.values():
            observations.sort(
                key=lambda value: (
                    value.geometry_ordinal,
                    value.observation_uid,
                )
            )
        return dict(values)

    def _component_edges(self) -> dict[str, list[TrackCorrespondence]]:
        values: dict[str, list[TrackCorrespondence]] = defaultdict(list)
        for edge in self._edges.values():
            root = self._union_find.find(edge.first_observation_uid)
            values[self._component_uid_by_root[root]].append(edge)
        return dict(values)

    def _grid_cell(self, observation: TrackObservation) -> tuple[int, int]:
        column = min(
            self.budget.coverage_grid_columns - 1,
            floor(
                observation.x
                / observation.image_width
                * self.budget.coverage_grid_columns
            ),
        )
        row = min(
            self.budget.coverage_grid_rows - 1,
            floor(
                observation.y
                / observation.image_height
                * self.budget.coverage_grid_rows
            ),
        )
        return row, column

    def _parallax(self, observations: list[TrackObservation]) -> float:
        maximum = 0.0
        for index, first in enumerate(observations):
            first_diagonal = hypot(first.image_width, first.image_height)
            for second in observations[index + 1 :]:
                if first_diagonal <= 0:
                    continue
                maximum = max(
                    maximum,
                    hypot(second.x - first.x, second.y - first.y) / first_diagonal,
                )
        return maximum

    def evaluate(
        self,
        *,
        active_first_ordinal: int,
        active_last_ordinal: int,
        freeze_through_ordinal: int,
    ) -> dict[str, Any]:
        if (
            active_first_ordinal < 0
            or active_last_ordinal < active_first_ordinal
            or freeze_through_ordinal < active_first_ordinal
            or freeze_through_ordinal >= active_last_ordinal
        ):
            raise ActiveTrackStoreError("active/freeze interval is invalid")
        components = self._components()
        component_edges = self._component_edges()
        candidates: list[dict[str, Any]] = []
        rejected = Counter()
        for track_uid, all_observations in components.items():
            observations = [
                value
                for value in all_observations
                if active_first_ordinal <= value.geometry_ordinal <= active_last_ordinal
            ]
            frame_ordinals = sorted({value.geometry_ordinal for value in observations})
            reasons: list[str] = []
            if len(frame_ordinals) < self.budget.minimum_track_views:
                reasons.append("insufficient-views")
            parallax = self._parallax(observations)
            if parallax < self.budget.minimum_parallax_diagonals:
                reasons.append("insufficient-parallax")
            edges = component_edges.get(track_uid, [])
            maximum_error = max(
                (value.match_error_pixels for value in edges), default=0.0
            )
            if maximum_error > self.budget.maximum_match_error_pixels:
                reasons.append("match-error")
            if reasons:
                rejected.update(reasons)
                continue
            bridge = (
                frame_ordinals[0] <= freeze_through_ordinal < frame_ordinals[-1]
            )
            representative = observations[-1]
            bucket = (
                floor(representative.seconds / self.budget.selection_time_bin_seconds),
                *self._grid_cell(representative),
            )
            candidates.append(
                {
                    "trackUid": track_uid,
                    "observationUids": [
                        value.observation_uid for value in observations
                    ],
                    "geometryOrdinals": frame_ordinals,
                    "viewCount": len(frame_ordinals),
                    "bridge": bridge,
                    "parallaxDiagonals": parallax,
                    "maximumMatchErrorPixels": maximum_error,
                    "meanScore": sum(value.score for value in observations)
                    / len(observations),
                    "bucket": bucket,
                }
            )

        selected: list[dict[str, Any]] = []
        bucket_counts: Counter[tuple[int, int, int]] = Counter()
        remaining = list(candidates)
        quality = lambda value: (
            not value["bridge"],
            -value["viewCount"],
            -value["parallaxDiagonals"],
            -value["meanScore"],
            value["maximumMatchErrorPixels"],
            value["trackUid"],
        )
        remaining.sort(key=quality)
        while remaining and len(selected) < self.budget.maximum_active_tracks:
            progressed = False
            for bucket in sorted({value["bucket"] for value in remaining}):
                if bucket_counts[bucket] >= self.budget.maximum_tracks_per_grid_cell:
                    continue
                candidate = min(
                    (value for value in remaining if value["bucket"] == bucket),
                    key=quality,
                )
                remaining.remove(candidate)
                selected.append(candidate)
                bucket_counts[bucket] += 1
                progressed = True
                if len(selected) >= self.budget.maximum_active_tracks:
                    break
            if not progressed:
                break
        rejected["active-budget-or-cell-cap"] += len(remaining)

        constraints_per_frame = Counter()
        for track in selected:
            constraints_per_frame.update(track["geometryOrdinals"])
        frame_constraints = {
            ordinal: constraints_per_frame[ordinal]
            for ordinal in range(active_first_ordinal, active_last_ordinal + 1)
        }
        zero_constraint_ordinals = [
            ordinal for ordinal, count in frame_constraints.items() if count == 0
        ]
        under_constraint_ordinals = [
            ordinal
            for ordinal, count in frame_constraints.items()
            if count < self.budget.minimum_constraints_per_keyframe
        ]
        bridge_tracks = [value for value in selected if value["bridge"]]
        reason_codes: list[str] = []
        if not selected or zero_constraint_ordinals:
            reason_codes.append("ZERO_CONSTRAINT")
        if len(bridge_tracks) < self.budget.minimum_bridge_tracks:
            reason_codes.append("BRIDGE_TRACKS_BELOW_MINIMUM")
        if under_constraint_ordinals:
            reason_codes.append("KEYFRAME_CONSTRAINTS_BELOW_MINIMUM")
        status = "passed" if not reason_codes else "failed"
        report = {
            "contractId": "jarailsense.gluemap-active-track-gate/v1",
            "status": status,
            "reasonCodes": reason_codes,
            "storeIdentitySha256": self.store_identity_sha256,
            "activeFirstOrdinal": active_first_ordinal,
            "activeLastOrdinal": active_last_ordinal,
            "freezeThroughOrdinal": freeze_through_ordinal,
            "candidateTrackCount": len(candidates),
            "selectedTrackCount": len(selected),
            "bridgeTrackCount": len(bridge_tracks),
            "maximumActiveTracks": self.budget.maximum_active_tracks,
            "observationCount": self.observation_count,
            "edgeCount": self.edge_count,
            "selectedTrackUids": [value["trackUid"] for value in selected],
            "selectedTrackUidsSha256": _canonical_sha256(
                [value["trackUid"] for value in selected]
            ),
            "bridgeTrackUids": [value["trackUid"] for value in bridge_tracks],
            "constraintsPerFrame": frame_constraints,
            "zeroConstraintOrdinals": zero_constraint_ordinals,
            "underConstraintOrdinals": under_constraint_ordinals,
            "timeGridBucketCounts": {
                f"{bucket[0]}:{bucket[1]}:{bucket[2]}": count
                for bucket, count in sorted(bucket_counts.items())
            },
            "rejectedReasonHistogram": dict(sorted(rejected.items())),
            "pixelArtifactCount": 0,
        }
        report["reportSha256"] = _canonical_sha256(report)
        return report

    def propose_release(self, finalized_through_ordinal: int) -> dict[str, Any]:
        if self._pending_release is not None:
            raise ActiveTrackStoreError("a release proposal is already pending")
        if finalized_through_ordinal < 0:
            raise ActiveTrackStoreError("release ordinal is invalid")
        release_uids = sorted(
            uid
            for uid, value in self._observations.items()
            if value.geometry_ordinal <= finalized_through_ordinal
        )
        payload = {
            "contractId": "jarailsense.gluemap-active-track-release-proposal/v1",
            "storeIdentitySha256": self.store_identity_sha256,
            "previousAcceptedJournalHead": self._last_accepted_journal_head,
            "finalizedThroughOrdinal": finalized_through_ordinal,
            "releasedObservationUidsSha256": _canonical_sha256(release_uids),
            "releasedObservationCount": len(release_uids),
        }
        proposal_uid = _canonical_sha256(payload)
        self._pending_release = {
            **payload,
            "proposalUid": proposal_uid,
            "uids": release_uids,
        }
        return {
            key: value
            for key, value in self._pending_release.items()
            if key != "uids"
        }

    def commit_release(
        self, proposal_uid: str, accepted_journal_head: str
    ) -> dict[str, Any]:
        if (
            self._pending_release is None
            or self._pending_release["proposalUid"] != proposal_uid
        ):
            raise ActiveTrackStoreError("release proposal identity differs")
        _require_sha256(accepted_journal_head, "accepted journal head")
        release_uids = set(self._pending_release["uids"])
        for uid in release_uids:
            observation = self._observations.pop(uid)
            frame_values = self._observations_by_frame[observation.geometry_ordinal]
            frame_values.remove(uid)
            if not frame_values:
                del self._observations_by_frame[observation.geometry_ordinal]
        self._edges = {
            key: value
            for key, value in self._edges.items()
            if key[0] not in release_uids and key[1] not in release_uids
        }
        self._rebuild_components()
        result = {
            "contractId": "jarailsense.gluemap-active-track-release/v1",
            "status": "passed",
            "proposalUid": proposal_uid,
            "acceptedJournalHead": accepted_journal_head,
            "releasedObservationCount": len(release_uids),
            "remainingObservationCount": self.observation_count,
            "remainingEdgeCount": self.edge_count,
        }
        self._last_accepted_journal_head = accepted_journal_head
        self._pending_release = None
        return result

    def _rebuild_components(self) -> None:
        previous_component = {
            uid: self._component_uid_by_root[self._union_find.find(uid)]
            for uid in self._observations
        }
        self._union_find = UnionFind()
        self._component_uid_by_root = {}
        grouped: dict[str, list[str]] = defaultdict(list)
        for uid, component_uid in previous_component.items():
            grouped[component_uid].append(uid)
        for component_uid, uids in grouped.items():
            first = min(uids)
            self._union_find.find(first)
            for uid in sorted(uids):
                self._union_find.union(first, uid)
            root = self._union_find.find(first)
            self._component_uid_by_root[root] = component_uid
