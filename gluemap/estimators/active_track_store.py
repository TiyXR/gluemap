"""Bounded active-window track ownership and coverage gates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
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
    maximum_candidate_observations_per_keyframe: int
    active_track_budget_per_keyframe: int
    minimum_constraints_per_keyframe: int
    minimum_bridge_tracks: int
    minimum_track_views: int
    coverage_grid_columns: int
    coverage_grid_rows: int
    selection_time_bin_seconds: float
    maximum_tracks_per_grid_cell: int
    intra_image_merge_radius_pixels: float
    minimum_visibility: float
    minimum_confidence: float
    minimum_parallax_diagonals: float

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
        self._spatial_buckets_by_frame: dict[
            int, dict[tuple[int, int], set[str]]
        ] = defaultdict(lambda: defaultdict(set))
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
            "maximum_candidate_observations_per_keyframe": (
                self.budget.maximum_candidate_observations_per_keyframe,
                1,
            ),
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
            or not isfinite(self.budget.intra_image_merge_radius_pixels)
            or self.budget.intra_image_merge_radius_pixels <= 0
            or not isfinite(self.budget.minimum_parallax_diagonals)
            or self.budget.minimum_parallax_diagonals < 0
            or not isfinite(self.budget.minimum_visibility)
            or not 0 <= self.budget.minimum_visibility <= 1
            or not isfinite(self.budget.minimum_confidence)
            or self.budget.minimum_confidence < 0
        ):
            raise ActiveTrackStoreError("track metric thresholds are invalid")
        if (
            self.budget.minimum_constraints_per_keyframe
            > self.budget.maximum_active_tracks
        ):
            raise ActiveTrackStoreError("minimum constraints exceed the active budget")
        if (
            self.budget.maximum_candidate_observations_per_keyframe
            < self.budget.query_tracks_per_keyframe
        ):
            raise ActiveTrackStoreError(
                "candidate observation bound is below the query track count"
            )

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
            if (
                len(frame_values)
                >= self.budget.maximum_candidate_observations_per_keyframe
            ):
                raise ActiveTrackStoreError("per-keyframe observation bound exceeded")
            self._observations[value.observation_uid] = value
            frame_values.add(value.observation_uid)
            self._add_spatial_observation(value)
            root = self._union_find.find(value.observation_uid)
            self._component_uid_by_root[root] = value.observation_uid

    def _spatial_bucket(self, value: TrackObservation) -> tuple[int, int]:
        radius = self.budget.intra_image_merge_radius_pixels
        return floor(value.x / radius), floor(value.y / radius)

    def _add_spatial_observation(self, value: TrackObservation) -> None:
        self._spatial_buckets_by_frame[value.geometry_ordinal][
            self._spatial_bucket(value)
        ].add(value.observation_uid)

    def intern_observation(self, value: TrackObservation) -> str:
        """Return a stable nearby observation UID or insert ``value``.

        Overlapping Stars often predict the same image keypoint more than once.
        The caller processes Stars in deterministic center order; this method
        merges nearby predictions without materializing a per-Star keypoint
        file or allowing overlap degree to multiply resident state.
        """

        self._validate_observation(value)
        bucket = self._spatial_bucket(value)
        radius_squared = self.budget.intra_image_merge_radius_pixels**2
        candidates: list[tuple[float, str]] = []
        frame_buckets = self._spatial_buckets_by_frame[value.geometry_ordinal]
        for column_delta in (-1, 0, 1):
            for row_delta in (-1, 0, 1):
                for uid in frame_buckets.get(
                    (bucket[0] + column_delta, bucket[1] + row_delta), ()
                ):
                    existing = self._observations[uid]
                    distance_squared = (existing.x - value.x) ** 2 + (
                        existing.y - value.y
                    ) ** 2
                    if distance_squared <= radius_squared:
                        candidates.append((distance_squared, uid))
        if candidates:
            return min(candidates)[1]
        self.add_observations([value])
        return value.observation_uid

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
            ):
                raise ActiveTrackStoreError("track correspondence is invalid")
            key = tuple(sorted((first, second)))
            existing = self._edges.get(key)
            if existing is not None:
                if existing != value and existing != TrackCorrespondence(
                    second, first
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
        if len(observations) < 2:
            return 0.0
        diagonal = hypot(
            observations[0].image_width,
            observations[0].image_height,
        )
        if diagonal <= 0:
            return 0.0
        points = sorted({(value.x, value.y) for value in observations})
        if len(points) < 2:
            return 0.0

        def cross(
            origin: tuple[float, float],
            first: tuple[float, float],
            second: tuple[float, float],
        ) -> float:
            return (first[0] - origin[0]) * (second[1] - origin[1]) - (
                first[1] - origin[1]
            ) * (second[0] - origin[0])

        lower: list[tuple[float, float]] = []
        for point in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
                lower.pop()
            lower.append(point)
        upper: list[tuple[float, float]] = []
        for point in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
                upper.pop()
            upper.append(point)
        hull = lower[:-1] + upper[:-1]
        maximum = 0.0
        for index, first in enumerate(hull):
            for second in hull[index + 1 :]:
                maximum = max(
                    maximum,
                    hypot(second[0] - first[0], second[1] - first[1]),
                )
        return maximum / diagonal

    def evaluate(
        self,
        *,
        active_first_ordinal: int,
        active_last_ordinal: int,
        freeze_through_ordinal: int,
    ) -> dict[str, Any]:
        return self._evaluate_components(
            self._components(),
            active_first_ordinal=active_first_ordinal,
            active_last_ordinal=active_last_ordinal,
            freeze_through_ordinal=freeze_through_ordinal,
        )

    def evaluate_batch(
        self,
        intervals: Iterable[tuple[int, int, int]],
    ) -> list[dict[str, Any]]:
        """Evaluate one release group while building components only once."""
        values = list(intervals)
        components = self._components()
        return [
            self._evaluate_components(
                components,
                active_first_ordinal=active_first,
                active_last_ordinal=active_last,
                freeze_through_ordinal=freeze_through,
            )
            for active_first, active_last, freeze_through in values
        ]

    def _evaluate_components(
        self,
        components: dict[str, list[TrackObservation]],
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
            value["trackUid"],
        )
        remaining.sort(key=quality)
        candidates_by_bucket: dict[
            tuple[int, int, int], deque[dict[str, Any]]
        ] = defaultdict(deque)
        for candidate in remaining:
            candidates_by_bucket[candidate["bucket"]].append(candidate)
        while (
            candidates_by_bucket
            and len(selected) < self.budget.maximum_active_tracks
        ):
            progressed = False
            empty_buckets: list[tuple[int, int, int]] = []
            for bucket in sorted(candidates_by_bucket):
                if bucket_counts[bucket] >= self.budget.maximum_tracks_per_grid_cell:
                    continue
                values = candidates_by_bucket[bucket]
                candidate = values.popleft()
                selected.append(candidate)
                bucket_counts[bucket] += 1
                progressed = True
                if not values:
                    empty_buckets.append(bucket)
                if len(selected) >= self.budget.maximum_active_tracks:
                    break
            for bucket in empty_buckets:
                del candidates_by_bucket[bucket]
            if not progressed:
                break
        rejected["active-budget-or-cell-cap"] += sum(
            len(values) for values in candidates_by_bucket.values()
        )

        constraints_per_frame = Counter()
        for track in selected:
            constraints_per_frame.update(track["geometryOrdinals"])
        frame_constraints = {
            str(ordinal): constraints_per_frame[ordinal]
            for ordinal in range(active_first_ordinal, active_last_ordinal + 1)
        }
        zero_constraint_ordinals = [
            ordinal
            for ordinal in range(active_first_ordinal, active_last_ordinal + 1)
            if constraints_per_frame[ordinal] == 0
        ]
        under_constraint_ordinals = [
            ordinal
            for ordinal in range(active_first_ordinal, active_last_ordinal + 1)
            if constraints_per_frame[ordinal]
            < self.budget.minimum_constraints_per_keyframe
        ]
        bridge_tracks = [value for value in selected if value["bridge"]]
        selected_track_uids = [value["trackUid"] for value in selected]
        bridge_track_uids = [value["trackUid"] for value in bridge_tracks]
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
            "selectedTrackUidsSha256": _canonical_sha256(selected_track_uids),
            "bridgeTrackUidsSha256": _canonical_sha256(bridge_track_uids),
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
            bucket = self._spatial_bucket(observation)
            spatial_values = self._spatial_buckets_by_frame[
                observation.geometry_ordinal
            ][bucket]
            spatial_values.remove(uid)
            if not spatial_values:
                del self._spatial_buckets_by_frame[
                    observation.geometry_ordinal
                ][bucket]
            if not self._spatial_buckets_by_frame[observation.geometry_ordinal]:
                del self._spatial_buckets_by_frame[observation.geometry_ordinal]
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
