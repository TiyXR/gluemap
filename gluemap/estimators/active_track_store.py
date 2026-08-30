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


def _load_track_native_module() -> Any:
    try:
        import pygluemap_tracks

        return pygluemap_tracks
    except (ImportError, OSError):
        import pygluemap

        return pygluemap


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
    parallax_backend_policy: str = "cuda-preferred"
    parallax_microbatch_components: int = 256

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


@dataclass(frozen=True)
class SelectedTrackState:
    """One in-memory BA track selected by the existing GPU gate."""

    track_uid: str
    observations: tuple[TrackObservation, ...]


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
        self._edges: set[tuple[str, str]] = set()
        self._union_find = UnionFind()
        self._component_uid_by_root: dict[Any, str] = {}
        self._last_accepted_journal_head: str | None = None
        self._pending_release: dict[str, Any] | None = None
        self._parallax_backend = self._resolve_parallax_backend()
        self._component_rebuild_backend = self._resolve_component_rebuild_backend()
        self._spatial_intern_backend = self._resolve_spatial_intern_backend()
        self.store_identity_sha256 = _canonical_sha256(asdict(budget))

    @staticmethod
    def _resolve_component_rebuild_backend() -> str:
        try:
            native = _load_track_native_module()

            if hasattr(native, "compute_connected_components"):
                return "native-openmp"
        except (ImportError, OSError):
            pass
        return "python"

    @staticmethod
    def _resolve_spatial_intern_backend() -> str:
        try:
            native = _load_track_native_module()

            if hasattr(native, "batch_spatial_intern"):
                return "native-openmp"
        except (ImportError, OSError):
            pass
        return "python"

    def _resolve_parallax_backend(self) -> str:
        import torch

        policy = self.budget.parallax_backend_policy
        if policy not in {"cuda-required", "cuda-preferred", "cpu"}:
            raise ActiveTrackStoreError("parallax backend policy is invalid")
        if policy == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
        if policy == "cuda-required":
            raise ActiveTrackStoreError("CUDA parallax backend is unavailable")
        return "cpu"

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
            "parallax_microbatch_components": (
                self.budget.parallax_microbatch_components,
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

    @property
    def parallax_backend(self) -> str:
        return self._parallax_backend

    @property
    def gpu_used(self) -> bool:
        return self._parallax_backend == "cuda"

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
            self._insert_observation(value)

    def _insert_observation(self, value: TrackObservation) -> None:
        """Insert one already validated, previously unseen observation."""
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

    def intern_observations(
        self,
        values: Iterable[TrackObservation],
        *,
        assume_valid: bool = False,
    ) -> list[str]:
        """Intern one deterministic Star batch without per-point API calls."""
        if self._pending_release is not None:
            raise ActiveTrackStoreError("cannot ingest while release is pending")
        batch = list(values)
        if not assume_valid:
            for value in batch:
                self._validate_observation(value)
        if self._spatial_intern_backend == "native-openmp" and batch:
            return self._intern_observations_native(batch)
        radius_squared = self.budget.intra_image_merge_radius_pixels**2
        candidates: list[tuple[float, str, str, int, str]] = []
        for index, value in enumerate(batch):
            existing = self._observations.get(value.observation_uid)
            if existing is not None and existing != value:
                raise ActiveTrackStoreError("observation identity was reused")
            bucket = self._spatial_bucket(value)
            frame_buckets = self._spatial_buckets_by_frame[
                value.geometry_ordinal
            ]
            for column_delta in (-1, 0, 1):
                for row_delta in (-1, 0, 1):
                    for uid in frame_buckets.get(
                        (bucket[0] + column_delta, bucket[1] + row_delta), ()
                    ):
                        candidate = self._observations[uid]
                        distance_squared = (candidate.x - value.x) ** 2 + (
                            candidate.y - value.y
                        ) ** 2
                        if distance_squared <= radius_squared:
                            candidates.append(
                                (
                                    distance_squared,
                                    uid,
                                    value.observation_uid,
                                    index,
                                    uid,
                                )
                            )
        candidates.sort()
        resolved_uids: list[str | None] = [None] * len(batch)
        used_existing: set[str] = set()
        for _, _, _, index, uid in candidates:
            if resolved_uids[index] is not None or uid in used_existing:
                continue
            resolved_uids[index] = uid
            used_existing.add(uid)
        for index, value in enumerate(batch):
            if resolved_uids[index] is not None:
                continue
            self._insert_observation(value)
            resolved_uids[index] = value.observation_uid
        if any(value is None for value in resolved_uids):
            raise ActiveTrackStoreError("spatial intern result is incomplete")
        return [str(value) for value in resolved_uids]

    def _intern_observations_native(
        self, values: list[TrackObservation]
    ) -> list[str]:
        import numpy as np
        native = _load_track_native_module()

        relevant_frames = {value.geometry_ordinal for value in values}
        existing_uids = sorted(
            uid
            for frame in relevant_frames
            for uid in self._observations_by_frame.get(frame, ())
        )
        existing_values = [self._observations[uid] for uid in existing_uids]
        representatives = native.batch_spatial_intern(
            np.fromiter(
                (value.geometry_ordinal for value in existing_values),
                dtype=np.int64,
            ),
            np.fromiter((value.x for value in existing_values), dtype=np.float64),
            np.fromiter((value.y for value in existing_values), dtype=np.float64),
            existing_uids,
            np.fromiter(
                (value.geometry_ordinal for value in values), dtype=np.int64
            ),
            np.fromiter((value.x for value in values), dtype=np.float64),
            np.fromiter((value.y for value in values), dtype=np.float64),
            [value.observation_uid for value in values],
            self.budget.intra_image_merge_radius_pixels,
        ).tolist()
        existing_count = len(existing_uids)
        resolved_uids: list[str] = []
        for index, (value, representative) in enumerate(
            zip(values, representatives, strict=True)
        ):
            if representative < existing_count:
                resolved_uids.append(existing_uids[representative])
                continue
            representative_index = representative - existing_count
            if representative_index != index:
                raise ActiveTrackStoreError(
                    "native spatial intern merged observations within one Star"
                )
            existing = self._observations.get(value.observation_uid)
            if existing is not None:
                if existing != value:
                    raise ActiveTrackStoreError("observation identity was reused")
            else:
                self._insert_observation(value)
            resolved_uids.append(value.observation_uid)
        return resolved_uids

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

        return self.intern_observations([value])[0]

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
        self.add_correspondence_pairs(
            (
                (value.first_observation_uid, value.second_observation_uid)
                for value in values
            )
        )

    def add_correspondence_pairs(
        self, values: Iterable[tuple[str, str]]
    ) -> None:
        if self._pending_release is not None:
            raise ActiveTrackStoreError("cannot ingest while release is pending")
        for first, second in values:
            if (
                first == second
                or first not in self._observations
                or second not in self._observations
            ):
                raise ActiveTrackStoreError("track correspondence is invalid")
            key = (first, second) if first < second else (second, first)
            if key in self._edges:
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
            self._edges.add(key)

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
        return self.evaluate_batch(
            [
                (
                    active_first_ordinal,
                    active_last_ordinal,
                    freeze_through_ordinal,
                )
            ]
        )[0]

    def evaluate_batch(
        self,
        intervals: Iterable[tuple[int, int, int]],
    ) -> list[dict[str, Any]]:
        """Evaluate one release group with one tensor upload and one reduction."""
        reports, _ = self._evaluate_batch_impl(intervals, materialize_tracks=False)
        return reports

    def evaluate_batch_materialized(
        self,
        intervals: Iterable[tuple[int, int, int]],
    ) -> tuple[list[dict[str, Any]], list[list[SelectedTrackState]]]:
        """Return ephemeral selected observations from the same gate pass."""
        return self._evaluate_batch_impl(intervals, materialize_tracks=True)

    def evaluate_frame_sets_materialized(
        self,
        windows: Iterable[tuple[Iterable[int], int]],
    ) -> tuple[list[dict[str, Any]], list[list[SelectedTrackState]]]:
        """Evaluate arbitrary fixed-lag frame sets in one GPU membership batch."""
        values: list[tuple[int, ...]] = []
        intervals: list[tuple[int, int, int]] = []
        for frame_ids, freeze_through in windows:
            ordered = tuple(sorted(set(int(value) for value in frame_ids)))
            if (
                len(ordered) < 2
                or ordered[0] < 0
                or freeze_through not in ordered
                or freeze_through == ordered[-1]
            ):
                raise ActiveTrackStoreError(
                    "active frame-set/freeze identity is invalid"
                )
            values.append(ordered)
            intervals.append((ordered[0], ordered[-1], freeze_through))
        return self._evaluate_batch_impl(
            intervals,
            materialize_tracks=True,
            active_frame_sets=values,
        )

    def _evaluate_batch_impl(
        self,
        intervals: Iterable[tuple[int, int, int]],
        *,
        materialize_tracks: bool,
        active_frame_sets: list[tuple[int, ...]] | None = None,
    ) -> tuple[list[dict[str, Any]], list[list[SelectedTrackState]]]:
        values = list(intervals)
        for active_first, active_last, freeze_through in values:
            self._validate_interval(active_first, active_last, freeze_through)
        components = self._components()
        if not values:
            return [], []
        if not components:
            return (
                [
                    self._build_report(
                        interval=value,
                        candidates=[],
                        selected=[],
                        bucket_counts=Counter(),
                        rejected=Counter({"active-budget-or-cell-cap": 0}),
                        constraints_per_frame=Counter(),
                    )
                    for value in values
                ],
                [[] for _ in values],
            )
        frame_sets = active_frame_sets or [
            tuple(range(first, last + 1)) for first, last, _ in values
        ]
        if len(frame_sets) != len(values):
            raise ActiveTrackStoreError("active frame-set batch size differs")

        (
            candidates_by_interval,
            rejected_by_interval,
            ordinals,
            unique_active_observations,
        ) = self._batch_component_metrics(components, values, frame_sets)
        selected_by_interval: list[list[dict[str, Any]]] = []
        bucket_counts_by_interval: list[Counter[tuple[int, int, int]]] = []
        selected_component_indexes: list[list[int]] = []
        for candidates, rejected in zip(
            candidates_by_interval, rejected_by_interval, strict=True
        ):
            selected, bucket_counts = self._select_candidates(candidates, rejected)
            selected_by_interval.append(selected)
            bucket_counts_by_interval.append(bucket_counts)
            selected_component_indexes.append(
                [value["componentIndex"] for value in selected]
            )
        constraints_by_interval = self._batch_constraints_per_frame(
            ordinals=ordinals,
            unique_active_observations=unique_active_observations,
            selected_component_indexes=selected_component_indexes,
            intervals=values,
            frame_sets=frame_sets,
        )
        reports = []
        materialized_by_interval: list[list[SelectedTrackState]] = []
        component_values = list(components.items())
        for index, interval in enumerate(values):
            selected = selected_by_interval[index]
            materialized = []
            if materialize_tracks:
                active_frames = set(frame_sets[index])
                for candidate in selected:
                    component_uid, component_observations = component_values[
                        candidate["componentIndex"]
                    ]
                    observation_by_frame: dict[int, TrackObservation] = {}
                    for observation in component_observations:
                        if observation.geometry_ordinal in active_frames:
                            observation_by_frame.setdefault(
                                observation.geometry_ordinal, observation
                            )
                    materialized.append(
                        SelectedTrackState(
                            track_uid=component_uid,
                            observations=tuple(observation_by_frame.values()),
                        )
                    )
            materialized_by_interval.append(materialized)
            for candidate in selected:
                candidate.pop("componentIndex", None)
            reports.append(
                self._build_report(
                    interval=interval,
                    candidates=candidates_by_interval[index],
                    selected=selected,
                    bucket_counts=bucket_counts_by_interval[index],
                    rejected=rejected_by_interval[index],
                    constraints_per_frame=constraints_by_interval[index],
                    frame_ids=frame_sets[index],
                )
            )
        return reports, materialized_by_interval

    @staticmethod
    def _validate_interval(
        active_first_ordinal: int,
        active_last_ordinal: int,
        freeze_through_ordinal: int,
    ) -> None:
        if (
            active_first_ordinal < 0
            or active_last_ordinal < active_first_ordinal
            or freeze_through_ordinal < active_first_ordinal
            or freeze_through_ordinal >= active_last_ordinal
        ):
            raise ActiveTrackStoreError("active/freeze interval is invalid")

    def _batch_component_metrics(
        self,
        components: dict[str, list[TrackObservation]],
        intervals: list[tuple[int, int, int]],
        frame_sets: list[tuple[int, ...]],
    ) -> tuple[
        list[list[dict[str, Any]]],
        list[Counter[str]],
        Any,
        Any,
    ]:
        """Compute every read-only component/window metric as one tensor batch."""
        import torch

        component_values = list(components.items())
        component_uids = [value[0] for value in component_values]
        maximum_observations = max(
            len(observations) for _, observations in component_values
        )
        padded_coordinates = []
        padded_ordinals = []
        padded_dimensions = []
        padded_scores = []
        padded_seconds = []
        for _, observations in component_values:
            padding = maximum_observations - len(observations)
            padded_coordinates.append(
                [(value.x, value.y) for value in observations]
                + [(0.0, 0.0)] * padding
            )
            padded_ordinals.append(
                [value.geometry_ordinal for value in observations] + [-1] * padding
            )
            padded_dimensions.append(
                [(value.image_width, value.image_height) for value in observations]
                + [(1, 1)] * padding
            )
            padded_scores.append(
                [value.score for value in observations] + [0.0] * padding
            )
            padded_seconds.append(
                [value.seconds for value in observations] + [0.0] * padding
            )

        device = self._parallax_backend
        coordinates = torch.tensor(
            padded_coordinates, dtype=torch.float32, device=device
        )
        ordinals = torch.tensor(padded_ordinals, dtype=torch.int64, device=device)
        dimensions = torch.tensor(
            padded_dimensions, dtype=torch.float32, device=device
        )
        scores = torch.tensor(padded_scores, dtype=torch.float64, device=device)
        seconds = torch.tensor(padded_seconds, dtype=torch.float64, device=device)
        minimum_frame = min(value[0] for value in frame_sets)
        maximum_frame = max(value[-1] for value in frame_sets)
        membership = torch.zeros(
            (len(frame_sets), maximum_frame - minimum_frame + 1),
            dtype=torch.bool,
            device=device,
        )
        for index, frame_ids in enumerate(frame_sets):
            frame_indexes = (
                torch.tensor(frame_ids, device=device) - minimum_frame
            )
            membership[index, frame_indexes] = True
        shifted_ordinals = ordinals - minimum_frame
        valid_ordinals = (shifted_ordinals >= 0) & (
            shifted_ordinals < membership.shape[1]
        )
        lookup_indexes = shifted_ordinals.clamp(0, membership.shape[1] - 1)
        mask = membership[:, lookup_indexes].permute(1, 0, 2)
        mask &= valid_ordinals[:, None, :]
        previous_is_same_view = torch.zeros_like(mask)
        if maximum_observations > 1:
            previous_is_same_view[:, :, 1:] = (
                mask[:, :, :-1]
                & (ordinals[:, None, 1:] == ordinals[:, None, :-1])
            )
        unique_active_observations = mask & ~previous_is_same_view
        view_counts = unique_active_observations.sum(dim=2)
        first_indexes = unique_active_observations.to(torch.int64).argmax(dim=2)
        last_indexes = maximum_observations - 1 - torch.flip(
            unique_active_observations, dims=(2,)
        ).to(torch.int64).argmax(dim=2)
        expanded_coordinates = coordinates[:, None, :, :].expand(
            -1, len(intervals), -1, -1
        )
        gather_indexes = first_indexes[:, :, None, None].expand(-1, -1, 1, 2)
        first_points = expanded_coordinates.gather(2, gather_indexes).squeeze(2)
        maximum_views = max(1, int(view_counts.max().item()))
        compact_positions = (
            unique_active_observations.to(torch.int64).cumsum(dim=2) - 1
        ).clamp_min_(0)
        compact_points = torch.zeros(
            (
                len(component_values),
                len(intervals),
                maximum_views,
                2,
            ),
            dtype=coordinates.dtype,
            device=device,
        )
        compact_points.scatter_add_(
            2,
            compact_positions[:, :, :, None].expand(-1, -1, -1, 2),
            expanded_coordinates
            * unique_active_observations[:, :, :, None],
        )
        selected_points = compact_points.reshape(-1, maximum_views, 2)
        expanded_dimensions = dimensions[:, None, :, :].expand(
            -1, len(intervals), -1, -1
        )
        first_dimensions = expanded_dimensions.gather(
            2, gather_indexes
        ).squeeze(2)
        diagonals = torch.sqrt((first_dimensions * first_dimensions).sum(dim=2))
        diagonals = diagonals.reshape(-1)
        flat_view_counts = view_counts.reshape(-1)

        batch_size = self.budget.parallax_microbatch_components
        parallax = torch.zeros(
            len(component_values) * len(intervals),
            dtype=torch.float32,
            device=device,
        )
        with torch.no_grad():
            for start in range(0, len(selected_points), batch_size):
                points = selected_points[start : start + batch_size]
                squared_norm = (points * points).sum(dim=2)
                squared_distance = (
                    squared_norm[:, :, None]
                    + squared_norm[:, None, :]
                    - 2.0 * torch.bmm(points, points.transpose(1, 2))
                )
                maximum = torch.sqrt(
                    squared_distance.amax(dim=(1, 2)).clamp_min_(0.0)
                ) / diagonals[start : start + batch_size]
                maximum = torch.where(
                    flat_view_counts[start : start + batch_size] >= 2,
                    maximum,
                    torch.zeros_like(maximum),
                )
                parallax[start : start + len(maximum)] = maximum
        parallax = parallax.reshape(len(component_values), len(intervals))

        first_ordinals = ordinals[:, None, :].expand(
            -1, len(intervals), -1
        ).gather(2, first_indexes[:, :, None]).squeeze(2)
        last_ordinals = ordinals[:, None, :].expand(
            -1, len(intervals), -1
        ).gather(2, last_indexes[:, :, None]).squeeze(2)
        freeze_through = torch.tensor(
            [value[2] for value in intervals], dtype=torch.int64, device=device
        )
        bridges = (first_ordinals <= freeze_through[None, :]) & (
            freeze_through[None, :] < last_ordinals
        )
        mean_scores = (
            (scores[:, None, :] * unique_active_observations).sum(dim=2)
            / view_counts.clamp_min(1)
        )
        representative_indexes = last_indexes[:, :, None]
        representative_seconds = seconds[:, None, :].expand(
            -1, len(intervals), -1
        ).gather(2, representative_indexes).squeeze(2)
        representative_points = expanded_coordinates.gather(
            2, representative_indexes[:, :, :, None].expand(-1, -1, -1, 2)
        ).squeeze(2)
        representative_dimensions = expanded_dimensions.gather(
            2, representative_indexes[:, :, :, None].expand(-1, -1, -1, 2)
        ).squeeze(2)
        time_bins = torch.floor(
            representative_seconds / self.budget.selection_time_bin_seconds
        ).to(torch.int64)
        grid_columns = torch.floor(
            representative_points[:, :, 0]
            / representative_dimensions[:, :, 0]
            * self.budget.coverage_grid_columns
        ).to(torch.int64).clamp_(0, self.budget.coverage_grid_columns - 1)
        grid_rows = torch.floor(
            representative_points[:, :, 1]
            / representative_dimensions[:, :, 1]
            * self.budget.coverage_grid_rows
        ).to(torch.int64).clamp_(0, self.budget.coverage_grid_rows - 1)

        integer_metrics = torch.stack(
            (view_counts, bridges, time_bins, grid_rows, grid_columns), dim=2
        ).cpu()
        float_metrics = torch.stack(
            (parallax.to(torch.float64), mean_scores), dim=2
        ).cpu()
        candidates_by_interval: list[list[dict[str, Any]]] = [
            [] for _ in intervals
        ]
        rejected_by_interval: list[Counter[str]] = [
            Counter() for _ in intervals
        ]
        for component_index, track_uid in enumerate(component_uids):
            for interval_index in range(len(intervals)):
                view_count, bridge, time_bin, row, column = integer_metrics[
                    component_index, interval_index
                ].tolist()
                parallax_value, mean_score = float_metrics[
                    component_index, interval_index
                ].tolist()
                reasons = []
                if view_count < self.budget.minimum_track_views:
                    reasons.append("insufficient-views")
                if parallax_value < self.budget.minimum_parallax_diagonals:
                    reasons.append("insufficient-parallax")
                if reasons:
                    rejected_by_interval[interval_index].update(reasons)
                    continue
                candidates_by_interval[interval_index].append(
                    {
                        "trackUid": track_uid,
                        "componentIndex": component_index,
                        "viewCount": view_count,
                        "bridge": bool(bridge),
                        "parallaxDiagonals": parallax_value,
                        "meanScore": mean_score,
                        "bucket": (time_bin, row, column),
                    }
                )
        return (
            candidates_by_interval,
            rejected_by_interval,
            ordinals,
            unique_active_observations,
        )

    def _select_candidates(
        self,
        candidates: list[dict[str, Any]],
        rejected: Counter[str],
    ) -> tuple[list[dict[str, Any]], Counter[tuple[int, int, int]]]:
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
        return selected, bucket_counts

    def _batch_constraints_per_frame(
        self,
        *,
        ordinals: Any,
        unique_active_observations: Any,
        selected_component_indexes: list[list[int]],
        intervals: list[tuple[int, int, int]],
        frame_sets: list[tuple[int, ...]],
    ) -> list[Counter[int]]:
        import torch

        component_count, interval_count, _ = unique_active_observations.shape
        selected_mask = torch.zeros(
            (component_count, interval_count),
            dtype=torch.bool,
            device=ordinals.device,
        )
        for interval_index, indexes in enumerate(selected_component_indexes):
            if indexes:
                selected_mask[indexes, interval_index] = True
        selected_observations = (
            unique_active_observations & selected_mask[:, :, None]
        )
        first_ordinal = min(value[0] for value in frame_sets)
        last_ordinal = max(value[-1] for value in frame_sets)
        span = last_ordinal - first_ordinal + 1
        expanded_ordinals = ordinals[:, None, :].expand(-1, interval_count, -1)
        interval_offsets = torch.arange(
            interval_count, dtype=torch.int64, device=ordinals.device
        )[None, :, None] * span
        keys = (
            expanded_ordinals - first_ordinal + interval_offsets
        )[selected_observations]
        counts = torch.bincount(keys, minlength=interval_count * span).reshape(
            interval_count, span
        ).cpu()
        return [
            Counter(
                {
                    ordinal: int(counts[interval_index, ordinal - first_ordinal])
                    for ordinal in frame_sets[interval_index]
                }
            )
            for interval_index, _interval in enumerate(intervals)
        ]

    def _build_report(
        self,
        *,
        interval: tuple[int, int, int],
        candidates: list[dict[str, Any]],
        selected: list[dict[str, Any]],
        bucket_counts: Counter[tuple[int, int, int]],
        rejected: Counter[str],
        constraints_per_frame: Counter[int],
        frame_ids: tuple[int, ...] | None = None,
    ) -> dict[str, Any]:
        active_first_ordinal, active_last_ordinal, freeze_through_ordinal = interval
        active_frame_ids = frame_ids or tuple(
            range(active_first_ordinal, active_last_ordinal + 1)
        )
        frame_constraints = {
            str(ordinal): constraints_per_frame[ordinal]
            for ordinal in active_frame_ids
        }
        zero_constraint_ordinals = [
            ordinal
            for ordinal in active_frame_ids
            if constraints_per_frame[ordinal] == 0
        ]
        under_constraint_ordinals = [
            ordinal
            for ordinal in active_frame_ids
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
            "parallaxBackendPolicy": self.budget.parallax_backend_policy,
            "parallaxBackend": self.parallax_backend,
            "gateMetricsBackend": self.parallax_backend,
            "componentRebuildBackend": self._component_rebuild_backend,
            "spatialInternBackend": self._spatial_intern_backend,
            "parallaxMicrobatchComponents": (
                self.budget.parallax_microbatch_components
            ),
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
        contiguous = active_frame_ids == tuple(
            range(active_first_ordinal, active_last_ordinal + 1)
        )
        if not contiguous:
            report["activeFrameIds"] = list(active_frame_ids)
            report["activeFrameCount"] = len(active_frame_ids)
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
            key
            for key in self._edges
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
        if self._component_rebuild_backend == "native-openmp":
            self._rebuild_components_native(previous_component)
            return
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

    def _rebuild_components_native(
        self, previous_component: dict[str, str]
    ) -> None:
        import numpy as np
        native = _load_track_native_module()

        uids = list(self._observations)
        index_by_uid = {uid: index for index, uid in enumerate(uids)}
        edge_first = np.fromiter(
            (index_by_uid[key[0]] for key in self._edges), dtype=np.int64
        )
        edge_second = np.fromiter(
            (index_by_uid[key[1]] for key in self._edges), dtype=np.int64
        )
        labels = native.compute_connected_components(
            len(uids), edge_first, edge_second
        )
        grouped: dict[int, list[str]] = defaultdict(list)
        for uid, label in zip(uids, labels.tolist(), strict=True):
            grouped[label].append(uid)
        parents: dict[str, str] = {}
        component_uid_by_root: dict[str, str] = {}
        for grouped_uids in grouped.values():
            root = min(grouped_uids)
            component_uid_by_root[root] = min(
                previous_component[uid] for uid in grouped_uids
            )
            parents.update((uid, root) for uid in grouped_uids)
        self._union_find = UnionFind()
        self._union_find.parent = parents
        self._component_uid_by_root = component_uid_by_root
