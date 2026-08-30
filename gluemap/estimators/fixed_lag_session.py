"""State-only fixed-lag window ownership and durable release gates."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any


class FixedLagSessionError(ValueError):
    """Raised when a fixed-lag state transition violates its contract."""


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class GeometryFrame:
    geometry_ordinal: int
    frame_uid: str
    keyframe_uid: str
    pts_value: int
    time_base_numerator: int
    time_base_denominator: int

    @property
    def seconds(self) -> float:
        return (
            self.pts_value
            * self.time_base_numerator
            / self.time_base_denominator
        )


@dataclass(frozen=True)
class FrontendCacheShard:
    shard_index: int
    first_star: int
    last_star: int
    sha256: str


class FixedLagSession:
    """Own one logical Sequence while advancing one keyframe at a time.

    A proposal never releases state. ``commit_advance`` requires the SHA-256
    token of an already accepted durable index head before it can advance the
    finalized/release watermark.
    """

    def __init__(
        self,
        frames: list[GeometryFrame],
        future_dependency_ordinals: list[int],
        cache_shards: list[FrontendCacheShard],
        *,
        window_size_keyframes: int,
        anchor_band_keyframes: int,
        maximum_lookahead_keyframes: int,
        required_lookahead_keyframes: int,
        minimum_window_duration_seconds: float,
        checkpoint_interval_advances: int,
    ) -> None:
        self.frames = frames
        self.future_dependency_ordinals = future_dependency_ordinals
        self.cache_shards = cache_shards
        self.window_size_keyframes = window_size_keyframes
        self.anchor_band_keyframes = anchor_band_keyframes
        self.maximum_lookahead_keyframes = maximum_lookahead_keyframes
        self.required_lookahead_keyframes = required_lookahead_keyframes
        self.minimum_window_duration_seconds = minimum_window_duration_seconds
        self.checkpoint_interval_advances = checkpoint_interval_advances
        self.next_ingest_ordinal = 0
        self.finalized_count = 0
        self.last_accepted_index_head: str | None = None
        self.released_cache_shards: set[int] = set()
        self._pending_proposal_uid: str | None = None
        self._validate_contract()
        self.session_identity_sha256 = _canonical_sha256(self._identity())

    def _validate_contract(self) -> None:
        count = len(self.frames)
        if count == 0 or len(self.future_dependency_ordinals) != count:
            raise FixedLagSessionError("frame/future dependency set differs")
        if self.window_size_keyframes < 2:
            raise FixedLagSessionError("window size must be at least two")
        if not 0 < self.anchor_band_keyframes < self.window_size_keyframes:
            raise FixedLagSessionError("anchor band must be inside the window")
        if not 0 <= self.required_lookahead_keyframes <= (
            self.maximum_lookahead_keyframes
        ):
            raise FixedLagSessionError("required lookahead exceeds its bound")
        if self.minimum_window_duration_seconds <= 0:
            raise FixedLagSessionError(
                "minimum window duration must be positive"
            )
        if self.checkpoint_interval_advances < 1:
            raise FixedLagSessionError("checkpoint interval must be positive")
        for ordinal, (frame, dependency) in enumerate(
            zip(self.frames, self.future_dependency_ordinals, strict=True)
        ):
            if (
                frame.geometry_ordinal != ordinal
                or not frame.frame_uid
                or not frame.keyframe_uid
                or frame.time_base_numerator <= 0
                or frame.time_base_denominator <= 0
                or dependency < ordinal
                or dependency >= count
                or dependency - ordinal > self.maximum_lookahead_keyframes
            ):
                raise FixedLagSessionError(
                    "frame/future dependency identity differs"
                )
            if ordinal and frame.seconds <= self.frames[ordinal - 1].seconds:
                raise FixedLagSessionError(
                    "geometry PTS order is not increasing"
                )
        previous_last = -1
        for index, shard in enumerate(self.cache_shards):
            if (
                shard.shard_index != index
                or shard.first_star != previous_last + 1
                or shard.last_star < shard.first_star
                or shard.last_star >= count
                or len(shard.sha256) != 64
            ):
                raise FixedLagSessionError(
                    "frontend cache shard coverage differs"
                )
            previous_last = shard.last_star
        if self.cache_shards and previous_last != count - 1:
            raise FixedLagSessionError(
                "frontend cache shards do not cover frames"
            )

    def _identity(self) -> dict[str, Any]:
        return {
            "frames": [asdict(frame) for frame in self.frames],
            "futureDependencyOrdinals": self.future_dependency_ordinals,
            "cacheShards": [asdict(shard) for shard in self.cache_shards],
            "windowSizeKeyframes": self.window_size_keyframes,
            "anchorBandKeyframes": self.anchor_band_keyframes,
            "maximumLookaheadKeyframes": self.maximum_lookahead_keyframes,
            "requiredLookaheadKeyframes": self.required_lookahead_keyframes,
            "minimumWindowDurationSeconds": (
                self.minimum_window_duration_seconds
            ),
            "checkpointIntervalAdvances": self.checkpoint_interval_advances,
        }

    @property
    def resident_count(self) -> int:
        return self.next_ingest_ordinal - self.finalized_count

    def ingest_next(self) -> dict[str, Any]:
        if self._pending_proposal_uid is not None:
            raise FixedLagSessionError(
                "cannot ingest while an advance is pending"
            )
        if self.next_ingest_ordinal >= len(self.frames):
            raise FixedLagSessionError(
                "all geometry frames are already ingested"
            )
        frame = self.frames[self.next_ingest_ordinal]
        self.next_ingest_ordinal += 1
        return {
            "geometryOrdinal": frame.geometry_ordinal,
            "frameUid": frame.frame_uid,
            "residentCount": self.resident_count,
        }

    def _window_duration_seconds(self) -> float:
        start = self.finalized_count
        end = start + self.window_size_keyframes - 1
        if end >= self.next_ingest_ordinal:
            return 0.0
        return self.frames[end].seconds - self.frames[start].seconds

    def can_advance(self) -> bool:
        if self._pending_proposal_uid is not None:
            return False
        minimum_resident = (
            self.window_size_keyframes + self.required_lookahead_keyframes
        )
        if self.resident_count < minimum_resident:
            return False
        candidate = self.finalized_count
        return (
            self.future_dependency_ordinals[candidate]
            < self.next_ingest_ordinal
            and self._window_duration_seconds()
            >= self.minimum_window_duration_seconds
        )

    def role_state(self) -> dict[str, Any]:
        resident = list(range(self.finalized_count, self.next_ingest_ordinal))
        active = resident[: self.window_size_keyframes]
        anchor = active[: self.anchor_band_keyframes]
        mutable = active[self.anchor_band_keyframes :]
        lookahead = resident[self.window_size_keyframes :]
        return {
            "finalizedCount": self.finalized_count,
            "nextIngestOrdinal": self.next_ingest_ordinal,
            "anchorOrdinals": anchor,
            "activeBodyOrdinals": mutable,
            "lookaheadOrdinals": lookahead,
            "residentCount": len(resident),
            "releasedCacheShardIndexes": sorted(self.released_cache_shards),
            "lastAcceptedIndexHead": self.last_accepted_index_head,
        }

    def propose_advance(self) -> dict[str, Any]:
        if not self.can_advance():
            raise FixedLagSessionError(
                "fixed-lag advance is not currently legal"
            )
        candidate = self.frames[self.finalized_count]
        payload = {
            "sessionIdentitySha256": self.session_identity_sha256,
            "finalizedCountBefore": self.finalized_count,
            "nextIngestOrdinal": self.next_ingest_ordinal,
            "candidateFrameUid": candidate.frame_uid,
            "candidateKeyframeUid": candidate.keyframe_uid,
            "previousAcceptedIndexHead": self.last_accepted_index_head,
        }
        proposal_uid = _canonical_sha256(payload)
        self._pending_proposal_uid = proposal_uid
        return {
            **payload,
            "proposalUid": proposal_uid,
            "windowDurationSeconds": self._window_duration_seconds(),
            "stateBefore": self.role_state(),
        }

    def commit_advance(
        self, proposal_uid: str, accepted_index_head_sha256: str
    ) -> dict[str, Any]:
        if proposal_uid != self._pending_proposal_uid:
            raise FixedLagSessionError("advance proposal identity differs")
        if (
            len(accepted_index_head_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in accepted_index_head_sha256
            )
        ):
            raise FixedLagSessionError("accepted index head token is invalid")
        self.finalized_count += 1
        self.last_accepted_index_head = accepted_index_head_sha256
        self._pending_proposal_uid = None
        released_now: list[int] = []
        for shard in self.cache_shards:
            if (
                shard.shard_index not in self.released_cache_shards
                and shard.last_star < self.finalized_count
            ):
                self.released_cache_shards.add(shard.shard_index)
                released_now.append(shard.shard_index)
        return {
            "acceptedIndexHead": accepted_index_head_sha256,
            "releasedCacheShardIndexes": released_now,
            "checkpointDue": (
                self.finalized_count % self.checkpoint_interval_advances == 0
            ),
            "stateAfter": self.role_state(),
        }

    def snapshot(self) -> dict[str, Any]:
        if self._pending_proposal_uid is not None:
            raise FixedLagSessionError("cannot checkpoint a pending advance")
        state = {
            "contractId": "jarailsense.gluemap-fixed-lag-state/v1",
            "sessionIdentitySha256": self.session_identity_sha256,
            "nextIngestOrdinal": self.next_ingest_ordinal,
            "finalizedCount": self.finalized_count,
            "lastAcceptedIndexHead": self.last_accepted_index_head,
            "releasedCacheShardIndexes": sorted(self.released_cache_shards),
        }
        return {**state, "stateSha256": _canonical_sha256(state)}

    def restore(self, snapshot: dict[str, Any]) -> None:
        state = {
            key: value
            for key, value in snapshot.items()
            if key != "stateSha256"
        }
        if (
            snapshot.get("contractId")
            != "jarailsense.gluemap-fixed-lag-state/v1"
            or snapshot.get("sessionIdentitySha256")
            != self.session_identity_sha256
            or snapshot.get("stateSha256") != _canonical_sha256(state)
        ):
            raise FixedLagSessionError("fixed-lag checkpoint identity differs")
        next_ingest = snapshot.get("nextIngestOrdinal")
        finalized = snapshot.get("finalizedCount")
        released = snapshot.get("releasedCacheShardIndexes")
        if (
            not isinstance(next_ingest, int)
            or not isinstance(finalized, int)
            or not 0 <= finalized <= next_ingest <= len(self.frames)
            or not isinstance(released, list)
            or any(
                not isinstance(value, int)
                or value < 0
                or value >= len(self.cache_shards)
                for value in released
            )
        ):
            raise FixedLagSessionError("fixed-lag checkpoint state is invalid")
        self.next_ingest_ordinal = next_ingest
        self.finalized_count = finalized
        self.last_accepted_index_head = snapshot.get("lastAcceptedIndexHead")
        self.released_cache_shards = set(released)
        self._pending_proposal_uid = None
