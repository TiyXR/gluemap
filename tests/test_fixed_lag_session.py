import hashlib

import pytest

from gluemap.estimators.fixed_lag_session import (
    FixedLagSession,
    FixedLagSessionError,
    FrontendCacheShard,
    GeometryFrame,
)


def frames(count=20):
    return [
        GeometryFrame(
            geometry_ordinal=index,
            frame_uid=f"frame-{index}",
            keyframe_uid=f"keyframe-{index}",
            pts_value=index,
            time_base_numerator=1,
            time_base_denominator=10,
        )
        for index in range(count)
    ]


def shards(count=20, shard_size=4):
    return [
        FrontendCacheShard(
            shard_index=index // shard_size,
            first_star=index,
            last_star=min(index + shard_size - 1, count - 1),
            sha256=hashlib.sha256(str(index).encode()).hexdigest(),
        )
        for index in range(0, count, shard_size)
    ]


def session():
    values = frames()
    return FixedLagSession(
        values,
        [min(index + 2, len(values) - 1) for index in range(len(values))],
        shards(),
        window_size_keyframes=8,
        anchor_band_keyframes=2,
        maximum_lookahead_keyframes=3,
        required_lookahead_keyframes=2,
        minimum_window_duration_seconds=0.5,
        checkpoint_interval_advances=4,
    )


def test_advance_requires_durable_head_before_release():
    value = session()
    for _ in range(10):
        value.ingest_next()
    proposal = value.propose_advance()
    assert value.finalized_count == 0
    assert value.released_cache_shards == set()
    with pytest.raises(FixedLagSessionError, match="token"):
        value.commit_advance(proposal["proposalUid"], "bad")
    result = value.commit_advance(proposal["proposalUid"], "a" * 64)
    assert result["stateAfter"]["finalizedCount"] == 1
    assert result["releasedCacheShardIndexes"] == []


def test_resident_state_reaches_a_bounded_plateau_and_releases_shards():
    value = session()
    peak = 0
    while value.next_ingest_ordinal < len(value.frames):
        value.ingest_next()
        while value.can_advance():
            proposal = value.propose_advance()
            value.commit_advance(
                proposal["proposalUid"],
                hashlib.sha256(proposal["proposalUid"].encode()).hexdigest(),
            )
        peak = max(peak, value.resident_count)
    assert peak <= (
        value.window_size_keyframes + value.required_lookahead_keyframes
    )
    assert value.finalized_count == 11
    assert value.resident_count == 9
    assert value.released_cache_shards == {0, 1}
    state = value.role_state()
    assert len(state["anchorOrdinals"]) == 2
    assert len(state["activeBodyOrdinals"]) == 6
    assert len(state["lookaheadOrdinals"]) == 1


def test_checkpoint_restore_reconstructs_identical_state():
    first = session()
    for _ in range(14):
        first.ingest_next()
        while first.can_advance():
            proposal = first.propose_advance()
            first.commit_advance(proposal["proposalUid"], "b" * 64)
    snapshot = first.snapshot()
    second = session()
    second.restore(snapshot)
    assert second.snapshot() == snapshot
    assert second.role_state() == first.role_state()


def test_group_commit_releases_only_after_one_accepted_batch_head():
    values = frames(24)
    value = FixedLagSession(
        values,
        [min(index + 2, len(values) - 1) for index in range(len(values))],
        shards(24),
        window_size_keyframes=8,
        anchor_band_keyframes=2,
        maximum_lookahead_keyframes=3,
        required_lookahead_keyframes=2,
        minimum_window_duration_seconds=0.5,
        checkpoint_interval_advances=4,
        durable_commit_batch_advances=4,
    )
    for _ in range(14):
        value.ingest_next()
    assert value.available_advance_count() == 4
    proposal = value.propose_batch()
    assert len(proposal["candidates"]) == 4
    assert value.finalized_count == 0
    assert value.released_cache_shards == set()
    result = value.commit_batch(proposal["proposalUid"], "b" * 64)
    assert result["logicalAdvanceCount"] == 4
    assert result["checkpointDue"] is True
    assert result["stateAfter"]["finalizedCount"] == 4
    assert result["releasedCacheShardIndexes"] == [0]


def test_cancelled_batch_allows_future_ingest_without_advancing_watermark():
    value = session()
    for _ in range(10):
        value.ingest_next()
    proposal = value.propose_advance()
    cancelled = value.cancel_batch(proposal["proposalUid"])
    assert cancelled["logicalAdvanceCount"] == 1
    assert value.finalized_count == 0
    assert value.last_accepted_index_head is None
    assert value.released_cache_shards == set()
    ingested = value.ingest_next()
    assert ingested["geometryOrdinal"] == 10
    replayed = value.propose_advance()
    assert replayed["proposalUid"] != proposal["proposalUid"]


def test_cancelled_batch_rejects_an_unrelated_token():
    value = session()
    for _ in range(10):
        value.ingest_next()
    proposal = value.propose_advance()
    with pytest.raises(FixedLagSessionError, match="identity"):
        value.cancel_batch("unrelated")
    value.commit_advance(proposal["proposalUid"], "c" * 64)


def test_checkpoint_from_other_frame_identity_is_rejected():
    first = session()
    first.ingest_next()
    snapshot = first.snapshot()
    changed_frames = frames()
    changed_frames[0] = GeometryFrame(
        geometry_ordinal=0,
        frame_uid="changed",
        keyframe_uid="keyframe-0",
        pts_value=0,
        time_base_numerator=1,
        time_base_denominator=10,
    )
    second = FixedLagSession(
        changed_frames,
        [min(index + 2, len(changed_frames) - 1) for index in range(20)],
        shards(),
        window_size_keyframes=8,
        anchor_band_keyframes=2,
        maximum_lookahead_keyframes=3,
        required_lookahead_keyframes=2,
        minimum_window_duration_seconds=0.5,
        checkpoint_interval_advances=4,
    )
    with pytest.raises(FixedLagSessionError, match="identity"):
        second.restore(snapshot)
