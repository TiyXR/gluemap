import hashlib
import json

import pytest

from gluemap.estimators.active_track_store import (
    ActiveTrackStore,
    ActiveTrackStoreError,
    TrackBudget,
    TrackCorrespondence,
    TrackObservation,
)


def budget(**changes):
    values = {
        "window_size_keyframes": 4,
        "query_tracks_per_keyframe": 8,
        "maximum_candidate_observations_per_keyframe": 16,
        "active_track_budget_per_keyframe": 2,
        "minimum_constraints_per_keyframe": 1,
        "minimum_bridge_tracks": 1,
        "minimum_track_views": 2,
        "coverage_grid_columns": 2,
        "coverage_grid_rows": 2,
        "selection_time_bin_seconds": 0.2,
        "maximum_tracks_per_grid_cell": 2,
        "intra_image_merge_radius_pixels": 2.0,
        "minimum_visibility": 0.5,
        "minimum_confidence": 0.5,
        "minimum_parallax_diagonals": 0.005,
    }
    values.update(changes)
    return TrackBudget(**values)


def observation(track, frame, *, x=None, y=None, score=1.0):
    return TrackObservation(
        observation_uid=f"track-{track}-frame-{frame}",
        geometry_ordinal=frame,
        frame_uid=f"frame-{frame}",
        pts_value=frame,
        time_base_numerator=1,
        time_base_denominator=10,
        x=float(10 + frame * 3 + track * 40 if x is None else x),
        y=float(20 + track * 30 if y is None else y),
        image_width=200,
        image_height=100,
        score=score,
    )


def add_track(store, track, frames):
    observations = [observation(track, frame) for frame in frames]
    store.add_observations(observations)
    store.add_correspondences(
        TrackCorrespondence(
            observations[index].observation_uid,
            observations[index + 1].observation_uid,
        )
        for index in range(len(observations) - 1)
    )


def test_bridge_and_constraint_gate_passes_with_deterministic_identity():
    store = ActiveTrackStore(budget())
    add_track(store, 0, [0, 1, 2, 3])
    add_track(store, 1, [0, 1, 2, 3])
    report = store.evaluate(
        active_first_ordinal=0,
        active_last_ordinal=3,
        freeze_through_ordinal=1,
    )
    assert report["status"] == "passed"
    assert report["bridgeTrackCount"] == 2
    assert report["zeroConstraintOrdinals"] == []
    assert report["pixelArtifactCount"] == 0
    assert len(report["reportSha256"]) == 64
    assert "selectedTrackUids" not in report
    assert "bridgeTrackUids" not in report
    assert len(report["selectedTrackUidsSha256"]) == 64
    assert len(report["bridgeTrackUidsSha256"]) == 64


def test_release_group_batch_matches_individual_gate_results():
    store = ActiveTrackStore(budget())
    add_track(store, 0, [0, 1, 2, 3])
    add_track(store, 1, [0, 1, 2, 3])
    intervals = [(0, 3, 1), (0, 3, 2)]
    expected = [
        store.evaluate(
            active_first_ordinal=active_first,
            active_last_ordinal=active_last,
            freeze_through_ordinal=freeze_through,
        )
        for active_first, active_last, freeze_through in intervals
    ]
    assert store.evaluate_batch(intervals) == expected


def test_parallax_backend_and_microbatch_are_reported():
    store = ActiveTrackStore(
        budget(
            parallax_backend_policy="cpu",
            parallax_microbatch_components=2,
        )
    )
    add_track(store, 0, [0, 1, 2, 3])
    report = store.evaluate(
        active_first_ordinal=0,
        active_last_ordinal=3,
        freeze_through_ordinal=1,
    )
    assert report["parallaxBackendPolicy"] == "cpu"
    assert report["parallaxBackend"] == "cpu"
    assert report["gateMetricsBackend"] == "cpu"
    assert report["parallaxMicrobatchComponents"] == 2


def test_batch_metrics_count_one_constraint_per_track_and_frame():
    store = ActiveTrackStore(budget(parallax_backend_policy="cpu"))
    first = observation(0, 0, x=10, y=10)
    duplicate_view = TrackObservation(
        **{
            **observation(0, 0, x=12, y=10).__dict__,
            "observation_uid": "track-0-frame-0-duplicate",
        }
    )
    last = observation(0, 3, x=30, y=10)
    store.add_observations([first, duplicate_view, last])
    store.add_correspondences(
        [
            TrackCorrespondence(first.observation_uid, duplicate_view.observation_uid),
            TrackCorrespondence(duplicate_view.observation_uid, last.observation_uid),
        ]
    )
    report = store.evaluate_batch([(0, 3, 1)])[0]
    assert report["constraintsPerFrame"] == {"0": 1, "1": 0, "2": 0, "3": 1}
    assert report["selectedTrackCount"] == 1


def test_gate_report_is_canonical_across_decimal_key_width_boundary():
    store = ActiveTrackStore(budget(window_size_keyframes=4))
    add_track(store, 0, [8, 9, 10, 11])
    report = store.evaluate(
        active_first_ordinal=8,
        active_last_ordinal=11,
        freeze_through_ordinal=9,
    )
    encoded = json.dumps(
        report, sort_keys=True, separators=(",", ":")
    )
    round_trip = json.dumps(
        json.loads(encoded), sort_keys=True, separators=(",", ":")
    )
    assert list(report["constraintsPerFrame"]) == ["8", "9", "10", "11"]
    assert encoded == round_trip


def test_zero_constraint_and_bridge_failure_stop_instead_of_filling_pose():
    store = ActiveTrackStore(budget())
    add_track(store, 0, [0, 1])
    report = store.evaluate(
        active_first_ordinal=0,
        active_last_ordinal=3,
        freeze_through_ordinal=1,
    )
    assert report["status"] == "failed"
    assert "ZERO_CONSTRAINT" in report["reasonCodes"]
    assert "BRIDGE_TRACKS_BELOW_MINIMUM" in report["reasonCodes"]
    assert report["zeroConstraintOrdinals"] == [2, 3]


def test_time_grid_round_robin_prevents_one_bucket_from_consuming_budget():
    store = ActiveTrackStore(
        budget(
            active_track_budget_per_keyframe=1,
            maximum_tracks_per_grid_cell=1,
        )
    )
    for track, x in enumerate((10, 12, 150, 152)):
        observations = [
            observation(track, 0, x=x, y=20),
            observation(track, 3, x=x + 10, y=20),
        ]
        store.add_observations(observations)
        store.add_correspondences(
            [
                TrackCorrespondence(
                    observations[0].observation_uid,
                    observations[1].observation_uid,
                )
            ]
        )
    report = store.evaluate(
        active_first_ordinal=0,
        active_last_ordinal=3,
        freeze_through_ordinal=1,
    )
    assert report["selectedTrackCount"] == 2
    assert len(report["timeGridBucketCounts"]) == 2
    assert report["rejectedReasonHistogram"]["active-budget-or-cell-cap"] == 2


def test_low_parallax_rejection_is_reported():
    store = ActiveTrackStore(budget())
    values = [
        observation(1, 0, x=10, y=10),
        observation(1, 2, x=10.1, y=10.1),
    ]
    store.add_observations(values)
    store.add_correspondences(
        [TrackCorrespondence(values[0].observation_uid, values[1].observation_uid)]
    )
    report = store.evaluate(
        active_first_ordinal=0,
        active_last_ordinal=3,
        freeze_through_ordinal=1,
    )
    assert report["rejectedReasonHistogram"]["insufficient-parallax"] == 1


def test_release_requires_accepted_journal_head_and_rebuilds_components():
    store = ActiveTrackStore(budget())
    add_track(store, 0, [0, 1, 2, 3])
    proposal = store.propose_release(1)
    assert store.observation_count == 4
    with pytest.raises(ActiveTrackStoreError, match="SHA-256"):
        store.commit_release(proposal["proposalUid"], "bad")
    head = hashlib.sha256(b"accepted-journal").hexdigest()
    result = store.commit_release(proposal["proposalUid"], head)
    assert result["releasedObservationCount"] == 2
    assert result["remainingObservationCount"] == 2
    assert result["remainingEdgeCount"] == 1
    assert store.last_accepted_journal_head == head
    report = store.evaluate(
        active_first_ordinal=2,
        active_last_ordinal=3,
        freeze_through_ordinal=2,
    )
    assert report["candidateTrackCount"] == 1


def test_per_frame_observation_bound_and_identity_reuse_are_rejected():
    store = ActiveTrackStore(
        budget(
            query_tracks_per_keyframe=2,
            maximum_candidate_observations_per_keyframe=2,
        )
    )
    store.add_observations([observation(0, 0), observation(1, 0)])
    with pytest.raises(ActiveTrackStoreError, match="bound"):
        store.add_observations([observation(2, 0)])
    changed = TrackObservation(**{**observation(0, 0).__dict__, "x": 99.0})
    with pytest.raises(ActiveTrackStoreError, match="reused"):
        store.add_observations([changed])


def test_overlapping_stars_merge_nearby_observations_without_new_files():
    store = ActiveTrackStore(budget(intra_image_merge_radius_pixels=3.0))
    first = observation(0, 0, x=50.0, y=20.0)
    nearby = TrackObservation(
        **{
            **observation(1, 0, x=51.5, y=21.0).__dict__,
            "observation_uid": "second-star-candidate",
        }
    )
    first_uid = store.intern_observation(first)
    second_uid = store.intern_observation(nearby)
    assert second_uid == first_uid
    assert store.observation_count == 1
