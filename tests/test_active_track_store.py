import hashlib

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
        "minimum_parallax_diagonals": 0.005,
        "maximum_match_error_pixels": 2.0,
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


def add_track(store, track, frames, *, error=0.5):
    observations = [observation(track, frame) for frame in frames]
    store.add_observations(observations)
    store.add_correspondences(
        TrackCorrespondence(
            observations[index].observation_uid,
            observations[index + 1].observation_uid,
            error,
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
                    0.5,
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


def test_match_error_and_parallax_rejections_are_reported():
    store = ActiveTrackStore(budget())
    add_track(store, 0, [0, 1, 2], error=3.0)
    values = [
        observation(1, 0, x=10, y=10),
        observation(1, 2, x=10.1, y=10.1),
    ]
    store.add_observations(values)
    store.add_correspondences(
        [TrackCorrespondence(values[0].observation_uid, values[1].observation_uid, 0.1)]
    )
    report = store.evaluate(
        active_first_ordinal=0,
        active_last_ordinal=3,
        freeze_through_ordinal=1,
    )
    assert report["rejectedReasonHistogram"]["match-error"] == 1
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
