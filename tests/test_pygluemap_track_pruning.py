"""Unit tests for the pygluemap track-pruning bindings.

These cover ``compute_tracks_to_delete`` and
``compute_virtual_tracks_to_delete``
in [gluemap/pybind/track_selection.cc](gluemap/pybind/track_selection.cc), which
classify tracks as SIFT vs non-SIFT and prune non-SIFT tracks whose image-pair
coverage is already above a threshold. SIFT-vs-non-SIFT classification is
implicit: an observation is SIFT iff its ``pt2d_idx`` is below
``sift_count[image_id]``.

The pair-count dictionary uses a packed canonical key ``(min<<32) | max`` over
the two image IDs.
"""

import numpy as np
import pygluemap

import hashlib


def test_connected_components_returns_minimum_integer_root():
    labels = pygluemap.compute_connected_components(
        7,
        np.asarray([0, 1, 4], dtype=np.int64),
        np.asarray([1, 2, 5], dtype=np.int64),
    )
    assert labels.tolist() == [0, 0, 0, 3, 4, 4, 6]


def test_batch_spatial_intern_preserves_frame_and_input_order():
    representatives = pygluemap.batch_spatial_intern(
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([10.0, 10.0], dtype=np.float64),
        np.asarray([10.0, 10.0], dtype=np.float64),
        ["existing-frame-0", "existing-frame-1"],
        np.asarray([0, 2, 2], dtype=np.int64),
        np.asarray([11.0, 50.0, 51.0], dtype=np.float64),
        np.asarray([10.0, 20.0, 20.0], dtype=np.float64),
        ["incoming-0", "incoming-1", "incoming-2"],
        2.0,
    )
    assert representatives.tolist() == [0, 3, 3]


def test_batch_observation_uids_match_the_python_contract():
    values = pygluemap.batch_observation_uids(
        "a" * 64,
        np.asarray([0, 12], dtype=np.int64),
        np.asarray([1, 3], dtype=np.int64),
        ["frame-a", "frame-b"],
    )
    expected = []
    for track, view, frame in ((0, 1, "frame-a"), (12, 3, "frame-b")):
        payload = (
            "jarailsense.gluemap-track-observation/v1\0"
            f"{'a' * 64}\0{track}\0{view}\0{frame}"
        ).encode("utf-8")
        expected.append(hashlib.sha256(payload).hexdigest())
    assert values == expected


def _pair_key(a, b):
    lo, hi = (a, b) if a < b else (b, a)
    return (np.uint64(lo) << np.uint64(32)) | np.uint64(hi)


def _csr(tracks):
    """Pack a list-of-lists track structure into the CSR layout the binding
    expects.

    Each track is ``[(image_id, pt2d_idx), ...]``. Returns
    ``(point3d_ids, image_ids, pt2d_idxs, lengths)``.
    """
    point3d_ids = np.arange(1, len(tracks) + 1, dtype=np.int64)
    image_ids = np.array(
        [obs[0] for tr in tracks for obs in tr], dtype=np.int64
    )
    pt2d_idxs = np.array(
        [obs[1] for tr in tracks for obs in tr], dtype=np.int64
    )
    lengths = np.array([len(tr) for tr in tracks], dtype=np.int32)
    return point3d_ids, image_ids, pt2d_idxs, lengths


def test_is_cuda_available_returns_bool():
    assert isinstance(pygluemap.is_cuda_available(), bool)


def test_compute_tracks_to_delete_preserves_sift():
    # All tracks have pt2d_idx < sift_count[image_id] -> all classified as SIFT
    # and never deleted.
    sift_count = {1: 100, 2: 100, 3: 100}
    tracks = [
        [(1, 0), (2, 1), (3, 2)],
        [(1, 5), (2, 5)],
        [(2, 9), (3, 9)],
    ]
    point3d_ids, image_ids, pt2d_idxs, lengths = _csr(tracks)
    ids_to_delete, pair_count = pygluemap.compute_tracks_to_delete(
        point3d_ids, image_ids, pt2d_idxs, lengths, sift_count, 1
    )
    assert ids_to_delete.size == 0
    # Pair (1,2) is covered by tracks 0 and 1 -> count = 2.
    assert pair_count[int(_pair_key(1, 2))] == 2
    # Pair (2,3) is covered by tracks 0 and 2 -> count = 2.
    assert pair_count[int(_pair_key(2, 3))] == 2


def test_compute_tracks_to_delete_threshold_boundary():
    # 3 SIFT tracks all sharing pair (1,2): seed pair_count[(1,2)] = 3.
    # 1 non-SIFT track on (1,2): kept iff 3 <= min_num_support_abs.
    sift_count = {1: 100, 2: 100}
    tracks = [
        [(1, 0), (2, 0)],  # SIFT
        [(1, 1), (2, 1)],  # SIFT
        [(1, 2), (2, 2)],  # SIFT
        [(1, 105), (2, 105)],  # non-SIFT (pt2d_idx >= 100)
    ]
    point3d_ids, image_ids, pt2d_idxs, lengths = _csr(tracks)
    non_sift_id = int(point3d_ids[3])

    # threshold = 2 -> 3 <= 2 is False -> non-SIFT track is deleted.
    ids_to_delete, _ = pygluemap.compute_tracks_to_delete(
        point3d_ids.copy(),
        image_ids.copy(),
        pt2d_idxs.copy(),
        lengths.copy(),
        sift_count,
        2,
    )
    assert non_sift_id in ids_to_delete.tolist()

    # threshold = 3 -> 3 <= 3 is True -> non-SIFT track is kept.
    ids_to_delete, _ = pygluemap.compute_tracks_to_delete(
        point3d_ids.copy(),
        image_ids.copy(),
        pt2d_idxs.copy(),
        lengths.copy(),
        sift_count,
        3,
    )
    assert non_sift_id not in ids_to_delete.tolist()


def test_compute_tracks_to_delete_empty_input():
    point3d_ids = np.array([], dtype=np.int64)
    image_ids = np.array([], dtype=np.int64)
    pt2d_idxs = np.array([], dtype=np.int64)
    lengths = np.array([], dtype=np.int32)
    ids_to_delete, pair_count = pygluemap.compute_tracks_to_delete(
        point3d_ids, image_ids, pt2d_idxs, lengths, {}, 512
    )
    assert ids_to_delete.size == 0
    assert len(pair_count) == 0


def test_compute_virtual_tracks_to_delete_all_above_threshold_removed():
    # Pre-existing pair_count well above the threshold for every pair the
    # virtual track touches -> track is deleted.
    pair_count = {int(_pair_key(1, 2)): 100}
    tracks = [[(1, 0), (2, 0)]]
    point3d_ids, image_ids, pt2d_idxs, lengths = _csr(tracks)
    ids_to_delete, _ = pygluemap.compute_virtual_tracks_to_delete(
        point3d_ids, image_ids, pt2d_idxs, lengths, pair_count, 50
    )
    assert ids_to_delete.tolist() == [int(point3d_ids[0])]


def test_compute_virtual_tracks_to_delete_partial_coverage_kept():
    # One pair under threshold -> track is kept and its pairs incremented.
    pair_count = {
        int(_pair_key(1, 2)): 100,  # well-covered
        int(_pair_key(2, 3)): 10,  # under threshold (≤ 50)
        int(_pair_key(1, 3)): 100,
    }
    tracks = [[(1, 0), (2, 0), (3, 0)]]
    point3d_ids, image_ids, pt2d_idxs, lengths = _csr(tracks)
    ids_to_delete, updated = pygluemap.compute_virtual_tracks_to_delete(
        point3d_ids, image_ids, pt2d_idxs, lengths, pair_count, 50
    )
    assert ids_to_delete.size == 0
    # Each pair the track touches gains 1.
    assert updated[int(_pair_key(1, 2))] == 101
    assert updated[int(_pair_key(2, 3))] == 11
    assert updated[int(_pair_key(1, 3))] == 101
