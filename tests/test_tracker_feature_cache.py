from __future__ import annotations

import torch

from gluemap.estimators.track_inference import (
    TrackInference,
    TrackerCoarseFeatureCache,
)


class FakeTracker(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.extracted_frame_count = 0

    def process_images_to_fmaps(self, images):
        self.extracted_frame_count += images.shape[0]
        values = images.mean(dim=(1, 2, 3))
        return values[:, None, None, None].repeat(1, 4, 3, 3)


def images(values):
    return torch.stack(
        [torch.full((3, 32, 32), value) for value in values]
    ).unsqueeze(0)


def test_overlapping_stars_extract_each_coarse_feature_once():
    model = FakeTracker()
    cache = TrackerCoarseFeatureCache(model, 4, "cuda:0")

    first = cache.features(model, images([1.0, 2.0, 3.0]), ["a", "b", "c"])
    second = cache.features(model, images([2.0, 3.0, 4.0]), ["b", "c", "d"])

    assert first[:, :, 0, 0, 0].tolist() == [[1.0, 2.0, 3.0]]
    assert second[:, :, 0, 0, 0].tolist() == [[2.0, 3.0, 4.0]]
    report = cache.report()
    assert model.extracted_frame_count == 4
    assert report["requestedFrameCount"] == 6
    assert report["cacheHitCount"] == 2
    assert report["cacheMissCount"] == 4
    assert report["extractedFrameCount"] == 4
    assert report["hitRate"] == 2 / 6
    assert report["evictionCount"] == 0
    assert report["peakResidentFrameCount"] == 4


def test_tracker_feature_cache_is_bounded():
    model = FakeTracker()
    cache = TrackerCoarseFeatureCache(model, 2, "cuda:0")
    cache.features(model, images([1.0, 2.0]), ["a", "b"])
    cache.features(model, images([2.0, 3.0]), ["b", "c"])
    report = cache.report()
    assert report["residentFrameCount"] == 2
    assert report["evictionCount"] == 1


def test_collated_frame_uids_remain_stable_tracker_cache_keys():
    assert TrackInference._frame_uids(
        {"frame_uids": [("frame-a",), ("frame-b",)]}, 2
    ) == ["frame-a", "frame-b"]
    assert TrackInference._frame_uids(
        {"indexes": torch.tensor([[4, 5]])}, 2
    ) == ["geometry-4", "geometry-5"]
