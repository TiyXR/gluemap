from __future__ import annotations

from dataclasses import dataclass

import torch

from gluemap.ff_inference.pi3_inference import (
    Pi3EncoderTokenCache,
    Pi3LocalInference,
)


@dataclass(frozen=True)
class FakeEncodedFrames:
    tokens: torch.Tensor
    image_height: int
    image_width: int


class FakePi3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoded_frame_count = 0

    def encode_frames(self, images):
        self.encoded_frame_count += images.shape[1]
        values = images.mean(dim=(2, 3, 4), keepdim=False)
        tokens = values[:, :, None, None].repeat(1, 1, 4, 2)
        return FakeEncodedFrames(tokens, images.shape[-2], images.shape[-1])


def images(values):
    return torch.stack(
        [torch.full((3, 28, 28), value) for value in values]
    ).unsqueeze(0)


def test_overlapping_context_encodes_each_unique_frame_once():
    model = FakePi3()
    cache = Pi3EncoderTokenCache(
        model,
        capacity_frames=4,
        device="cuda:0",
        dtype=torch.bfloat16,
    )

    first = cache.encode_context(
        model, images([1.0, 2.0, 3.0]), ["a", "b", "c"]
    )
    second = cache.encode_context(
        model, images([2.0, 3.0, 4.0]), ["b", "c", "d"]
    )

    assert first.tokens[:, :, 0, 0].tolist() == [[1.0, 2.0, 3.0]]
    assert second.tokens[:, :, 0, 0].tolist() == [[2.0, 3.0, 4.0]]
    report = cache.report()
    assert model.encoded_frame_count == 4
    assert report["requestedFrameCount"] == 6
    assert report["encodedFrameCount"] == 4
    assert report["cacheHitCount"] == 2
    assert report["cacheMissCount"] == 4
    assert report["hitRate"] == 2 / 6
    assert report["evictionCount"] == 0
    assert report["peakResidentFrameCount"] == 4


def test_cache_is_bounded_and_reports_eviction():
    model = FakePi3()
    cache = Pi3EncoderTokenCache(
        model,
        capacity_frames=2,
        device="cuda:0",
        dtype=torch.float16,
    )
    cache.encode_context(model, images([1.0, 2.0]), ["a", "b"])
    cache.encode_context(model, images([2.0, 3.0]), ["b", "c"])
    report = cache.report()
    assert report["residentFrameCount"] == 2
    assert report["peakResidentFrameCount"] == 2
    assert report["evictionCount"] == 1


def test_context_microbatch_deduplicates_overlapping_encoder_inputs():
    model = FakePi3()
    cache = Pi3EncoderTokenCache(
        model,
        capacity_frames=4,
        device="cuda:0",
        dtype=torch.bfloat16,
    )
    batched_images = torch.cat(
        [images([1.0, 2.0, 3.0]), images([2.0, 3.0, 4.0])], dim=0
    )

    encoded = cache.encode_context(
        model,
        batched_images,
        [["a", "b", "c"], ["b", "c", "d"]],
    )

    assert encoded.tokens[:, :, 0, 0].tolist() == [
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ]
    report = cache.report()
    assert model.encoded_frame_count == 4
    assert report["requestedFrameCount"] == 6
    assert report["cacheMissCount"] == 4
    assert report["microbatchReuseCount"] == 2
    assert report["hitRate"] == 2 / 6
    assert report["residentHitRate"] == 0.0


def test_collated_frame_uids_remain_stable_cache_keys():
    assert Pi3LocalInference._frame_uids(
        {"frame_uids": [("frame-a",), ("frame-b",)]}, 2
    ) == ["frame-a", "frame-b"]
    assert Pi3LocalInference._frame_uids(
        {"indexes": torch.tensor([[4, 5]])}, 2
    ) == ["geometry-4", "geometry-5"]
    assert Pi3LocalInference._frame_uids(
        {
            "images": torch.zeros((2, 2, 3, 4, 4)),
            "frame_uids": [("frame-a", "frame-b"), ("frame-c", "frame-d")],
        },
        2,
    ) == [["frame-a", "frame-c"], ["frame-b", "frame-d"]]
