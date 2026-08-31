"""Tests for bounded center-ordered GPU star assembly."""

from __future__ import annotations

import argparse

import numpy as np
import pytest
import torch

from gluemap.controllers.base_inference import BaseInferencePipeline
from gluemap.datasets.streaming_video_star import StreamingVideoStarDataset
from gluemap.video.gpu_frame_stream import (
    DecodedRouteFrame,
    FrameRoute,
    GpuFrameStreamError,
)


def routes() -> tuple[FrameRoute, ...]:
    return tuple(
        FrameRoute(
            observation_ordinal=index,
            geometry_ordinal=index,
            observation_uid=f"observation-{index}",
            keyframe_uid=f"keyframe-{index}",
            frame_uid=f"frame-{index}",
            presentation_ordinal=index,
            pts_value=index * 1001,
            time_base_numerator=1,
            time_base_denominator=30000,
            frontend_input=True,
        )
        for index in range(5)
    )


def stream() -> iter:
    for index, route in enumerate(routes()):
        tensor = torch.full((3, 24, 32), index, dtype=torch.uint8)
        yield DecodedRouteFrame(route, tensor, tensor)


def query_points(image: torch.Tensor) -> torch.Tensor:
    return torch.zeros((1, 8, 2), device=image.device)


def test_star_stream_waits_for_lookahead_and_releases_finished_frames() -> None:
    pairs = np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], dtype=np.int64)
    dataset = StreamingVideoStarDataset(
        "fixture.mp4",
        routes(),
        pairs,
        maximum_decoder_frames=4,
        maximum_resident_frames=3,
        decode_batch_size=2,
        stream_factory=stream,
        query_point_provider=query_points,
        require_cuda=False,
    )

    items = list(dataset)

    assert [item["star_indexes"] for item in items] == [0, 1, 2, 3, 4]
    assert [item["indexes"].tolist() for item in items] == [
        [0, 1],
        [1, 0, 2],
        [2, 1, 3],
        [3, 2, 4],
        [4, 3],
    ]
    assert items[1]["frame_uids"] == ["frame-1", "frame-0", "frame-2"]
    assert dataset.peak_resident_frames <= 3
    assert dataset.released_frame_count == 5
    assert dataset.emitted_star_count == 5
    assert all(item["images"].device.type == "cpu" for item in items)


class DummyPipeline(BaseInferencePipeline):
    def _load_models(self):
        return {}

    def _create_batch_inference(self, models):
        return None

    def _run_batch_step(self, batch_inference, batch):
        return {}, {}

    def _pack_local_outputs(self, all_outputs, all_indices):
        return {}

    def _merge_gathered_outputs(self, data_list, index_mapping, dataset_size):
        return {}

    def _postprocess_global_outputs(self, global_outputs, dataset):
        return global_outputs


def test_gpu_stream_dataloader_forces_main_process_without_pin_memory() -> None:
    dataset = StreamingVideoStarDataset(
        "fixture.mp4",
        routes(),
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]),
        maximum_decoder_frames=4,
        maximum_resident_frames=3,
        stream_factory=stream,
        query_point_provider=query_points,
        require_cuda=False,
    )
    args = argparse.Namespace(
        distributed=False,
        batch_size=1,
        num_workers=8,
    )
    pipeline = DummyPipeline(args, 1, 0, "unused.pth")
    loader = pipeline._make_dataloader(dataset)
    assert loader.num_workers == 0
    assert loader.pin_memory is False

    args.distributed = True
    with pytest.raises(ValueError, match="DistributedSampler"):
        pipeline._make_dataloader(dataset)


def test_streaming_dataset_requires_locked_query_extractor_checkpoint() -> None:
    dataset = StreamingVideoStarDataset(
        "fixture.mp4",
        routes(),
        np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]]),
        maximum_decoder_frames=4,
        maximum_resident_frames=3,
        stream_factory=stream,
        require_cuda=False,
    )
    with pytest.raises(GpuFrameStreamError, match="locked ALIKED"):
        next(iter(dataset))
