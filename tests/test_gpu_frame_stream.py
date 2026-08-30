"""Tests for zero-copy routed NVDEC frame ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from gluemap.video.gpu_frame_stream import (
    FrameRoute,
    GpuFrameStreamError,
    ThreadedNvdecRouteStream,
    load_frame_routes,
)


class FakeFrame:
    def __init__(self, pts: int, value: int) -> None:
        self.pts = pts
        self.tensor = torch.full((3, 4, 6), value, dtype=torch.uint8)

    def getPTS(self) -> int:
        return self.pts

    def __dlpack__(self, stream=None):
        return self.tensor.__dlpack__(stream=stream)

    def __dlpack_device__(self):
        return self.tensor.__dlpack_device__()


@dataclass
class FakeMetadata:
    width: int = 6
    height: int = 4
    num_frames: int = 6


class FakeDecoder:
    def __init__(self, frames: list[FakeFrame], **kwargs) -> None:
        self.frames = frames
        self.offset = 0
        self.kwargs = kwargs

    def get_stream_metadata(self) -> FakeMetadata:
        return FakeMetadata()

    def get_batch_frames(self, batch_size: int) -> list[FakeFrame]:
        batch = self.frames[self.offset : self.offset + batch_size]
        self.offset += len(batch)
        return batch


def routes() -> tuple[FrameRoute, ...]:
    return tuple(
        FrameRoute(
            observation_ordinal=index,
            geometry_ordinal=index,
            observation_uid=f"observation-{index}",
            keyframe_uid=f"keyframe-{index}",
            frame_uid=f"frame-{presentation}",
            presentation_ordinal=presentation,
            pts_value=presentation * 1001,
            time_base_numerator=1,
            time_base_denominator=30000,
            frontend_input=True,
        )
        for index, presentation in enumerate((0, 2, 5))
    )


def test_threaded_stream_decodes_every_frame_but_yields_only_routes() -> None:
    frames = [FakeFrame(i * 1001, i) for i in range(6)]
    decoder = FakeDecoder(frames)
    constructor: dict = {}

    def decoder_factory(**kwargs):
        constructor.update(kwargs)
        return decoder

    stream = ThreadedNvdecRouteStream(
        "fixture.mp4",
        routes(),
        maximum_in_flight_frames=4,
        decode_batch_size=2,
        require_cuda=False,
        decoder_factory=decoder_factory,
        output_color_type="RGBP",
    )

    selected = list(stream)

    assert [value.route.presentation_ordinal for value in selected] == [0, 2, 5]
    assert [int(value.tensor[0, 0, 0]) for value in selected] == [0, 2, 5]
    assert all(
        value.tensor.data_ptr() == value.decoder_frame_owner.tensor.data_ptr()
        for value in selected
    )
    assert stream.decoded_frame_count == 6
    assert stream.yielded_route_count == 3
    assert constructor == {
        "enc_file_path": "fixture.mp4",
        "buffer_size": 4,
        "gpu_id": 0,
        "use_device_memory": True,
        "output_color_type": "RGBP",
    }


def test_threaded_stream_rejects_pts_drift() -> None:
    frames = [FakeFrame(i * 1001, i) for i in range(6)]
    frames[2].pts += 1
    stream = ThreadedNvdecRouteStream(
        "fixture.mp4",
        routes(),
        maximum_in_flight_frames=4,
        decode_batch_size=2,
        require_cuda=False,
        decoder_factory=lambda **kwargs: FakeDecoder(frames),
        output_color_type="RGBP",
    )
    with pytest.raises(GpuFrameStreamError, match="decoded PTS differs"):
        list(stream)


def test_route_manifest_must_be_strict_and_role_consistent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "routes.jsonl"
    payloads = []
    for route in routes():
        payloads.append(
            {
                "contractId": "jarailsense.gluemap-frame-route/v1",
                "routingIdentitySha256": "a" * 64,
                "observationOrdinal": route.observation_ordinal,
                "geometryOrdinal": route.geometry_ordinal,
                "observationUid": route.observation_uid,
                "keyframeUid": route.keyframe_uid,
                "frameUid": route.frame_uid,
                "presentationOrdinal": route.presentation_ordinal,
                "ptsValue": route.pts_value,
                "timeBase": {"numerator": 1, "denominator": 30000},
                "frontendInput": True,
            }
        )
    path.write_text("".join(json.dumps(value) + "\n" for value in payloads))
    assert load_frame_routes(path) == routes()

    payloads[1]["frontendInput"] = False
    path.write_text("".join(json.dumps(value) + "\n" for value in payloads))
    with pytest.raises(GpuFrameStreamError, match="role"):
        load_frame_routes(path)
