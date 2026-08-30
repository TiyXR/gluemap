"""Decode routed video frames directly into CUDA tensors through DLPack."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch


class GpuFrameStreamError(RuntimeError):
    """The NVDEC stream cannot preserve the frozen frame-route contract."""


@dataclass(frozen=True)
class FrameRoute:
    observation_ordinal: int
    geometry_ordinal: int | None
    observation_uid: str
    keyframe_uid: str | None
    frame_uid: str
    presentation_ordinal: int
    pts_value: int
    time_base_numerator: int
    time_base_denominator: int
    frontend_input: bool


@dataclass(frozen=True)
class DecodedRouteFrame:
    """One selected frame whose tensor shares the decoder's CUDA allocation."""

    route: FrameRoute
    tensor: torch.Tensor
    decoder_frame_owner: Any


class _StreamMetadata(Protocol):
    width: int
    height: int
    num_frames: int


class _ThreadedDecoder(Protocol):
    def get_stream_metadata(self) -> _StreamMetadata: ...

    def get_batch_frames(self, batch_size: int) -> list[Any]: ...


def _required_integer(value: Any, scope: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise GpuFrameStreamError(f"{scope} is invalid")
    return value


def load_frame_routes(path: str | Path) -> tuple[FrameRoute, ...]:
    """Load the A07 route manifest and reject any identity/order ambiguity."""
    routes: list[FrameRoute] = []
    previous_presentation = -1
    path = Path(path)
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise GpuFrameStreamError(
                    f"frame route line {line_number} contains invalid JSON"
                ) from error
            if not isinstance(value, dict):
                raise GpuFrameStreamError(
                    f"frame route line {line_number} is not an object"
                )
            observation_ordinal = _required_integer(
                value.get("observationOrdinal"), "observation ordinal"
            )
            presentation_ordinal = _required_integer(
                value.get("presentationOrdinal"), "presentation ordinal"
            )
            pts_value = _required_integer(value.get("ptsValue"), "PTS")
            geometry_ordinal = value.get("geometryOrdinal")
            if geometry_ordinal is not None:
                geometry_ordinal = _required_integer(
                    geometry_ordinal, "geometry ordinal"
                )
            time_base = value.get("timeBase")
            if not isinstance(time_base, dict):
                raise GpuFrameStreamError("frame route time base is invalid")
            numerator = _required_integer(
                time_base.get("numerator"), "time base numerator", 1
            )
            denominator = _required_integer(
                time_base.get("denominator"), "time base denominator", 1
            )
            frontend_input = value.get("frontendInput")
            if (
                value.get("contractId") != "jarailsense.gluemap-frame-route/v1"
                or observation_ordinal != len(routes)
                or presentation_ordinal <= previous_presentation
                or not isinstance(value.get("observationUid"), str)
                or not isinstance(value.get("frameUid"), str)
                or not isinstance(frontend_input, bool)
                or frontend_input != (geometry_ordinal is not None)
                or (
                    frontend_input
                    and not isinstance(value.get("keyframeUid"), str)
                )
                or (not frontend_input and value.get("keyframeUid") is not None)
            ):
                raise GpuFrameStreamError(
                    "frame route identity/order/role differs"
                )
            routes.append(
                FrameRoute(
                    observation_ordinal=observation_ordinal,
                    geometry_ordinal=geometry_ordinal,
                    observation_uid=value["observationUid"],
                    keyframe_uid=value.get("keyframeUid"),
                    frame_uid=value["frameUid"],
                    presentation_ordinal=presentation_ordinal,
                    pts_value=pts_value,
                    time_base_numerator=numerator,
                    time_base_denominator=denominator,
                    frontend_input=frontend_input,
                )
            )
            previous_presentation = presentation_ordinal
    if not routes:
        raise GpuFrameStreamError("frame route manifest is empty")
    return tuple(routes)


class ThreadedNvdecRouteStream:
    """Background NVDEC prefetch with zero-copy DLPack handoff to Torch.

    Every presentation frame is decoded in order. Only frozen route frames are
    yielded, and their PTS must match the A07 route exactly. No host image,
    JPEG, random seek, or image directory is created.
    """

    def __init__(
        self,
        video_path: str | Path,
        routes: tuple[FrameRoute, ...],
        *,
        maximum_in_flight_frames: int,
        decode_batch_size: int = 8,
        gpu_id: int = 0,
        require_cuda: bool = True,
        decoder_factory: Callable[..., _ThreadedDecoder] | None = None,
        output_color_type: Any | None = None,
    ) -> None:
        if not routes:
            raise GpuFrameStreamError("at least one frame route is required")
        if maximum_in_flight_frames < 1 or maximum_in_flight_frames > 256:
            raise GpuFrameStreamError(
                "maximum in-flight frames is outside 1..256"
            )
        if (
            decode_batch_size < 1
            or decode_batch_size > maximum_in_flight_frames
        ):
            raise GpuFrameStreamError(
                "decode batch size exceeds the in-flight bound"
            )
        self.video_path = Path(video_path)
        self.routes = routes
        self.maximum_in_flight_frames = maximum_in_flight_frames
        self.decode_batch_size = decode_batch_size
        self.gpu_id = gpu_id
        self.require_cuda = require_cuda
        self.decoded_frame_count = 0
        self.yielded_route_count = 0
        self.last_decoded_pts: int | None = None

        if decoder_factory is None:
            try:
                import PyNvVideoCodec as nvc
            except ImportError as error:
                raise GpuFrameStreamError(
                    "PyNvVideoCodec is required for the GPU-resident "
                    "frame pipeline"
                ) from error
            decoder_factory = nvc.ThreadedDecoder
            output_color_type = nvc.OutputColorType.RGBP
        if output_color_type is None:
            raise GpuFrameStreamError("RGBP output color type is required")
        self._decoder = decoder_factory(
            enc_file_path=str(self.video_path),
            buffer_size=maximum_in_flight_frames,
            gpu_id=gpu_id,
            use_device_memory=True,
            output_color_type=output_color_type,
        )
        self.metadata = self._decoder.get_stream_metadata()
        if (
            self.metadata.width <= 0
            or self.metadata.height <= 0
            or self.metadata.num_frames <= routes[-1].presentation_ordinal
        ):
            raise GpuFrameStreamError(
                "decoder metadata cannot cover the frozen routes"
            )

    def __iter__(self) -> Iterator[DecodedRouteFrame]:
        route_index = 0
        presentation_ordinal = 0
        previous_pts: int | None = None
        while route_index < len(self.routes):
            decoder_frames = self._decoder.get_batch_frames(
                self.decode_batch_size
            )
            if not decoder_frames:
                break
            if len(decoder_frames) > self.maximum_in_flight_frames:
                raise GpuFrameStreamError(
                    "decoder exceeded the in-flight frame bound"
                )
            for owner in decoder_frames:
                pts = int(owner.getPTS())
                if previous_pts is not None and pts <= previous_pts:
                    raise GpuFrameStreamError(
                        "decoder PTS is not strictly increasing"
                    )
                previous_pts = pts
                self.last_decoded_pts = pts
                route = self.routes[route_index]
                if presentation_ordinal == route.presentation_ordinal:
                    if pts != route.pts_value:
                        raise GpuFrameStreamError(
                            "decoded PTS differs from frame route"
                        )
                    tensor = torch.from_dlpack(owner)
                    if (
                        tensor.ndim != 3
                        or tensor.shape[0] != 3
                        or tensor.shape[1] != self.metadata.height
                        or tensor.shape[2] != self.metadata.width
                        or tensor.dtype != torch.uint8
                    ):
                        raise GpuFrameStreamError(
                            "decoded RGBP tensor shape/type differs"
                        )
                    if self.require_cuda and (
                        not tensor.is_cuda or tensor.device.index != self.gpu_id
                    ):
                        raise GpuFrameStreamError(
                            "decoded frame is not on requested CUDA GPU"
                        )
                    yield DecodedRouteFrame(route, tensor, owner)
                    route_index += 1
                    self.yielded_route_count += 1
                presentation_ordinal += 1
                self.decoded_frame_count += 1
                if route_index == len(self.routes):
                    break
        if route_index != len(self.routes):
            raise GpuFrameStreamError(
                "decoder ended before all frame routes were consumed"
            )
