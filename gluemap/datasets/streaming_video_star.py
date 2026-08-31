"""Bounded GPU-resident star batches built from one sequential video decode."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lightglue import ALIKED
from torch.utils.data import IterableDataset

from gluemap.estimators.feature_extraction import (
    get_query_points_from_extractors,
)
from gluemap.utils.load_fn import (
    load_and_preprocess_images_1024,
    load_and_preprocess_images_inner,
)
from gluemap.video.gpu_frame_stream import (
    DecodedRouteFrame,
    FrameRoute,
    GpuFrameStreamError,
    ThreadedNvdecRouteStream,
)


@dataclass
class _PreparedFrame:
    route: FrameRoute
    image: torch.Tensor
    image_1024: torch.Tensor
    image_change: list[float]
    image_change_1024: list[float]
    original_shape: tuple[int, int]


class StreamingVideoStarDataset(IterableDataset):
    """Emit center-ordered stars while retaining only their live GPU frames."""

    gpu_resident_stream = True

    def __init__(
        self,
        video_path: str | Path,
        routes: tuple[FrameRoute, ...],
        pairs: np.ndarray,
        *,
        maximum_decoder_frames: int,
        maximum_resident_frames: int,
        decode_batch_size: int = 8,
        num_tracks: int = 1024,
        gpu_id: int = 0,
        stream_factory: Callable[[], Iterator[DecodedRouteFrame]] | None = None,
        query_point_provider: Callable[[torch.Tensor], torch.Tensor]
        | None = None,
        query_extractor_checkpoint: str | Path | None = None,
        require_cuda: bool = True,
    ) -> None:
        super().__init__()
        self.video_path = Path(video_path)
        self.routes = routes
        self.maximum_decoder_frames = maximum_decoder_frames
        self.maximum_resident_frames = maximum_resident_frames
        self.decode_batch_size = decode_batch_size
        self.num_tracks = num_tracks
        self.gpu_id = gpu_id
        self.require_cuda = require_cuda
        self._stream_factory = stream_factory
        self._query_point_provider = query_point_provider
        self.query_extractor_checkpoint = (
            None
            if query_extractor_checkpoint is None
            else Path(query_extractor_checkpoint)
        )
        self._extractor: ALIKED | None = None
        self.peak_resident_frames = 0
        self.released_frame_count = 0
        self.emitted_star_count = 0

        frontend_routes = [value for value in routes if value.frontend_input]
        if not frontend_routes:
            raise GpuFrameStreamError(
                "streaming star dataset has no frontend routes"
            )
        for geometry_ordinal, route in enumerate(frontend_routes):
            if route.geometry_ordinal != geometry_ordinal:
                raise GpuFrameStreamError(
                    "frontend geometry ordinals are not contiguous"
                )
        self.frontend_routes = tuple(frontend_routes)
        self.N = len(frontend_routes)
        self.pairs = np.asarray(pairs, dtype=np.int64)
        if self.pairs.ndim != 2 or self.pairs.shape[1:] != (2,):
            raise GpuFrameStreamError("pair plan shape differs")
        if (
            self.pairs.size == 0
            or np.any(self.pairs < 0)
            or np.any(self.pairs >= self.N)
        ):
            raise GpuFrameStreamError(
                "pair plan index is outside frontend routes"
            )
        normalized_pairs = np.sort(self.pairs, axis=1)
        if np.any(normalized_pairs[:, 0] == normalized_pairs[:, 1]):
            raise GpuFrameStreamError("pair plan contains a self edge")
        self.pairs = np.unique(normalized_pairs, axis=0)

        neighbors = {index: set() for index in range(self.N)}
        for left, right in self.pairs:
            neighbors[int(left)].add(int(right))
            neighbors[int(right)].add(int(left))
        if any(not value for value in neighbors.values()):
            raise GpuFrameStreamError(
                "pair plan leaves a frontend frame isolated"
            )
        self.stars = tuple(
            np.asarray([center, *sorted(neighbors[center])], dtype=np.int64)
            for center in range(self.N)
        )
        self.image_index_to_star_index = {
            center: center for center in range(self.N)
        }
        self._ready_at_geometry = tuple(
            int(max(star.tolist())) for star in self.stars
        )
        last_use = []
        for frame_index in range(self.N):
            users = [frame_index, *neighbors[frame_index]]
            last_use.append(max(users))
        self._last_use_center = tuple(last_use)
        minimum_required_residency = max(
            self._ready_at_geometry[center] - center + 1
            for center in range(self.N)
        )
        if maximum_resident_frames < minimum_required_residency:
            raise GpuFrameStreamError(
                "maximum resident frames cannot cover pair-plan lookahead"
            )

        self.images_list = [
            f"{route.geometry_ordinal:08d}-{route.frame_uid}.video-frame"
            for route in frontend_routes
        ]
        self.images_path = [str(self.video_path)] * self.N
        self.images_shape_ori: list[tuple[int, int]] = []
        self.images_change: list[list[float]] = []
        self.intrinsics_mapping = {index: 0 for index in range(self.N)}
        self.sequential_edges = [tuple(value) for value in self.pairs.tolist()]

    def __len__(self) -> int:
        return self.N

    def _create_stream(self) -> Iterator[DecodedRouteFrame]:
        if self._stream_factory is not None:
            return self._stream_factory()
        return iter(
            ThreadedNvdecRouteStream(
                self.video_path,
                self.routes,
                maximum_in_flight_frames=self.maximum_decoder_frames,
                decode_batch_size=self.decode_batch_size,
                gpu_id=self.gpu_id,
                require_cuda=self.require_cuda,
            )
        )

    def _prepare_frame(self, decoded: DecodedRouteFrame) -> _PreparedFrame:
        tensor = decoded.tensor
        if self.require_cuda and not tensor.is_cuda:
            raise GpuFrameStreamError("streaming star input left CUDA memory")
        original_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        source = tensor.float().mul_(1.0 / 255.0)
        images, _, changes = load_and_preprocess_images_inner(
            [source], image_size=518, patch_size=14, force_square=False
        )
        images_1024, changes_1024 = load_and_preprocess_images_1024([source])
        return _PreparedFrame(
            route=decoded.route,
            image=images[0],
            image_1024=images_1024[0],
            image_change=changes[0],
            image_change_1024=changes_1024[0],
            original_shape=original_shape,
        )

    def _query_points(self, query_image: torch.Tensor) -> torch.Tensor:
        if self._query_point_provider is not None:
            return self._query_point_provider(query_image)
        if self._extractor is None:
            if (
                self.query_extractor_checkpoint is None
                or not self.query_extractor_checkpoint.is_file()
            ):
                raise GpuFrameStreamError(
                    "locked ALIKED query-extractor checkpoint is required"
                )
            original_loader = torch.hub.load_state_dict_from_url

            def load_locked_state_dict(
                url: str, *args: Any, **kwargs: Any
            ) -> dict[str, torch.Tensor]:
                if not url.endswith("/aliked-n16.pth"):
                    raise GpuFrameStreamError(
                        "ALIKED requested an unexpected checkpoint"
                    )
                return torch.load(
                    self.query_extractor_checkpoint,
                    map_location="cpu",
                    weights_only=True,
                )

            torch.hub.load_state_dict_from_url = load_locked_state_dict
            try:
                self._extractor = (
                    ALIKED(
                        max_num_keypoints=self.num_tracks,
                        detection_threshold=0.005,
                    )
                    .eval()
                    .to(query_image.device)
                )
            finally:
                torch.hub.load_state_dict_from_url = original_loader
        return get_query_points_from_extractors(
            query_image,
            [self._extractor],
            max_query_num=self.num_tracks,
            strict_num=False,
        )[0]

    def _star_item(
        self, center: int, cache: dict[int, _PreparedFrame]
    ) -> dict[str, Any]:
        indexes = self.stars[center]
        try:
            frames = [cache[int(index)] for index in indexes]
        except KeyError as error:
            raise GpuFrameStreamError(
                "ready star is missing a resident frame"
            ) from error
        images = torch.stack([value.image for value in frames])
        images_1024 = torch.stack([value.image_1024 for value in frames])
        return {
            "images": images,
            "images_change": np.asarray(
                [value.image_change for value in frames], dtype=np.float64
            ),
            "images_shape_ori": np.asarray(
                [value.original_shape for value in frames], dtype=np.int64
            ),
            "images_1024": images_1024,
            "images_change_1024": np.asarray(
                [value.image_change_1024 for value in frames], dtype=np.float64
            ),
            "image_paths": [
                f"video://{value.route.frame_uid}" for value in frames
            ],
            "frame_uids": [value.route.frame_uid for value in frames],
            "star_indexes": center,
            "indexes": indexes,
            "query_points": self._query_points(images_1024[:1]),
        }

    def __iter__(self) -> Iterator[dict[str, Any]]:
        cache: dict[int, _PreparedFrame] = {}
        next_center = 0
        self.images_shape_ori = [(-1, -1)] * self.N
        self.images_change = [[] for _ in range(self.N)]
        for decoded in self._create_stream():
            if not decoded.route.frontend_input:
                continue
            geometry_ordinal = decoded.route.geometry_ordinal
            assert geometry_ordinal is not None
            prepared = self._prepare_frame(decoded)
            cache[geometry_ordinal] = prepared
            self.images_shape_ori[geometry_ordinal] = prepared.original_shape
            self.images_change[geometry_ordinal] = prepared.image_change
            self.peak_resident_frames = max(
                self.peak_resident_frames, len(cache)
            )
            if len(cache) > self.maximum_resident_frames:
                raise GpuFrameStreamError(
                    "resident GPU frame cache exceeded its bound"
                )

            while (
                next_center < self.N
                and self._ready_at_geometry[next_center] <= geometry_ordinal
            ):
                yield self._star_item(next_center, cache)
                self.emitted_star_count += 1
                release = [
                    frame_index
                    for frame_index in cache
                    if self._last_use_center[frame_index] <= next_center
                ]
                for frame_index in release:
                    del cache[frame_index]
                    self.released_frame_count += 1
                next_center += 1
        if next_center != self.N:
            raise GpuFrameStreamError(
                "video stream ended before every star was emitted"
            )
        if cache:
            raise GpuFrameStreamError("resident GPU frame cache did not drain")
