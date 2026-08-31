"""GPU-resident reference-landmark tracking for dense pose-only localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gluemap.estimators.track_inference import TrackInference
from gluemap.math.scaling import rescale_tracks_single, standardize_query_points


@dataclass(frozen=True)
class DensePoseTrackedFrame:
    query_index: int
    query_xy: torch.Tensor
    visible: torch.Tensor
    confidence: torch.Tensor


class DensePoseTrackInference:
    """Track persisted reference observations into a microbatch of query frames."""

    def __init__(
        self,
        model_track: torch.nn.Module,
        *,
        device: str = "cuda",
        feature_cache_frames: int = 24,
    ) -> None:
        if not device.startswith("cuda") or not torch.cuda.is_available():
            raise ValueError("dense pose tracker requires CUDA")
        self.tracker = TrackInference(
            model_track,
            device=device,
            feature_cache_frames=feature_cache_frames,
        )
        self.device = torch.device(device)

    def predict(
        self,
        *,
        images_518: torch.Tensor,
        images_1024: torch.Tensor,
        image_changes_518: np.ndarray,
        image_changes_1024: np.ndarray,
        frame_uids: list[str],
        reference_xy_518: torch.Tensor,
        minimum_visibility: float,
        minimum_confidence: float,
    ) -> tuple[DensePoseTrackedFrame, ...]:
        if images_518.ndim != 4 or images_1024.ndim != 4:
            raise ValueError("dense pose tracker images must have shape N,C,H,W")
        frame_count = int(images_518.shape[0])
        if frame_count < 2 or images_1024.shape[0] != frame_count:
            raise ValueError("dense pose tracker requires reference plus query frames")
        if len(frame_uids) != frame_count:
            raise ValueError("dense pose tracker frame identity count differs")
        if image_changes_518.shape != (frame_count, 4) or image_changes_1024.shape != (
            frame_count,
            4,
        ):
            raise ValueError("dense pose tracker preprocessing identity differs")
        query_points = reference_xy_518.to(self.device).clone()
        query_points = rescale_tracks_single(query_points, image_changes_518[0])
        query_points = standardize_query_points(
            query_points, image_changes_1024[0]
        )
        if query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
        batch: dict[str, Any] = {
            "images": images_518.unsqueeze(0),
            "images_1024": images_1024.unsqueeze(0),
            "query_points": query_points,
            "indexes": torch.arange(frame_count).unsqueeze(0),
            "images_change": image_changes_518[None, ...],
            "images_change_1024": image_changes_1024[None, ...],
            "frame_uids": frame_uids,
        }
        output = self.tracker.predict(batch)
        tracks = output["track"][0]
        visibility = output["vis"][0]
        confidence = output["conf"][0]
        return tuple(
            DensePoseTrackedFrame(
                query_index=query_index,
                query_xy=tracks[query_index],
                visible=(
                    (visibility[query_index] >= minimum_visibility)
                    & (confidence[query_index] >= minimum_confidence)
                ),
                confidence=confidence[query_index],
            )
            for query_index in range(1, frame_count)
        )

    def cache_report(self) -> dict[str, Any] | None:
        return self.tracker.feature_cache_report()
