"""Incremental Star inference for GPU-resident video frames."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch

from gluemap.controllers.star_inference import StarInferencePipeline


class StreamingStarInferenceError(ValueError):
    """Raised when incremental Star inference violates its ordering contract."""


def move_output_to_cpu(value: Any) -> Any:
    """Detach every tensor recursively before handing an output to its sink."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: move_output_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [move_output_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(move_output_to_cpu(item) for item in value)
    return value


class StreamingStarInferencePipeline(StarInferencePipeline):
    """Run center-ordered Star inference without retaining whole-video output.

    The sink owns persistence and may group consecutive centers into durable
    shards. Pixel tensors remain in the dataset/decoder GPU cache; only model
    predictions are detached to CPU immediately before the sink is called.
    """

    def _synchronize(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize()

    def run_to_sink(
        self,
        dataset: Any,
        sink: Callable[[int, dict[str, Any]], None],
    ) -> dict[str, Any]:
        if self.world_size != 1 or self.rank != 0 or self.args.distributed:
            raise StreamingStarInferenceError(
                "GPU-resident Star streaming currently requires one process"
            )
        if not getattr(dataset, "gpu_resident_stream", False):
            raise StreamingStarInferenceError(
                "streaming Star inference requires a GPU-resident dataset"
            )

        data_loader = self._make_dataloader(dataset)
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(self.device)

        load_started = time.perf_counter()
        models = self._load_models()
        model_load_seconds = time.perf_counter() - load_started
        loaded_model_keys = sorted(
            key for key in models if key != "mv"
        )
        if any(key in {"salad", "dg", "faiss"} for key in loaded_model_keys):
            raise StreamingStarInferenceError(
                "G0 Star streaming loaded a retrieval or Doppelgangers model"
            )
        batch_inference = self._create_batch_inference(models)

        expected_center = 0
        batch_seconds: list[float] = []
        forward_seconds: list[float] = []
        tracking_seconds: list[float] = []
        try:
            with torch.no_grad():
                for batch in data_loader:
                    center_value = batch[self._index_key]
                    if (
                        not isinstance(center_value, torch.Tensor)
                        or center_value.numel() != 1
                    ):
                        raise StreamingStarInferenceError(
                            "streaming Star batch must contain one center"
                        )
                    center = int(center_value.item())
                    if center != expected_center:
                        raise StreamingStarInferenceError(
                            "streaming Star centers are not contiguous"
                        )

                    self._synchronize()
                    batch_started = time.perf_counter()
                    output, timings = self._run_batch_step(
                        batch_inference, batch
                    )
                    self._synchronize()
                    batch_seconds.append(time.perf_counter() - batch_started)
                    forward_seconds.append(
                        float(timings.get("forward_times", 0.0))
                    )
                    tracking_seconds.append(
                        float(timings.get("tracking_times", 0.0))
                    )
                    sink(center, move_output_to_cpu(output))
                    expected_center += 1
        finally:
            self._release_models()

        if expected_center != len(dataset):
            raise StreamingStarInferenceError(
                "streaming Star output count differs from dataset"
            )
        peak_allocated = 0
        peak_reserved = 0
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        return {
            "starCount": expected_center,
            "modelLoadSeconds": model_load_seconds,
            "inferenceSeconds": sum(batch_seconds),
            "forwardSeconds": sum(forward_seconds),
            "trackingSeconds": sum(tracking_seconds),
            "loadedModelKeys": loaded_model_keys,
            "saladLoadCount": 0,
            "descriptorArtifactCount": 0,
            "faissArtifactCount": 0,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "peakCudaAllocatedBytes": peak_allocated,
            "peakCudaReservedBytes": peak_reserved,
            "peakResidentFrames": int(
                getattr(dataset, "peak_resident_frames", 0)
            ),
            "releasedFrameCount": int(
                getattr(dataset, "released_frame_count", 0)
            ),
        }
