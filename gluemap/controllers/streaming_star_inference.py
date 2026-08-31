"""Incremental Star inference for GPU-resident video frames."""

from __future__ import annotations

import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils.data import DataLoader, default_collate

from gluemap.controllers.star_inference import (
    CudaEventInterval,
    StarInferencePipeline,
)


class StreamingStarInferenceError(ValueError):
    """Raised when incremental Star inference violates its ordering contract."""


PROFILE_CONTRACT_ID = "jarailsense.gluemap-streaming-frontend-profile/v1"


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile / 100.0
    lower = int(position)
    upper = min(len(ordered) - 1, lower + 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def timing_summary(values: list[float]) -> dict[str, float | int]:
    total = sum(values)
    return {
        "count": len(values),
        "totalSeconds": total,
        "meanSeconds": total / len(values) if values else 0.0,
        "p50Seconds": _percentile(values, 50.0),
        "p95Seconds": _percentile(values, 95.0),
        "maximumSeconds": max(values, default=0.0),
    }


@dataclass
class CudaTimeline:
    """Aggregate hot-path activity into bounded one-second buckets."""

    bucket_seconds: float = 1.0
    buckets: dict[int, dict[str, float | int]] = field(default_factory=dict)

    def add(
        self,
        *,
        elapsed_seconds: float,
        star_count: int,
        frame_invocations: int,
        data_wait_seconds: float,
        inference_seconds: float,
        forward_seconds: float,
        tracking_seconds: float,
        covisibility_seconds: float,
        output_transfer_seconds: float,
        sink_seconds: float,
        allocated_bytes: int,
        reserved_bytes: int,
    ) -> None:
        index = max(0, int(elapsed_seconds / self.bucket_seconds))
        bucket = self.buckets.setdefault(
            index,
            {
                "bucketIndex": index,
                "startSeconds": index * self.bucket_seconds,
                "endSeconds": (index + 1) * self.bucket_seconds,
                "starCount": 0,
                "frameInvocationCount": 0,
                "dataWaitSeconds": 0.0,
                "inferenceSeconds": 0.0,
                "forwardSeconds": 0.0,
                "trackingSeconds": 0.0,
                "covisibilitySeconds": 0.0,
                "outputTransferSeconds": 0.0,
                "sinkSeconds": 0.0,
                "peakCudaAllocatedBytes": 0,
                "peakCudaReservedBytes": 0,
            },
        )
        bucket["starCount"] += star_count
        bucket["frameInvocationCount"] += frame_invocations
        for key, value in (
            ("dataWaitSeconds", data_wait_seconds),
            ("inferenceSeconds", inference_seconds),
            ("forwardSeconds", forward_seconds),
            ("trackingSeconds", tracking_seconds),
            ("covisibilitySeconds", covisibility_seconds),
            ("outputTransferSeconds", output_transfer_seconds),
            ("sinkSeconds", sink_seconds),
        ):
            bucket[key] += value
        bucket["peakCudaAllocatedBytes"] = max(
            int(bucket["peakCudaAllocatedBytes"]), allocated_bytes
        )
        bucket["peakCudaReservedBytes"] = max(
            int(bucket["peakCudaReservedBytes"]), reserved_bytes
        )

    def to_json(self) -> list[dict[str, float | int]]:
        return [self.buckets[index] for index in sorted(self.buckets)]


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


def resolve_deferred_timing(value: Any) -> float:
    """Resolve CUDA-event timing after the batch output has reached CPU."""
    if isinstance(value, CudaEventInterval):
        return value.elapsed_seconds()
    return float(value)


def _slice_batched_output(value: Any, index: int, batch_size: int) -> Any:
    if (
        isinstance(value, torch.Tensor)
        and value.ndim > 0
        and value.shape[0] == batch_size
    ):
        return value[index : index + 1]
    if isinstance(value, dict):
        return {
            key: _slice_batched_output(item, index, batch_size)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _slice_batched_output(item, index, batch_size) for item in value
        ]
    if isinstance(value, tuple):
        return tuple(
            _slice_batched_output(item, index, batch_size) for item in value
        )
    return value


class StreamingStarInferencePipeline(StarInferencePipeline):
    """Run center-ordered Star inference without retaining whole-video output.

    The sink owns persistence and may group consecutive centers into durable
    shards. Pixel tensors remain in the dataset/decoder GPU cache; only model
    predictions are detached to CPU immediately before the sink is called.
    """

    def _synchronize(self) -> None:
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.synchronize(self.device)
            self._stream_synchronization_count += 1

    def _make_dataloader(self, dataset: Any) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=None,
            num_workers=0,
            pin_memory=False,
        )

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
        forbidden_modules = (
            "faiss",
            "vpr_model",
            "mast3r",
            "gluemap.controllers.image_retrieval",
        )
        loaded_forbidden_before = sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in forbidden_modules
            )
        )
        if loaded_forbidden_before and getattr(
            self.args, "enforce_g0_clean_process", True
        ):
            raise StreamingStarInferenceError(
                "G0 process imported retrieval/Doppelgangers modules: "
                + ",".join(loaded_forbidden_before)
            )

        data_loader = self._make_dataloader(dataset)
        context_microbatch_size = int(
            getattr(self.args, "context_microbatch_size", 1)
        )
        if context_microbatch_size < 1:
            raise StreamingStarInferenceError(
                "context microbatch size must be positive"
            )
        self._stream_synchronization_count = 0
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.init()
            torch.cuda.reset_peak_memory_stats(torch.device(self.device))

        load_started = time.perf_counter()
        models = self._load_models()
        model_load_seconds = time.perf_counter() - load_started
        loaded_model_keys = sorted(key for key in models if key != "mv")
        if any(key in {"salad", "dg", "faiss"} for key in loaded_model_keys):
            raise StreamingStarInferenceError(
                "G0 Star streaming loaded a retrieval or Doppelgangers model"
            )
        batch_inference = self._create_batch_inference(models)

        expected_center = 0
        batch_seconds: list[float] = []
        forward_seconds: list[float] = []
        tracking_seconds: list[float] = []
        covisibility_seconds: list[float] = []
        covisibility_reports: list[dict[str, Any]] = []
        data_wait_seconds: list[float] = []
        output_transfer_seconds: list[float] = []
        sink_seconds: list[float] = []
        unique_frame_indexes: set[int] = set()
        frame_invocation_count = 0
        context_microbatch_sizes: list[int] = []
        cuda_timeline = CudaTimeline()
        run_started = time.perf_counter()
        wait_started = time.perf_counter()
        release_will_empty_cache = bool(
            self._owns_models
            and self.models is not None
            and torch.cuda.is_available()
            and str(self.device).startswith("cuda")
        )
        try:
            with torch.no_grad():
                pending_items: list[dict[str, Any]] = []
                pending_view_count: int | None = None

                def process_pending() -> None:
                    nonlocal \
                        expected_center, \
                        frame_invocation_count, \
                        wait_started
                    if not pending_items:
                        return
                    batch = default_collate(pending_items)
                    batch_ready = time.perf_counter()
                    data_wait = batch_ready - wait_started
                    data_wait_seconds.append(data_wait)
                    center_values = batch[self._index_key]
                    if not isinstance(center_values, torch.Tensor):
                        raise StreamingStarInferenceError(
                            "streaming Star batch centers must be a tensor"
                        )
                    centers = [int(value) for value in center_values.flatten()]
                    if centers != list(
                        range(expected_center, expected_center + len(centers))
                    ):
                        raise StreamingStarInferenceError(
                            "streaming Star centers are not contiguous"
                        )
                    context_microbatch_sizes.append(len(centers))

                    batch_started = time.perf_counter()
                    output, timings = self._run_batch_step(
                        batch_inference, batch
                    )
                    covisibility_report = timings.get("covisibility_reports")
                    if (
                        isinstance(covisibility_report, dict)
                        and covisibility_report
                    ):
                        covisibility_reports.append(covisibility_report)
                    raw_indexes = batch.get("indexes")
                    indexes = [
                        int(value) for value in raw_indexes.flatten().tolist()
                    ]
                    unique_frame_indexes.update(indexes)
                    frame_invocation_count += len(indexes)

                    transfer_started = time.perf_counter()
                    cpu_batch_output = move_output_to_cpu(output)
                    batch_seconds.append(time.perf_counter() - batch_started)
                    transfer_elapsed = time.perf_counter() - transfer_started
                    output_transfer_seconds.append(transfer_elapsed)
                    forward_seconds.append(
                        resolve_deferred_timing(
                            timings.get("forward_times", 0.0)
                        )
                    )
                    tracking_seconds.append(
                        resolve_deferred_timing(
                            timings.get("tracking_times", 0.0)
                        )
                    )
                    covisibility_seconds.append(
                        resolve_deferred_timing(
                            timings.get("covisibility_times", 0.0)
                        )
                    )
                    sink_started = time.perf_counter()
                    for batch_index, center in enumerate(centers):
                        cpu_output = _slice_batched_output(
                            cpu_batch_output, batch_index, len(centers)
                        )
                        cpu_output["indexes"] = batch["indexes"][
                            batch_index
                        ].tolist()
                        if "images_shape_ori" in batch:
                            cpu_output["images_shape_ori"] = batch[
                                "images_shape_ori"
                            ][batch_index].tolist()
                        if "images_change" in batch:
                            cpu_output["images_change"] = batch[
                                "images_change"
                            ][batch_index].tolist()
                        sink(center, cpu_output)
                    sink_elapsed = time.perf_counter() - sink_started
                    sink_seconds.append(sink_elapsed)
                    allocated_bytes = 0
                    reserved_bytes = 0
                    if torch.cuda.is_available() and str(
                        self.device
                    ).startswith("cuda"):
                        allocated_bytes = int(
                            torch.cuda.memory_allocated(self.device)
                        )
                        reserved_bytes = int(
                            torch.cuda.memory_reserved(self.device)
                        )
                    cuda_timeline.add(
                        elapsed_seconds=time.perf_counter() - run_started,
                        star_count=len(centers),
                        frame_invocations=len(indexes),
                        data_wait_seconds=data_wait,
                        inference_seconds=batch_seconds[-1],
                        forward_seconds=forward_seconds[-1],
                        tracking_seconds=tracking_seconds[-1],
                        covisibility_seconds=covisibility_seconds[-1],
                        output_transfer_seconds=transfer_elapsed,
                        sink_seconds=sink_elapsed,
                        allocated_bytes=allocated_bytes,
                        reserved_bytes=reserved_bytes,
                    )
                    expected_center += len(centers)
                    wait_started = time.perf_counter()

                for item in data_loader:
                    raw_indexes = item.get("indexes")
                    view_count = int(len(raw_indexes))
                    if pending_items and (
                        view_count != pending_view_count
                        or len(pending_items) >= context_microbatch_size
                    ):
                        process_pending()
                        pending_items = []
                    pending_items.append(item)
                    pending_view_count = view_count
                    if len(pending_items) >= context_microbatch_size:
                        process_pending()
                        pending_items = []
                        pending_view_count = None
                process_pending()
        finally:
            self._release_models()

        if expected_center != len(dataset):
            raise StreamingStarInferenceError(
                "streaming Star output count differs from dataset"
            )
        loaded_forbidden = sorted(
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(f"{prefix}.")
                for prefix in forbidden_modules
            )
        )
        imported_forbidden = sorted(
            set(loaded_forbidden) - set(loaded_forbidden_before)
        )
        if imported_forbidden:
            raise StreamingStarInferenceError(
                "G0 inference imported retrieval/Doppelgangers modules: "
                + ",".join(imported_forbidden)
            )
        peak_allocated = 0
        peak_reserved = 0
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            peak_allocated = int(torch.cuda.max_memory_allocated(self.device))
            peak_reserved = int(torch.cuda.max_memory_reserved(self.device))
        batch_synchronization_count = int(
            getattr(batch_inference, "synchronization_count", 0)
        )
        token_cache_report = None
        local_inference = getattr(batch_inference, "local_inference", None)
        if local_inference is not None and hasattr(
            local_inference, "encoder_token_cache_report"
        ):
            token_cache_report = local_inference.encoder_token_cache_report()
        actual_encoder_invocations = (
            int(token_cache_report["encodedFrameCount"])
            if token_cache_report is not None
            else frame_invocation_count
        )
        tracker_feature_cache_report = None
        track_inference = getattr(batch_inference, "track_inference", None)
        if track_inference is not None and hasattr(
            track_inference, "feature_cache_report"
        ):
            tracker_feature_cache_report = (
                track_inference.feature_cache_report()
            )
        stage_timings = {
            "dataWait": timing_summary(data_wait_seconds),
            "inference": timing_summary(batch_seconds),
            "forward": timing_summary(forward_seconds),
            "tracking": timing_summary(tracking_seconds),
            "covisibility": timing_summary(covisibility_seconds),
            "outputTransfer": timing_summary(output_transfer_seconds),
            "sink": timing_summary(sink_seconds),
        }
        covisibility_graph_report = None
        if covisibility_reports:
            identity_keys = (
                "graphPolicy",
                "virtualTrackDepthNoiseRatio",
                "virtualTrackNoiseSeed",
                "virtualTrackNoiseSeedPolicy",
            )
            identity = {
                key: covisibility_reports[0].get(key) for key in identity_keys
            }
            if any(
                any(report.get(key) != value for key, value in identity.items())
                for report in covisibility_reports[1:]
            ):
                raise StreamingStarInferenceError(
                    "covisibility policy changed during one streaming run"
                )
            covisibility_graph_report = {
                "contractId": "jarailsense.gluemap-covisibility-run/v1",
                **identity,
                "starCount": sum(
                    int(report.get("batchCount", 1))
                    for report in covisibility_reports
                ),
                "evaluatedDirectedPairCount": sum(
                    int(report["evaluatedDirectedPairCount"])
                    for report in covisibility_reports
                ),
                "denseEquivalentDirectedPairCount": sum(
                    int(report["denseEquivalentDirectedPairCount"])
                    for report in covisibility_reports
                ),
            }
            dense_count = int(
                covisibility_graph_report["denseEquivalentDirectedPairCount"]
            )
            covisibility_graph_report["evaluatedToDenseRatio"] = (
                float(covisibility_graph_report["evaluatedDirectedPairCount"])
                / dense_count
                if dense_count
                else 0.0
            )
        return {
            "profileContractId": PROFILE_CONTRACT_ID,
            "starCount": expected_center,
            "contextBatchCount": len(context_microbatch_sizes),
            "contextMicrobatchSize": context_microbatch_size,
            "peakContextMicrobatchSize": max(
                context_microbatch_sizes, default=0
            ),
            "overlappingContextCount": sum(
                max(0, value - 1) for value in context_microbatch_sizes
            ),
            "contextMicrobatchHistogram": {
                str(value): context_microbatch_sizes.count(value)
                for value in sorted(set(context_microbatch_sizes))
            },
            "modelLoadCount": 1,
            "timingBackend": getattr(
                batch_inference,
                "timing_backend",
                "synchronized-perf-counter/v1",
            ),
            "outputTransferIncludesGpuDrain": bool(
                getattr(batch_inference, "timing_backend", "")
                == "cuda-event-d2h-drain/v1"
            ),
            "modelLoadSeconds": model_load_seconds,
            "frontendRunSeconds": time.perf_counter() - run_started,
            "inferenceSeconds": sum(batch_seconds),
            "forwardSeconds": sum(forward_seconds),
            "trackingSeconds": sum(tracking_seconds),
            "covisibilitySeconds": sum(covisibility_seconds),
            "dataWaitSeconds": sum(data_wait_seconds),
            "outputTransferSeconds": sum(output_transfer_seconds),
            "sinkSeconds": sum(sink_seconds),
            "timingSummary": stage_timings,
            "cudaTimelineBucketSeconds": cuda_timeline.bucket_seconds,
            "cudaTimeline": cuda_timeline.to_json(),
            "encoderAccountingMode": (
                "frame-token-cache/v1"
                if token_cache_report is not None
                else "current-star-full-forward/v1"
            ),
            "pi3EncoderUniqueFrameCount": len(unique_frame_indexes),
            "pi3EncoderInvocationFrameCount": actual_encoder_invocations,
            "duplicateEncoderFrameInvocationCount": (
                actual_encoder_invocations - len(unique_frame_indexes)
            ),
            "encoderTokenCache": token_cache_report,
            "trackerFeatureCache": tracker_feature_cache_report,
            "covisibilityGraph": covisibility_graph_report,
            "synchronizationCount": (
                self._stream_synchronization_count + batch_synchronization_count
            ),
            "streamSynchronizationCount": self._stream_synchronization_count,
            "batchSynchronizationCount": batch_synchronization_count,
            "hotPathEmptyCacheCount": 0,
            "modelReleaseEmptyCacheCount": int(release_will_empty_cache),
            "loadedModelKeys": loaded_model_keys,
            "saladLoadCount": 0,
            "descriptorArtifactCount": 0,
            "faissArtifactCount": 0,
            "forbiddenModulesPresentBefore": loaded_forbidden_before,
            "forbiddenModuleImportDelta": imported_forbidden,
            "device": str(self.device),
            "dtype": str(self.dtype),
            "resolvedAttentionBackend": getattr(
                batch_inference, "resolved_attention_backend", "native"
            ),
            "peakCudaAllocatedBytes": peak_allocated,
            "peakCudaReservedBytes": peak_reserved,
            "peakResidentFrames": int(
                getattr(dataset, "peak_resident_frames", 0)
            ),
            "releasedFrameCount": int(
                getattr(dataset, "released_frame_count", 0)
            ),
        }
