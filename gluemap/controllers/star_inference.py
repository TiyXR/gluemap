from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from gluemap.controllers.base_inference import BaseInferencePipeline
from gluemap.estimators.covisibility_extraction import CovisibilityExtraction
from gluemap.estimators.track_inference import TrackInference
from gluemap.ff_inference.local_inference import create_local_inference
from gluemap.utils.model_loader import load_models

if TYPE_CHECKING:
    from gluemap.datasets.star import BaseStarDataset

logger = logging.getLogger(__name__)


def resolve_pi3_sdpa_backend(device: str | torch.device) -> str:
    """Return the Pi3 attention backend that is valid on this GPU."""
    resolved = torch.device(device)
    if resolved.type != "cuda":
        return "native"
    major, _minor = torch.cuda.get_device_capability(resolved)
    return "flash" if major >= 8 else "math"


@contextmanager
def pi3_sdpa_compatibility(
    device: str | torch.device,
) -> Iterator[str]:
    """Make upstream Pi3 bf16 attention runnable on pre-Ampere GPUs.

    Pi3 requests Flash SDPA unconditionally for bf16 tensors. PyTorch only
    provides that kernel on sm80+, so a P6000 (sm61) otherwise aborts instead
    of falling back. Keep the upstream submodule clean and replace only that
    context-manager request while one Pi3 forward is active.
    """
    backend = resolve_pi3_sdpa_backend(device)
    if backend != "math":
        yield backend
        return

    original = torch.nn.attention.sdpa_kernel

    def compatible(backends):
        values = backends if isinstance(backends, list) else [backends]
        if torch.nn.attention.SDPBackend.FLASH_ATTENTION in values:
            values = [
                torch.nn.attention.SDPBackend.MATH
                if value == torch.nn.attention.SDPBackend.FLASH_ATTENTION
                else value
                for value in values
            ]
        return original(values)

    torch.nn.attention.sdpa_kernel = compatible
    try:
        yield backend
    finally:
        torch.nn.attention.sdpa_kernel = original


class BatchInferenceStar:
    """Per-batch multi-view (star) inference driver.

    Runs the chosen multi-view model + (optional) point tracker on one star
    batch, then extracts covisibility-derived poses, intrinsics, scores and
    virtual 3D points via ``CovisibilityExtraction``.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_type: str = "pi3",
        model_track: torch.nn.Module | None = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        encoder_token_cache_frames: int = 0,
        tracker_feature_cache_frames: int = 0,
    ):
        self.model = model
        self.model_type = model_type
        self.model_track = model_track
        self.device = device
        self.dtype = dtype
        self.synchronization_count = 0
        self.resolved_attention_backend = (
            resolve_pi3_sdpa_backend(device)
            if model_type == "pi3"
            else "native"
        )

        self.local_inference = create_local_inference(
            model,
            model_type,
            device,
            dtype,
            encoder_token_cache_frames=encoder_token_cache_frames,
        )
        self.track_inference = TrackInference(
            model_track, device, tracker_feature_cache_frames
        )
        self.covisibility_extraction = CovisibilityExtraction()

    def _synchronize(self) -> None:
        device = getattr(self, "device", "cpu")
        if torch.cuda.is_available() and str(device).startswith("cuda"):
            torch.cuda.synchronize(device)
            self.synchronization_count = (
                getattr(self, "synchronization_count", 0) + 1
            )

    def main(
        self,
        batch: dict,
        use_dummy_tracks: bool = False,
        include_track: bool = True,
    ) -> dict:
        """Run multi-view + track inference on a single star batch.

        Args:
            batch: DataLoader batch; must contain ``"indexes"`` (image indices
                making up the star) and ``"images_change"`` (per-image
                image-shape rescale info).
            use_dummy_tracks: If ``True``, emit dummy tracks (query points
                expanded over frames) instead of running the VGGSfM tracker;
                track outputs are still kept in the result dict (must combine
                with ``include_track=True``).
            include_track: If ``True``, include ``"tracks"``, ``"vis"`` and
                ``"conf"`` in the returned dict.

        Returns:
            Dict with per-batch predictions; keys include ``"indexes"``,
            ``"extrinsics"``, ``"intrinsics"``, ``"pose_scores"``,
            ``"tracks_virtual"``, ``"points3d_virtual"``, ``"valid_virtual"``,
            and (when ``include_track``) ``"tracks"``, ``"vis"``, ``"conf"``.
        """
        predictions, forward_time, track_time = self._predict_images(
            batch,
            use_dummy_tracks=use_dummy_tracks,
            include_track=include_track,
        )

        self._synchronize()
        covisibility_started = time.perf_counter()
        (
            extrinsics,
            intrinsics,
            scores,
            tracks_virtual,
            points3d_virtual,
            valid_virtual,
        ) = self.covisibility_extraction.main(
            predictions,
            batch["indexes"],
            batch["images_change"],
        )
        self._synchronize()
        covisibility_time = time.perf_counter() - covisibility_started

        result_dict = {
            "indexes": batch["indexes"][0].tolist(),
            "working_image_shape": [
                int(batch["images"].shape[-2]),
                int(batch["images"].shape[-1]),
            ],
            "images_shape_ori": batch["images_shape_ori"][0].tolist(),
            "images_change": batch["images_change"][0].tolist(),
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "pose_scores": scores,
            "tracks_virtual": tracks_virtual,
            "points3d_virtual": points3d_virtual,
            "valid_virtual": valid_virtual,
            "_forward_time": forward_time,
            "_track_time": track_time,
            "_covisibility_time": covisibility_time,
        }

        if include_track:
            result_dict["tracks"] = predictions["track"].cpu()
            result_dict["vis"] = predictions["vis"].cpu()
            result_dict["conf"] = predictions["conf"].cpu()

        return result_dict

    @torch.no_grad()
    def _predict_images(
        self,
        batch: dict,
        use_dummy_tracks: bool = False,
        include_track: bool = True,
    ) -> tuple[dict, float, float]:
        """Run local + track inference.

        Returns ``(predictions, forward_time, track_time)``.
        """
        if not use_dummy_tracks and not include_track:
            raise ValueError(
                "running the real tracker without include_track would discard"
                " its output"
            )

        # Local inference (timed)
        self._synchronize()
        t0 = time.perf_counter()
        compatibility = (
            pi3_sdpa_compatibility(self.device)
            if self.model_type == "pi3"
            else nullcontext("native")
        )
        with compatibility as attention_backend:
            predictions = self.local_inference.predict(batch)
        self.resolved_attention_backend = attention_backend
        self._synchronize()
        forward_time = time.perf_counter() - t0

        # Track inference
        track_preds, track_time = self._run_track_inference(
            batch, use_dummy_tracks, include_track
        )
        predictions.update(track_preds)

        return predictions, forward_time, track_time

    def _run_track_inference(
        self,
        batch: dict,
        use_dummy_tracks: bool,
        include_track: bool,
    ) -> tuple[dict, float]:
        """Returns (track_preds_dict, track_time)."""
        if not include_track:
            return {}, 0.0

        self._synchronize()
        t0 = time.perf_counter()
        track_preds = self.track_inference.predict(
            batch=batch,
            use_dummy_tracks=use_dummy_tracks,
        )
        self._synchronize()
        track_time = time.perf_counter() - t0
        return track_preds, track_time


class StarInferencePipeline(BaseInferencePipeline):
    """Pipeline object for star (multi-view) inference."""

    _index_key: ClassVar[str] = "star_indexes"
    _rerun_from_triggers: ClassVar[frozenset[str] | None] = None
    _profiling_label: ClassVar[str] = "Star inference"

    def _batch_size(self) -> int:
        return 1

    def _load_models(self) -> dict[str, torch.nn.Module]:
        chosen_model = getattr(self.args, "chosen_model", "pi3")
        if self.models is not None:
            return self.models
        if (
            self.preloaded_models is not None
            and chosen_model in self.preloaded_models
        ):
            self.models = self.preloaded_models
            return self.models
        model_keys = (
            {chosen_model}
            if getattr(self.args, "use_dummy_tracks", False)
            else {chosen_model, "vggsfm"}
        )
        models, self.device = load_models(self.args, keys=model_keys)
        self.models = models
        self._owns_models = True
        return self.models

    def _create_batch_inference(
        self, models: dict[str, torch.nn.Module]
    ) -> BatchInferenceStar:
        chosen_model = getattr(self.args, "chosen_model", "pi3")
        return BatchInferenceStar(
            models[chosen_model],
            chosen_model,
            models.get("vggsfm"),
            device=self.device,
            dtype=self.dtype,
            encoder_token_cache_frames=getattr(
                self.args, "encoder_token_cache_frames", 0
            ),
            tracker_feature_cache_frames=getattr(
                self.args, "tracker_feature_cache_frames", 0
            ),
        )

    def _run_batch_step(
        self, batch_inference: BatchInferenceStar, batch: dict
    ) -> tuple[dict, dict[str, float]]:
        outputs = batch_inference.main(
            batch,
            use_dummy_tracks=getattr(self.args, "use_dummy_tracks", False),
        )
        extras = {
            "forward_times": outputs.pop("_forward_time", 0.0),
            "tracking_times": outputs.pop("_track_time", 0.0),
            "covisibility_times": outputs.pop("_covisibility_time", 0.0),
        }
        return outputs, extras

    def _pack_local_outputs(
        self, all_outputs: list[dict], all_indices: list[int]
    ) -> dict:
        output_keys = list(all_outputs[0].keys())
        return {
            key: [output[key] for output in all_outputs] for key in output_keys
        }

    def _merge_gathered_outputs(
        self,
        data_list: list[dict],
        index_mapping: np.ndarray,
        dataset_size: int,
    ) -> dict:
        if self.rank != 0:
            return {}
        output_keys = list(data_list[0].keys())
        predictions_dict: dict = {}
        for key in output_keys:
            gathered = [
                out[key][x] for out in data_list for x in range(len(out[key]))
            ]
            predictions_dict[key] = [
                gathered[index_mapping[i]] for i in range(len(index_mapping))
            ]
        return predictions_dict

    def _postprocess_global_outputs(
        self, global_outputs: dict, dataset: BaseStarDataset
    ) -> dict:
        return global_outputs

    def _profiling_extra(
        self,
        batch_times: list[float],
        extra_timings: dict[str, list[float]],
    ) -> str:
        forward_times = extra_timings.get("forward_times", [])
        tracking_times = extra_timings.get("tracking_times", [])
        covisibility_times = extra_timings.get("covisibility_times", [])
        return (
            f", forward={sum(forward_times):.2f}s, "
            f"tracking={sum(tracking_times):.2f}s, "
            f"covisibility={sum(covisibility_times):.2f}s"
        )


def run_star_inference(
    args: argparse.Namespace,
    dataset: BaseStarDataset,
    world_size: int,
    rank: int,
    file_name: str = "star_result.pth",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    preloaded_models: dict[str, torch.nn.Module] | None = None,
):
    """Module-level wrapper to instantiate StarInferencePipeline and run it."""
    pipeline = StarInferencePipeline(
        args,
        world_size,
        rank,
        file_name=file_name,
        device=device,
        dtype=dtype,
        preloaded_models=preloaded_models,
    )
    return pipeline.run(dataset)
