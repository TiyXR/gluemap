import logging
import os
import time

import torch
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from gluemap.estimators.covisibility_extraction import CovisibilityExtraction
from gluemap.ff_inference.local_inference import create_local_inference
from gluemap.estimators.track_inference import TrackInference
from gluemap.utils.gpu_utils import all_gather_object_cpu
from gluemap.utils.model_loader import load_models

logger = logging.getLogger(__name__)


class BatchInferenceStar:
    def __init__(
        self,
        model,
        model_type="pi3",
        model_track=None,
        device="cuda",
        dtype=torch.bfloat16,
        pointmap_dir=None,
    ):
        self.model = model
        self.model_type = model_type
        self.model_track = model_track
        self.device = device
        self.dtype = dtype
        self.pointmap_dir = pointmap_dir

        self.local_inference = create_local_inference(model, model_type, device, dtype)
        self.track_inference = TrackInference(model_track, device)
        self.covisibility_extraction = CovisibilityExtraction()

    def main(self, batch, disable_track=False, include_track=True):
        predictions, forward_time, track_time = self._predict_images(
            batch, disable_track=disable_track, include_track=include_track
        )

        (
            extrinsics,
            intrinsics,
            scores,
            tracks_virtual,
            points3d_virtual,
            valid_virtual,
            cam_points,
            cam_points_conf,
            depth_transformed,
        ) = self.covisibility_extraction.main(
            predictions,
            batch["indexes"],
            batch["images_change"],
        )

        result_dict = {
            "indexes": batch["indexes"][0].tolist(),
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "pose_scores": scores,
            "tracks_virtual": tracks_virtual,
            "points3d_virtual": points3d_virtual,
            "valid_virtual": valid_virtual,
            "_forward_time": forward_time,
            "_track_time": track_time,
        }

        if include_track:
            result_dict["tracks"] = predictions["track"].cpu()
            result_dict["vis"] = predictions["vis"].cpu()
            result_dict["conf"] = predictions["conf"].cpu()

        return result_dict

    @torch.no_grad()
    def _predict_images(self, batch, disable_track=False, include_track=True):
        if not disable_track and not include_track:
            raise ValueError("if not disable_track, include_track should be True")

        # Local inference (timed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictions = self.local_inference.predict(batch)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - t0

        # Track inference
        track_preds, track_time = self._run_track_inference(
            batch, disable_track, include_track
        )
        predictions.update(track_preds)

        return predictions, forward_time, track_time

    def _run_track_inference(self, batch, disable_track, include_track):
        """Returns (track_preds_dict, track_time)."""
        if not include_track:
            return {}, 0.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        track_preds = self.track_inference.predict(
            batch=batch,
            disable_track=disable_track,
        )
        torch.cuda.synchronize()
        track_time = time.perf_counter() - t0
        return track_preds, track_time


class StarInferencePipeline:
    """Pipeline object for star (multi-view) inference.

    Follows the same pattern as ``TwoViewInferencePipeline``: stable config is
    stored as instance attributes; per-dataset inputs are passed to ``run()``.
    """

    def __init__(
        self,
        args,
        world_size,
        rank,
        file_name="star_result.pth",
        device="cuda",
        dtype=torch.bfloat16,
        preloaded_models=None,
    ):
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.file_name = file_name
        self.device = device
        self.dtype = dtype
        self.preloaded_models = preloaded_models

    def _make_dataloader(self, dataset):
        if self.args.distributed:
            sampler = DistributedSampler(
                dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        return torch.utils.data.DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _load_models(self):
        chosen_model = getattr(self.args, "chosen_model", "pi3")
        if self.preloaded_models is not None and chosen_model in self.preloaded_models:
            return self.preloaded_models, chosen_model
        model_keys = (
            {chosen_model}
            if getattr(self.args, "disable_tracking", False)
            else {chosen_model, "vggsfm"}
        )
        models, self.device = load_models(self.args, keys=model_keys)
        return models, chosen_model

    def _create_batch_inference(self, models, chosen_model):
        return BatchInferenceStar(
            models[chosen_model],
            chosen_model,
            models.get("vggsfm"),
            device=self.device,
            dtype=self.dtype,
            pointmap_dir=os.path.join(self.args.curr_path, "pointmap"),
        )

    def _run_inference(self, data_loader):
        all_outputs = []
        all_indices = []
        batch_times = []
        forward_times = []
        tracking_times = []

        t0_load = time.perf_counter()
        models, chosen_model = self._load_models()
        t_model_load = time.perf_counter() - t0_load

        star_inference = self._create_batch_inference(models, chosen_model)

        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc=f"Inference (Rank {self.rank})",
                disable=self.rank != 0,
            ):
                torch.cuda.synchronize()
                t_batch_start = time.perf_counter()

                outputs = star_inference.main(
                    batch,
                    disable_track=getattr(self.args, "disable_tracking", False),
                )

                torch.cuda.synchronize()
                t_batch_end = time.perf_counter()
                batch_times.append(t_batch_end - t_batch_start)
                forward_times.append(outputs.pop("_forward_time", 0.0))
                tracking_times.append(outputs.pop("_track_time", 0.0))

                all_outputs.append(outputs)
                all_indices.extend(batch["star_indexes"].cpu().numpy().tolist())

        return (
            all_outputs,
            all_indices,
            batch_times,
            forward_times,
            tracking_times,
            t_model_load,
        )

    def _gather_outputs(self, all_outputs, all_indices, dataset):
        output_keys = list(all_outputs[0].keys())
        local_outputs = {
            key: [output[key] for output in all_outputs] for key in output_keys
        } | {"star_indexes": all_indices}

        output_keys = output_keys + ["star_indexes"]

        if self.args.distributed:
            data_list = all_gather_object_cpu(
                local_outputs,
                tmpdir=self.args.temp_path + "/tmp_save",
                rank_zero_return_only=False,
                use_system_tmp=False,
            )

            star_index_order = [
                output["star_indexes"][x]
                for output in data_list
                for x in range(len(output["star_indexes"]))
            ]
            index_mapping = [0 for _ in range(len(dataset))]
            for i, pair_index in enumerate(star_index_order):
                index_mapping[pair_index] = i

            predictions_dict = {}
            if self.rank == 0:
                for key in output_keys:
                    gathered_outputs = [
                        output[key][x]
                        for output in data_list
                        for x in range(len(output[key]))
                    ]
                    predictions_dict[key] = [
                        gathered_outputs[index_mapping[i]]
                        for i in range(len(index_mapping))
                    ]
        else:
            predictions_dict = {
                key: [local_outputs[key][x] for x in range(len(local_outputs[key]))]
                for key in output_keys
            }

        return predictions_dict

    def run(self, dataset, save_intermediate_results=True):
        """Run star inference on the given dataset.

        Returns:
            Tuple of (predictions_dict, star_timing).
        """
        args = self.args
        file_name = self.file_name

        data_loader = self._make_dataloader(dataset)

        batch_times = []
        forward_times = []
        tracking_times = []
        t_model_load = 0.0

        from gluemap.controllers.pipeline_wrapper import is_stage_cached

        if not is_stage_cached(args, file_name):
            (
                all_outputs,
                all_indices,
                batch_times,
                forward_times,
                tracking_times,
                t_model_load,
            ) = self._run_inference(data_loader)

            predictions_dict = self._gather_outputs(all_outputs, all_indices, dataset)

            if self.rank == 0 and save_intermediate_results:
                torch.save(
                    predictions_dict, os.path.join(args.curr_path, file_name)
                )
        else:
            logger.info("Loading existing results...")
            predictions_dict = torch.load(
                os.path.join(args.curr_path, file_name), weights_only=False
            )

        star_timing = {
            "batch_times": batch_times,
            "forward_times": forward_times,
            "tracking_times": tracking_times,
            "num_batches": len(batch_times),
            "total": sum(batch_times) if batch_times else 0.0,
            "model_loading": t_model_load,
        }
        if self.rank == 0 and batch_times:
            logger.info(
                f"[Profiling] Star inference: {len(batch_times)} batches, "
                f"total={sum(batch_times):.2f}s, "
                f"mean={sum(batch_times) / len(batch_times):.3f}s/batch, "
                f"forward={sum(forward_times):.2f}s, "
                f"tracking={sum(tracking_times):.2f}s"
            )

        return predictions_dict, star_timing


def run_star_inference(
    args,
    dataset,
    world_size,
    rank,
    file_name="star_result.pth",
    save_intermediate_results=True,
    device="cuda",
    dtype=torch.bfloat16,
    preloaded_models=None,
):
    """Module-level wrapper that instantiates StarInferencePipeline and runs it."""
    pipeline = StarInferencePipeline(
        args,
        world_size,
        rank,
        file_name=file_name,
        device=device,
        dtype=dtype,
        preloaded_models=preloaded_models,
    )
    return pipeline.run(dataset, save_intermediate_results=save_intermediate_results)
