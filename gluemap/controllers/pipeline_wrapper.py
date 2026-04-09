import logging
import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import os
import numpy as np


logger = logging.getLogger(__name__)
from gluemap.controllers.star_inference import BatchInferenceStar
from gluemap.utils.gpu_utils import all_gather_object_cpu, synchronize
from gluemap.utils.model_loader import load_models


STAGE_FILES = {
    "retrieval": ["salad_descriptors.pt"],
    "twoview": ["twoview_result.pth"],
    "star": ["star_result.pth"],
}
STAGE_ORDER = ["retrieval", "twoview", "star"]


def is_stage_cached(args, file_name):
    """Return True if force_load is enabled and the cached result file exists."""
    return args.force_load and os.path.exists(
        os.path.join(args.curr_path, file_name)
    )


def invalidate_cache_from(args, stage):
    """Delete cached files from `stage` onward so they get recomputed."""
    if stage is None:
        return
    idx = STAGE_ORDER.index(stage)
    for s in STAGE_ORDER[idx:]:
        for fname in STAGE_FILES[s]:
            if s == "retrieval" and getattr(args, "curr_processed", None):
                path = os.path.join(args.curr_processed, fname)
            else:
                path = os.path.join(args.curr_path, fname)
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"[rerun_from={stage}] Deleted {path}")


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

    # TODO: write function for generating samper and dataloader
    if args.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Important for inference to maintain order
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    # Here, since the dataset is heterogeneous, we use batch size 1
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=1,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    all_outputs = []
    all_indices = []
    batch_times = []
    forward_times = []
    tracking_times = []
    chosen_model = getattr(args, "chosen_model", "pi3")
    t_model_load = 0.0
    if not is_stage_cached(args, file_name):
        # Lazy load: resolve models only when inference is needed
        t0_load = time.perf_counter()
        if preloaded_models is not None and chosen_model in preloaded_models:
            models = preloaded_models
        else:
            model_keys = {chosen_model} if getattr(args, "disable_tracking", False) else {chosen_model, "vggsfm"}
            models, device = load_models(args, keys=model_keys)
        t_model_load = time.perf_counter() - t0_load
        star_inferece = BatchInferenceStar(
            models[chosen_model], chosen_model, models.get("vggsfm"), device=device, dtype=dtype,
            pointmap_dir=os.path.join(args.curr_path, "pointmap"),
            repredict_verified=getattr(args, "repredict_verified", False),
            repredict_threshold=getattr(args, "repredict_threshold", 0.05),
        )
        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc=f"Inference (Rank {rank})", disable=rank != 0
            ):
                torch.cuda.synchronize()
                t_batch_start = time.perf_counter()

                outputs = star_inferece.main(batch, disable_track=getattr(args, "disable_tracking", False))

                torch.cuda.synchronize()
                t_batch_end = time.perf_counter()
                batch_times.append(t_batch_end - t_batch_start)
                forward_times.append(outputs.pop("_forward_time", 0.0))
                tracking_times.append(outputs.pop("_track_time", 0.0))

                # Store results (adjust based on your needs)
                all_outputs.append(outputs)
                all_indices.extend(batch["star_indexes"].cpu().numpy().tolist())

        # collect the output
        output_keys = list(all_outputs[0].keys())
        local_outputs = {
            key: [output[key] for output in all_outputs] for key in output_keys
        } | {"star_indexes": all_indices}

        output_keys = output_keys + ["star_indexes"]

        if args.distributed:
            data_list = all_gather_object_cpu(
                local_outputs,
                tmpdir=args.temp_path + "/tmp_save",
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

            # For the gathering here, since they are heterogeneous, we need to gather them separately
            predictions_dict = {}
            if rank == 0:
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

        if rank == 0:
            if save_intermediate_results:
                torch.save(predictions_dict, os.path.join(args.curr_path, file_name))
    else:
        logger.info("Loading existing results...")
        predictions_dict = torch.load(os.path.join(args.curr_path, file_name), weights_only=False)

    star_timing = {
        "batch_times": batch_times,
        "forward_times": forward_times,
        "tracking_times": tracking_times,
        "num_batches": len(batch_times),
        "total": sum(batch_times) if batch_times else 0.0,
        "model_loading": t_model_load,
    }
    if rank == 0 and batch_times:
        logger.info(f"[Profiling] Star inference: {len(batch_times)} batches, "
              f"total={sum(batch_times):.2f}s, mean={sum(batch_times)/len(batch_times):.3f}s/batch, "
              f"forward={sum(forward_times):.2f}s, tracking={sum(tracking_times):.2f}s")

    return predictions_dict, star_timing
