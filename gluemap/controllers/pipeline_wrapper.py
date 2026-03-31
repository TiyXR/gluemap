import time
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm
import os
import numpy as np


from gluemap.controllers.run_salad_retrival import SALADRetrieval
from gluemap.controllers.run_twoview_inference import BatchInferenceDG
from gluemap.controllers.run_star_inference import BatchInferenceStar
from gluemap.utils.gpu_utils import all_gather_object_cpu, synchronize


STAGE_FILES = {
    "retrieval": ["salad_descriptors.pt"],
    "twoview": ["twoview_result.pth"],
    "star": ["star_result.pth"],
}
STAGE_ORDER = ["retrieval", "twoview", "star"]


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
                print(f"[rerun_from={stage}] Deleted {path}")


def run_salad_retrieval(
    model,
    args,
    world_size,
    rank,
    file_name="salad_descriptors.pt",
    device="cuda",
    dtype=torch.bfloat16,
):

    base_path = args.curr_path
    if args.curr_processed:
        base_path = args.curr_processed
    if not args.force_load or not os.path.exists(
        os.path.join(base_path, file_name)
    ):
        if rank == 0:
            print("Computing SALAD descriptors...")
            salad_retrieval = SALADRetrieval(model, args, device=device, dtype=dtype)

            descriptors = salad_retrieval.main()

            os.makedirs(base_path, exist_ok=True)
            torch.save(descriptors, os.path.join(base_path, file_name))

        if args.distributed:
            synchronize()

    return


def run_twoview_inference(
    model,
    args,
    dataset_pair,
    world_size,
    rank,
    file_name="twoview_result.pth",
    save_intermediate_results=True,
    device="cuda",
    dtype=torch.bfloat16,
):
    # TODO: write function for generating sampler and dataloader
    if args.distributed:
        sampler = DistributedSampler(
            dataset_pair,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,  # Important for inference to maintain order
        )
    else:
        sampler = torch.utils.data.SequentialSampler(dataset_pair)

    data_loader = torch.utils.data.DataLoader(
        dataset_pair,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    two_view_inference = BatchInferenceDG(
        model, device=device, dtype=dtype, store_dense=False
    )

    # 7. Storage for results (adjust based on your needs)
    all_outputs = []
    all_indices = []
    batch_times = []

    # 8. Run inference
    if not args.force_load or not os.path.exists(
        os.path.join(args.curr_path, file_name)
    ):
        with torch.no_grad():
            for batch in tqdm(
                data_loader, desc=f"Inference (Rank {rank})", disable=rank != 0
            ):
                torch.cuda.synchronize()
                t_batch_start = time.perf_counter()

                outputs = two_view_inference.main(batch)

                torch.cuda.synchronize()
                t_batch_end = time.perf_counter()
                batch_times.append(t_batch_end - t_batch_start)

                # Store results (adjust based on your needs)
                all_outputs.append(outputs)
                all_indices.extend(batch["pair_indexes"].cpu().numpy().tolist())

                # If you need to track which samples these outputs correspond to:
        # indices = sampler.indices[len(all_indices):len(all_indices)+len(outputs)]
        # all_indices.extend(indices)

        # TODO: collect the result and save them without repetition
        # 9. Gather results from all processes
        # First, concatenate all outputs on each process
        # TODO: store the results for resuming!
        output_keys = list(all_outputs[0].keys())
        local_outputs = (
            {
                key: torch.cat(
                    [output[key] for output in all_outputs], dim=0
                ).contiguous()
                for key in ["scores"]
            }
            | {"pair_indexes": all_indices}
            | {
                key: [
                    output[key][i]
                    for output in all_outputs
                    for i in range(len(output[key]))
                ]
                for key in output_keys
                if key not in ["scores"]
            }
        )

        if args.distributed:
            output_keys = output_keys + ["pair_indexes"]

            data_list = all_gather_object_cpu(
                local_outputs,
                tmpdir=args.temp_path + "/tmp_save",
                rank_zero_return_only=False,
                use_system_tmp=False,
            )

            global_outputs = {}
            pair_index_order = [
                output["pair_indexes"][x]
                for output in data_list
                for x in range(len(output["pair_indexes"]))
            ]
            index_mapping = np.zeros(len(dataset_pair), dtype=np.int64)
            for i, pair_index in enumerate(pair_index_order):
                index_mapping[pair_index] = i

            for i, pair_index in enumerate(pair_index_order):
                for key in output_keys:
                    if key == "pair_indexes":
                        continue
                    elif key == "scores":
                        gathered_outputs = [output[key] for output in data_list]
                        gathered_outputs = torch.cat(
                            gathered_outputs, dim=0
                        ).contiguous()
                        global_outputs[key] = gathered_outputs[index_mapping]
                    else:
                        gathered_outputs = [
                            output[key][x]
                            for output in data_list
                            for x in range(len(output[key]))
                        ]
                        global_outputs[key] = [
                            gathered_outputs[index_mapping[i]]
                            for i in range(len(index_mapping))
                        ]

            # global_outputs["images_path"] = dataset_pair.images_path
            # global_outputs["image_list"] = dataset_pair.images_list
            # global_outputs["rotations_gt"] = dataset_pair.rotations_gt_all
            # global_outputs["centers_gt"] = dataset_pair.centers_gt_all
        else:
            global_outputs = local_outputs

        global_outputs["pairs"] = dataset_pair.pairs

        if rank == 0 and save_intermediate_results:
            os.makedirs(args.curr_path, exist_ok=True)
            torch.save(global_outputs, os.path.join(args.curr_path, file_name))
    else:
        print("Loading existing results...")
        global_outputs = torch.load(os.path.join(args.curr_path, file_name))

    twoview_timing = {
        "batch_times": batch_times,
        "num_batches": len(batch_times),
        "total": sum(batch_times) if batch_times else 0.0,
    }
    if rank == 0 and batch_times:
        print(f"[Profiling] Two-view inference: {len(batch_times)} batches, "
              f"total={sum(batch_times):.2f}s, mean={sum(batch_times)/len(batch_times):.3f}s/batch")

    return global_outputs, twoview_timing


def run_star_inference(
    models,
    args,
    dataset,
    world_size,
    rank,
    file_name="star_result.pth",
    save_intermediate_results=True,
    device="cuda",
    dtype=torch.bfloat16,
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
    star_inferece = BatchInferenceStar(
        models[chosen_model], chosen_model, models.get("vggsfm"), device=device, dtype=dtype,
        pointmap_dir=os.path.join(args.curr_path, "pointmap"),
        repredict_verified=getattr(args, "repredict_verified", False),
        repredict_threshold=getattr(args, "repredict_threshold", 0.05),
    )
    if not args.force_load or not os.path.exists(
        os.path.join(args.curr_path, file_name)
    ):
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
        print("Loading existing results...")
        predictions_dict = torch.load(os.path.join(args.curr_path, file_name))

    star_timing = {
        "batch_times": batch_times,
        "forward_times": forward_times,
        "tracking_times": tracking_times,
        "num_batches": len(batch_times),
        "total": sum(batch_times) if batch_times else 0.0,
    }
    if rank == 0 and batch_times:
        print(f"[Profiling] Star inference: {len(batch_times)} batches, "
              f"total={sum(batch_times):.2f}s, mean={sum(batch_times)/len(batch_times):.3f}s/batch, "
              f"forward={sum(forward_times):.2f}s, tracking={sum(tracking_times):.2f}s")

    return predictions_dict, star_timing
