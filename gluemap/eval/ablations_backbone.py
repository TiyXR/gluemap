import logging
import os
import time
import torch

logger = logging.getLogger(__name__)

from gluemap.controllers.twoview_inference import run_twoview_inference
from gluemap.controllers.star_inference import run_star_inference
from gluemap.controllers.gluemap_impl import run_postprocessing_pipeline
from gluemap.controllers.star_collection import run_star_collection


def run_backbone_ablation_pipeline(
    args, dataset_pair, world_size, rank, device, dtype, pairs=None, device_id="0",
    models=None,
):
    """
    Run the inference pipeline for backbone ablation studies.

    Uses args.chosen_model to select the backbone (pi3, vggt, map_anything).
    Stores intermediate and final results with backbone-specific names to avoid
    collisions when running multiple backbones on the same scene directory.

    - pi3 (default): no suffix (coarse/, gluemap_aba/, star_result.pth)
    - others: suffix added (coarse_{backbone}/, gluemap_aba_{backbone}/, star_result_{backbone}.pth)

    Args:
        args: Argument namespace (must include chosen_model)
        dataset_pair: The dataset pair object
        world_size: Number of distributed processes
        rank: Current process rank
        device: Torch device
        dtype: Torch dtype
        pairs: Optional pairs override for refinement
        device_id: GPU device ID for GLOMAP
        models: Optional pre-loaded models dict

    Returns:
        Tuple of (pred_dir, timing_dict) or (None, timing_dict)
    """
    timing = {}
    t_pipeline_start = time.perf_counter()

    backbone = getattr(args, "chosen_model", "pi3")

    # pi3 is the baseline — no suffix for backward compatibility
    if backbone == "pi3":
        args.output_suffix = ""
        star_file_name = "star_result.pth"
    else:
        args.output_suffix = f"_{backbone}"
        star_file_name = f"star_result_{backbone}.pth"

    # Ensure output directory exists
    os.makedirs(args.curr_path, exist_ok=True)

    logger.info(f"[Backbone Ablation] Running with backbone: {backbone}")
    logger.info(f"  Output suffix: '{args.output_suffix}'")
    logger.info(f"  Star cache file: {star_file_name}")

    # Step 1: Two-view inference (shared across backbones — DG is backbone-independent)
    global_outputs, twoview_timing = run_twoview_inference(
        args,
        dataset_pair,
        world_size,
        rank,
        file_name="twoview_result.pth",
        save_intermediate_results=True,
        device=device,
        preloaded_models=models,
    )
    timing["twoview_inference"] = twoview_timing

    # Step 2: Generate dataset from outputs
    t0 = time.perf_counter()
    dataset = run_star_collection(dataset_pair, global_outputs, args)
    timing["dataset_generation"] = time.perf_counter() - t0

    # Step 3: Star inference with chosen backbone
    predictions_dict, star_timing = run_star_inference(
        args,
        dataset,
        world_size,
        rank,
        file_name=star_file_name,
        save_intermediate_results=True,
        device=device,
        preloaded_models=models,
    )
    timing["star_inference"] = star_timing

    # Move predictions_dict tensors to CPU to free GPU memory before postprocessing
    for key, value in predictions_dict.items():
        if isinstance(value, torch.Tensor):
            predictions_dict[key] = value.cpu()
        elif isinstance(value, list):
            predictions_dict[key] = [
                v.cpu() if isinstance(v, torch.Tensor) else v for v in value
            ]
    torch.cuda.empty_cache()

    # Steps 4-5: Global mapping and refinement (rank 0 only)
    if rank == 0:
        t0 = time.perf_counter()
        pred_dir, postproc_timing = run_postprocessing_pipeline(
            args,
            predictions_dict,
            dataset_pair,
            dataset,
            pairs=pairs,
            device_id=device_id,
        )
        timing["postprocessing"] = postproc_timing
        timing["total_pipeline"] = time.perf_counter() - t_pipeline_start

        # Print summary
        logger.info(f"[Backbone Ablation Profiling] Backbone: {backbone}")
        logger.info(f"  Two-view (load+infer): model_load={twoview_timing.get('model_loading', 0):.2f}s, "
              f"inference={twoview_timing['total']:.2f}s")
        logger.info(f"  Dataset generation:    {timing['dataset_generation']:.2f}s")
        logger.info(f"  Star (load+infer):     model_load={star_timing.get('model_loading', 0):.2f}s, "
              f"inference={star_timing['total']:.2f}s")
        logger.info(f"  Postprocessing:        {postproc_timing['total']:.2f}s")
        logger.info(f"  Total pipeline:        {timing['total_pipeline']:.2f}s")

        # Save per-dataset timing
        timing_path = os.path.join(args.curr_path, f"pipeline_timing{args.output_suffix}.pth")
        torch.save(timing, timing_path)

        return pred_dir, timing

    timing["total_pipeline"] = time.perf_counter() - t_pipeline_start
    return None, timing
