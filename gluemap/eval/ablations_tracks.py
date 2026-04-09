import logging
import os
import time
import torch

logger = logging.getLogger(__name__)

from gluemap.controllers.twoview_inference import run_twoview_inference
from gluemap.controllers.pipeline_wrapper import run_star_inference
from gluemap.controllers.gluemap_impl import run_postprocessing_pipeline
from gluemap.controllers.results_collection import generate_dataset_from_outputs


def run_track_ablation_pipeline(
    args, dataset_pair, world_size, rank, device, dtype, pairs=None, device_id="0",
    models=None,
):
    """
    Run the inference pipeline with track-type ablation for bundle adjustment.

    Runs twoview + star inference once, then calls postprocessing (global mapping +
    refinement) once per track_mode. Each mode produces a separate output directory.

    Track modes are combinations of:
    - S: SIFT tracks (from database_sift.db)
    - P: Prior/real multi-view tracks (from star predictions)
    - V: Virtual point tracks (monocular depth estimates)

    Supported modes: SPV, SP, SV, PV, S, P

    Args:
        args: Argument namespace (must include track_modes list)
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

    track_modes = getattr(args, "track_modes", ["SPV"])

    # Ensure output directory exists
    os.makedirs(args.curr_path, exist_ok=True)

    # Step 1: Two-view inference (once for all modes)
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
    dataset = generate_dataset_from_outputs(
        dataset_pair, global_outputs, args, device=device, dtype=dtype
    )
    timing["dataset_generation"] = time.perf_counter() - t0

    # Step 3: Star inference (once for all modes)
    predictions_dict, star_timing = run_star_inference(
        args,
        dataset,
        world_size,
        rank,
        file_name="star_result.pth",
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

    # Step 4: Run postprocessing for each track mode (rank 0 only)
    if rank == 0:
        import copy

        pred_dirs = {}
        mode_timings = {}

        for mode in track_modes:
            logger.info(f"{'#'*60}")
            logger.info(f"# Track ablation mode: {mode}")
            logger.info(f"{'#'*60}")

            # Set track mode and output suffix on args
            args.track_mode = mode
            args.output_suffix = f"_{mode}"

            # Deep copy predictions_dict so postprocessing doesn't mutate shared state
            predictions_dict_copy = copy.deepcopy(predictions_dict)

            t0 = time.perf_counter()
            pred_dir, postproc_timing = run_postprocessing_pipeline(
                args,
                predictions_dict_copy,
                dataset_pair,
                dataset,
                pairs=pairs,
                device_id=device_id,
            )
            mode_timings[mode] = postproc_timing
            mode_timings[mode]["wall_time"] = time.perf_counter() - t0
            pred_dirs[mode] = pred_dir

        timing["track_ablation_modes"] = mode_timings
        timing["total_pipeline"] = time.perf_counter() - t_pipeline_start

        # Print summary
        logger.info(f"[Track Ablation Profiling] Modes: {', '.join(track_modes)}")
        logger.info(f"  Two-view (load+infer): model_load={twoview_timing.get('model_loading', 0):.2f}s, "
              f"inference={twoview_timing['total']:.2f}s")
        logger.info(f"  Dataset generation:    {timing['dataset_generation']:.2f}s")
        logger.info(f"  Star (load+infer):     model_load={star_timing.get('model_loading', 0):.2f}s, "
              f"inference={star_timing['total']:.2f}s")
        for mode in track_modes:
            mt = mode_timings[mode]
            logger.info(f"  Mode {mode}: wall_time={mt['wall_time']:.2f}s, total={mt['total']:.2f}s")
        logger.info(f"  Total pipeline:        {timing['total_pipeline']:.2f}s")

        # Save per-dataset timing
        timing_path = os.path.join(args.curr_path, "pipeline_timing.pth")
        torch.save(timing, timing_path)

        return pred_dirs, timing

    timing["total_pipeline"] = time.perf_counter() - t_pipeline_start
    return None, timing
