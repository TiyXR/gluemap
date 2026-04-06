import logging
import os
import time
import torch
from gluemap.utils.model_loader import load_models
from gluemap.utils.colmap_io import write_to_colmap_format

logger = logging.getLogger(__name__)

from gluemap.controllers.pipeline_wrapper import (
    run_twoview_inference,
    run_star_inference,
)
from gluemap.controllers.run_inference import run_postprocessing_pipeline
from gluemap.controllers.run_results_collection import generate_dataset_from_outputs


def run_ablation_inference_pipeline(
    args, dataset_pair, world_size, rank, device, dtype, pairs=None, device_id="0",
    models=None,
):
    """
    Run the inference pipeline with ablation options for filtering studies.

    Supports two ablation flags:
    - skip_doppelgangers: Skip DG model, set all pair scores to 1.0
    - skip_back_and_forth: Set all pose_scores to 1.0 after star inference,
      disabling the two-way consistency filtering in GlobalGluer

    Args:
        args: Argument namespace (must include skip_doppelgangers, skip_back_and_forth)
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

    skip_dg = getattr(args, "skip_doppelgangers", False)
    skip_bnf = getattr(args, "skip_back_and_forth", False)

    # Ensure output directory exists (normally created by twoview save path)
    os.makedirs(args.curr_path, exist_ok=True)

    # Build ablation-specific cache filename to avoid collisions with normal pipeline.
    # When only skip_back_and_forth is set (no skip_doppelgangers), the twoview and
    # star inputs are identical to the normal pipeline, so reuse star_result.pth.
    if skip_dg:
        star_file_name = "star_result_nodg.pth"
    else:
        star_file_name = "star_result.pth"

    # Step 1: Two-view inference (or skip with ablation)
    t0 = time.perf_counter()
    if skip_dg:
        logger.info("[Ablation] Skipping Doppelgangers: setting all pair scores to 1.0")
        global_outputs = {
            "scores": torch.ones(len(dataset_pair)),
            "pairs": dataset_pair.pairs,
        }
        twoview_timing = {"batch_times": [], "num_batches": 0, "total": 0.0}
    else:
        if models is not None and "dg" in models:
            dg_model = models["dg"]
        else:
            loaded, device = load_models(args, keys=set({"dg"}))
            dg_model = loaded["dg"]
        t_model_load = time.perf_counter() - t0

        global_outputs, twoview_timing = run_twoview_inference(
            dg_model,
            args,
            dataset_pair,
            world_size,
            rank,
            file_name="twoview_result.pth",
            save_intermediate_results=True,
            device=device,
        )
        twoview_timing["model_loading"] = t_model_load

    timing["twoview_inference"] = twoview_timing

    # Step 2: Generate dataset from outputs
    t0 = time.perf_counter()
    dataset = generate_dataset_from_outputs(
        dataset_pair, global_outputs, args, device=device, dtype=dtype
    )
    timing["dataset_generation"] = time.perf_counter() - t0

    # Step 3: Star inference
    t0 = time.perf_counter()
    if models is not None and "pi3" in models:
        star_models = models
    else:
        model_keys = {"pi3"} if getattr(args, "disable_tracking", False) else {"pi3", "vggsfm"}
        star_models, device = load_models(args, keys=model_keys)
    t_model_load = time.perf_counter() - t0

    predictions_dict, star_timing = run_star_inference(
        star_models,
        args,
        dataset,
        world_size,
        rank,
        file_name=star_file_name,
        save_intermediate_results=True,
        device=device,
    )
    star_timing["model_loading"] = t_model_load
    timing["star_inference"] = star_timing

    # Ablation: override pose_scores to disable back-and-forth filtering
    if skip_bnf:
        logger.info("[Ablation] Skipping back-and-forth filtering: setting all pose_scores to 1.0")
        for idx in range(len(predictions_dict["pose_scores"])):
            predictions_dict["pose_scores"][idx] = torch.ones_like(
                predictions_dict["pose_scores"][idx]
            )

    # Move predictions_dict tensors to CPU to free GPU memory before postprocessing
    for key, value in predictions_dict.items():
        if isinstance(value, torch.Tensor):
            predictions_dict[key] = value.cpu()
        elif isinstance(value, list):
            predictions_dict[key] = [
                v.cpu() if isinstance(v, torch.Tensor) else v for v in value
            ]
    torch.cuda.empty_cache()

    # Set output suffix so coarse results go to coarse_nodg/ or coarse_novo/
    suffix_parts = []
    if skip_dg:
        suffix_parts.append("nodg")
    if skip_bnf:
        suffix_parts.append("novo")
    args.output_suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""

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
        ablation_flags = []
        if skip_dg:
            ablation_flags.append("skip_doppelgangers")
        if skip_bnf:
            ablation_flags.append("skip_back_and_forth")
        logger.info(f"[Ablation Profiling] Active ablations: {', '.join(ablation_flags)}")
        logger.info(f"  Two-view (load+infer): model_load={twoview_timing.get('model_loading', 0):.2f}s, "
              f"inference={twoview_timing['total']:.2f}s")
        logger.info(f"  Dataset generation:    {timing['dataset_generation']:.2f}s")
        logger.info(f"  Star (load+infer):     model_load={star_timing.get('model_loading', 0):.2f}s, "
              f"inference={star_timing['total']:.2f}s")
        logger.info(f"  Postprocessing:        {postproc_timing['total']:.2f}s")
        logger.info(f"  Total pipeline:        {timing['total_pipeline']:.2f}s")

        # Save per-dataset timing
        timing_path = os.path.join(args.curr_path, "pipeline_timing.pth")
        torch.save(timing, timing_path)

        return pred_dir, timing

    timing["total_pipeline"] = time.perf_counter() - t_pipeline_start
    return None, timing
