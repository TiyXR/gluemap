import logging
import time
from gluemap.utils.model_loader import load_models
from gluemap.controllers.pipeline_wrapper import run_salad_retrieval, invalidate_cache_from

logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(
    args, world_size, rank, file_name="salad_descriptors.pt", models=None
):
    """
    Run preprocessing: SALAD retrieval for a single dataset.

    Returns:
        Tuple of (models, device) and preprocessing timing dict
    """
    invalidate_cache_from(args, getattr(args, "rerun_from", None))

    t0 = time.perf_counter()
    if models is not None and "salad" in models:
        device = next(models["salad"].parameters()).device
    else:
        models, device = load_models(args, keys=set({"salad"}))
    t1 = time.perf_counter()
    run_salad_retrieval(models["salad"], args, world_size, rank, file_name)
    t2 = time.perf_counter()

    timing = {
        "model_loading": t1 - t0,
        "salad_retrieval": t2 - t1,
        "total": t2 - t0,
    }
    if rank == 0:
        logger.info(f"[Profiling] Preprocessing: model_loading={t1-t0:.2f}s, "
              f"salad_retrieval={t2-t1:.2f}s, total={t2-t0:.2f}s")

    return (models, device), timing


def run_preprocessing_pipeline_multi(
    args, world_size, rank, datasets, file_name="salad_descriptors.pt", models=None
):
    """
    Run preprocessing: SALAD retrieval for multiple datasets (e.g., for lamar).

    Args:
        datasets: List of dataset folder names to process.

    Returns:
        Tuple of (models, device) and preprocessing timing dict
    """
    t0 = time.perf_counter()
    images_path_root = args.images_path
    if models is not None and "salad" in models:
        device = next(models["salad"].parameters()).device
    else:
        models, device = load_models(args, keys=set({"salad"}))
    t1 = time.perf_counter()

    retrieval_times = {}
    for dataset in datasets:
        args.images_path = f"{images_path_root}/{dataset}"
        args.curr_path = f"{args.write_path}/{dataset}"
        invalidate_cache_from(args, getattr(args, "rerun_from", None))
        t_ds_start = time.perf_counter()
        run_salad_retrieval(models["salad"], args, world_size, rank, file_name)
        t_ds_end = time.perf_counter()
        retrieval_times[dataset] = t_ds_end - t_ds_start

    # Restore original paths
    args.images_path = images_path_root
    args.curr_processed = args.write_path
    args.curr_path = args.write_path

    t2 = time.perf_counter()
    timing = {
        "model_loading": t1 - t0,
        "salad_retrieval": sum(retrieval_times.values()),
        "salad_retrieval_per_dataset": retrieval_times,
        "total": t2 - t0,
    }
    if rank == 0:
        logger.info(f"[Profiling] Preprocessing multi: model_loading={t1-t0:.2f}s, "
              f"salad_retrieval={sum(retrieval_times.values()):.2f}s, total={t2-t0:.2f}s")

    return (models, device), timing
