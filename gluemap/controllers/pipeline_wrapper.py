import logging
import os

logger = logging.getLogger(__name__)


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


