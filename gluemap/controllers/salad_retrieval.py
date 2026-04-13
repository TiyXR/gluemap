import logging
import os
import time

import torch
from tqdm import tqdm

from gluemap.datasets.utils import get_image_list
from gluemap.utils.gpu import synchronize
from gluemap.utils.load_fn import load_and_preprocess_images
from gluemap.utils.model_loader import load_models

logger = logging.getLogger(__name__)


class SaladRetrievalPipeline:
    def __init__(
        self,
        args,
        world_size,
        rank,
        file_name="salad_descriptors.pt",
        device="cuda",
        dtype=torch.bfloat16,
        models=None,
    ):
        self.args = args
        self.world_size = world_size
        self.rank = rank
        self.file_name = file_name
        self.device = device
        self.dtype = dtype

        if models is not None and "salad" in models:
            self.model = models["salad"]
            self.device = next(self.model.parameters()).device
        else:
            loaded, self.device = load_models(args, keys={"salad"})
            self.model = loaded["salad"]

    def _maybe_delete_retrieval_cache(self):
        if getattr(self.args, "rerun_from", None) != "retrieval":
            return
        base_path = self.args.curr_processed or self.args.curr_path
        path = os.path.join(base_path, self.file_name)
        if os.path.exists(path):
            os.remove(path)
            logger.info(f"[rerun_from=retrieval] Deleted {path}")

    @torch.no_grad()
    def _compute_descriptors(self):
        batch_size = self.args.retrieval_batch_size
        images_list = get_image_list(self.args.images_path)

        descriptors = []
        N = len(images_list)
        for i in tqdm(range(0, N, batch_size)):
            num_img = min(N - i, batch_size)
            images, _, _ = load_and_preprocess_images(
                images_list[i : i + num_img],
                image_size=322,  # use the fixed 322 size for SALAD retrieval
                patch_size=14,
                force_square=True,
            )
            output = self.model(images.to(self.device)).cpu()
            descriptors.append(output)

        descriptors = torch.cat(descriptors)
        return descriptors

    def _run_retrieval(self, file_name=None):
        if file_name is None:
            file_name = self.file_name

        base_path = self.args.curr_path
        if self.args.curr_processed:
            base_path = self.args.curr_processed
        if not self.args.force_load or not os.path.exists(
            os.path.join(base_path, file_name)
        ):
            if self.rank == 0:
                logger.info("Computing SALAD descriptors...")
                descriptors = self._compute_descriptors()

                os.makedirs(base_path, exist_ok=True)
                torch.save(descriptors, os.path.join(base_path, file_name))

            if self.args.distributed:
                synchronize()

    def run(self):
        self._maybe_delete_retrieval_cache()

        t0 = time.perf_counter()
        # Model already loaded in __init__
        t1 = time.perf_counter()
        self._run_retrieval()
        t2 = time.perf_counter()

        timing = {
            "model_loading": t1 - t0,
            "salad_retrieval": t2 - t1,
            "total": t2 - t0,
        }
        if self.rank == 0:
            logger.info(
                f"[Profiling] Preprocessing: model_loading={t1 - t0:.2f}s, "
                f"salad_retrieval={t2 - t1:.2f}s, total={t2 - t0:.2f}s"
            )

        return ({"salad": self.model}, self.device), timing

    def run_multi(self, datasets):
        t0 = time.perf_counter()
        # Model already loaded in __init__
        t1 = time.perf_counter()

        images_path_root = self.args.images_path
        retrieval_times = {}
        for dataset in datasets:
            self.args.images_path = f"{images_path_root}/{dataset}"
            self.args.curr_path = f"{self.args.write_path}/{dataset}"
            self._maybe_delete_retrieval_cache()
            t_ds_start = time.perf_counter()
            self._run_retrieval()
            t_ds_end = time.perf_counter()
            retrieval_times[dataset] = t_ds_end - t_ds_start

        # Restore original paths
        self.args.images_path = images_path_root
        self.args.curr_processed = self.args.write_path
        self.args.curr_path = self.args.write_path

        t2 = time.perf_counter()
        timing = {
            "model_loading": t1 - t0,
            "salad_retrieval": sum(retrieval_times.values()),
            "salad_retrieval_per_dataset": retrieval_times,
            "total": t2 - t0,
        }
        if self.rank == 0:
            logger.info(
                f"[Profiling] Preprocessing multi: model_loading={t1 - t0:.2f}s, "
                f"salad_retrieval={sum(retrieval_times.values()):.2f}s, total={t2 - t0:.2f}s"
            )

        return ({"salad": self.model}, self.device), timing


def run_preprocessing_pipeline(
    args, world_size, rank, file_name="salad_descriptors.pt", models=None
):
    pipeline = SaladRetrievalPipeline(
        args, world_size, rank, file_name, models=models
    )
    return pipeline.run()


def run_preprocessing_pipeline_multi(
    args,
    world_size,
    rank,
    datasets,
    file_name="salad_descriptors.pt",
    models=None,
):
    pipeline = SaladRetrievalPipeline(
        args, world_size, rank, file_name, models=models
    )
    return pipeline.run_multi(datasets)
