import logging
import os
import time

import numpy as np
import torch
from scipy.special import softmax
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from gluemap.utils.gpu import all_gather_object_cpu
from gluemap.utils.model_loader import load_models

logger = logging.getLogger(__name__)


class BatchInferenceDG:
    def __init__(self, model, device="cuda", dtype=torch.bfloat16):

        self.model = model
        self.device = device
        self.dtype = dtype

    def main(self, batch):
        images = batch["images"].to(self.device)

        view1 = {
            "img": images[:, 0],
            "instance": [i for i in range(images.shape[0])],
        }
        view2 = {
            "img": images[:, 1],
            "instance": [i for i in range(images.shape[0])],
        }

        res1, res2, pred1, pred2 = self.model(view1, view2)

        if isinstance(pred1, list):
            pred1 = torch.stack(pred1, dim=0)
        else:
            pred1 = pred1

        if isinstance(pred2, list):
            pred2 = torch.stack(pred2, dim=0)
        else:
            pred2 = pred2

        score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
        score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
        vote_0 = (score_s1[:, 0] > score_s1[:, 1]).astype(int) + (
            score_s2[:, 0] > score_s2[:, 1]
        ).astype(int)
        vote_1 = (score_s1[:, 1] > score_s1[:, 0]).astype(int) + (
            score_s2[:, 1] > score_s2[:, 0]
        ).astype(int)
        index_max = vote_1 > vote_0
        index_min = vote_1 < vote_0
        index_equal = vote_1 == vote_0
        score = np.zeros_like(score_s1[:, 0])
        score[index_max] = np.max(
            (score_s1[index_max, 1], score_s2[index_max, 1]), axis=0
        )
        score[index_min] = np.min(
            (score_s1[index_min, 1], score_s2[index_min, 1]), axis=0
        )
        score[index_equal] = np.mean(
            (score_s1[index_equal, 1], score_s2[index_equal, 1]), axis=0
        )

        result_dict = {
            "scores": torch.from_numpy(score),
        }

        return result_dict


class TwoViewInferencePipeline:
    """Pipeline object for two-view (Doppelgangers) inference.

    Follows the same pattern as ``SaladRetrievalPipeline``: stable config is
    stored as instance attributes; per-dataset inputs are passed to ``run()``.
    """

    def __init__(
        self,
        args,
        world_size,
        rank,
        file_name="twoview_result.pth",
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
        self.model = None
        self._owns_model = False

    def _make_dataloader(self, dataset_pair):
        if self.args.distributed:
            sampler = DistributedSampler(
                dataset_pair,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
        else:
            sampler = torch.utils.data.SequentialSampler(dataset_pair)

        return torch.utils.data.DataLoader(
            dataset_pair,
            sampler=sampler,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

    def _load_model(self):
        if self.model is not None:
            return self.model
        if self.preloaded_models is not None and "dg" in self.preloaded_models:
            self.model = self.preloaded_models["dg"]
            self.device = next(self.model.parameters()).device
        else:
            loaded, self.device = load_models(self.args, keys={"dg"})
            self.model = loaded["dg"]
            self._owns_model = True
        return self.model

    def _release_model(self):
        """Free the DG model if we own it (i.e. not caller-supplied)."""
        if not self._owns_model or self.model is None:
            return
        del self.model
        self.model = None
        self._owns_model = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_inference(self, data_loader):
        all_outputs = []
        all_indices = []
        batch_times = []

        model = self._load_model()
        t_model_load = time.perf_counter()
        two_view_inference = BatchInferenceDG(
            model, device=self.device, dtype=self.dtype
        )

        with torch.no_grad():
            for batch in tqdm(
                data_loader,
                desc=f"Inference (Rank {self.rank})",
                disable=self.rank != 0,
            ):
                torch.cuda.synchronize()
                t_batch_start = time.perf_counter()

                outputs = two_view_inference.main(batch)

                torch.cuda.synchronize()
                t_batch_end = time.perf_counter()
                batch_times.append(t_batch_end - t_batch_start)

                all_outputs.append(outputs)
                all_indices.extend(batch["pair_indexes"].cpu().numpy().tolist())

        return all_outputs, all_indices, batch_times, t_model_load

    def _gather_outputs(self, all_outputs, all_indices, dataset_pair):
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

        if self.args.distributed:
            output_keys = output_keys + ["pair_indexes"]

            data_list = all_gather_object_cpu(
                local_outputs,
                tmpdir=self.args.temp_path + "/tmp_save",
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
        else:
            global_outputs = local_outputs

        global_outputs["pairs"] = dataset_pair.pairs
        return global_outputs

    def run(self, dataset_pair, save_intermediate_results=True):
        """Run two-view inference on the given dataset.

        Returns:
            Tuple of (global_outputs, twoview_timing).
        """
        args = self.args
        file_name = self.file_name

        data_loader = self._make_dataloader(dataset_pair)

        batch_times = []
        t_model_load = 0.0

        cache_path = os.path.join(args.curr_path, file_name)
        if getattr(args, "rerun_from", None) in (
            "retrieval",
            "twoview",
        ) and os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"[rerun_from={args.rerun_from}] Deleted {cache_path}")

        if not (args.force_load and os.path.exists(cache_path)):
            t0_load = time.perf_counter()
            all_outputs, all_indices, batch_times, _ = self._run_inference(
                data_loader
            )
            t_model_load = time.perf_counter() - t0_load - sum(batch_times)

            global_outputs = self._gather_outputs(
                all_outputs, all_indices, dataset_pair
            )

            if self.rank == 0 and save_intermediate_results:
                os.makedirs(args.curr_path, exist_ok=True)
                torch.save(
                    global_outputs, os.path.join(args.curr_path, file_name)
                )
        else:
            logger.info("Loading existing results...")
            global_outputs = torch.load(
                os.path.join(args.curr_path, file_name), weights_only=False
            )

        twoview_timing = {
            "batch_times": batch_times,
            "num_batches": len(batch_times),
            "total": sum(batch_times) if batch_times else 0.0,
            "model_loading": t_model_load,
        }
        if self.rank == 0 and batch_times:
            logger.info(
                f"[Profiling] Two-view inference: {len(batch_times)} batches, "
                f"total={sum(batch_times):.2f}s, "
                f"mean={sum(batch_times) / len(batch_times):.3f}s/batch"
            )

        self._release_model()
        return global_outputs, twoview_timing


def run_twoview_inference(
    args,
    dataset_pair,
    world_size,
    rank,
    file_name="twoview_result.pth",
    save_intermediate_results=True,
    device="cuda",
    dtype=torch.bfloat16,
    preloaded_models=None,
):
    """Module-level wrapper that instantiates TwoViewInferencePipeline and runs it."""
    pipeline = TwoViewInferencePipeline(
        args,
        world_size,
        rank,
        file_name=file_name,
        device=device,
        dtype=dtype,
        preloaded_models=preloaded_models,
    )
    return pipeline.run(
        dataset_pair, save_intermediate_results=save_intermediate_results
    )
