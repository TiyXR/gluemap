import logging
import os

from gluemap.datasets.multi_sequence_twoview_dataset import MultiSequencePairs

from gluemap.utils.cli import get_args_parser, parse_args_with_config
from gluemap.utils.gpu_utils import init_distributed
from gluemap.controllers.salad_retrieval import run_preprocessing_pipeline_multi
from gluemap.controllers.gluemap_impl import run_inference_pipeline

logger = logging.getLogger(__name__)


def main(args):
    rank, world_size, device, dtype = init_distributed(args)

    # Prepare dataset list
    datasets = [x for x in sorted(os.listdir(args.images_path)) if x.startswith("ios")]
    logger.info(f"datasets to process: {datasets}")

    # Preprocessing: SALAD retrieval for multiple datasets
    (_, _), _ = run_preprocessing_pipeline_multi(args, world_size, rank, datasets)

    # Create dataset pair
    dataset_pair = MultiSequencePairs(args, datasets)

    # Inference pipeline: twoview -> star -> global mapping -> refinement
    run_inference_pipeline(
        args, dataset_pair, world_size, rank, device, dtype,
        pairs=dataset_pair.pairs,  # Use dataset_pair.pairs for lamar
    )


if __name__ == "__main__":
    parser = get_args_parser()
    args = parse_args_with_config(parser)
    main(args)
