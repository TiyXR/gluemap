from gluemap.datasets.twoview_dataset import BaseTwoViewDataset
from gluemap.datasets.sequential_twoview_dataset import SequentialTwoViewDataset

from gluemap.utils.cli import get_args_parser, parse_args_with_config
from gluemap.utils.gpu import init_distributed
from gluemap.controllers.salad_retrieval import run_preprocessing_pipeline
from gluemap.controllers.gluemap_impl import run_inference_pipeline


def main(args):
    rank, world_size, device, dtype = init_distributed(args)

    args.curr_processed = args.write_path
    args.curr_path = args.write_path

    # Preprocessing: SALAD retrieval
    (_, _), _ = run_preprocessing_pipeline(args, world_size, rank)

    # Create dataset pair
    if "is_sequential" in args and args.is_sequential:
        dataset_pair = SequentialTwoViewDataset(args)
    else:
        dataset_pair = BaseTwoViewDataset(args)

    # Inference pipeline: twoview -> star -> global mapping -> refinement
    run_inference_pipeline(args, dataset_pair, world_size, rank, device, dtype)


if __name__ == "__main__":
    parser = get_args_parser()
    parser.add_argument(
        "--is_sequential",
        action="store_true",
        help="whether the images are sequentially ordered",
    )
    parser.add_argument(
        "--sample_frequency",
        type=int,
        default=1,
        help="frequency to sample images if sequential",
    )
    args = parse_args_with_config(parser)
    main(args)
