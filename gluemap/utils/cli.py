import argparse

import yaml


def get_args_parser():
    parser = argparse.ArgumentParser("Distributed Demo Pipeline", add_help=True)

    parser.add_argument(
        "--chosen_model",
        default="pi3",
        choices=["pi3", "pi3x", "vggt", "map_anything"],
        help="which model to use for multi-view pose estimation",
    )

    parser.add_argument(
        "--path_feedforward",
        default="",
        type=str,
        help="path to the chosen feedforward model checkpoint",
    )
    parser.add_argument(
        "--path_retrieval", default="", type=str, help="path to the retrieval model"
    )
    parser.add_argument(
        "--path_tracker", default="", type=str, help="path to the tracker model"
    )
    parser.add_argument(
        "--path_dg", default="", type=str, help="path to the doppelganger model"
    )

    # IO
    parser.add_argument(
        "--write_path",
        default="results/",
        type=str,
        help="directory to write the results",
    )
    parser.add_argument(
        "--images_path",
        default=None,
        type=str,
        help="path to the images folder",
    )

    parser.add_argument(
        "--num_track_per_img",
        default=1024,
        type=int,
        help="number of tracks per image to track",
    )

    parser.add_argument(
        "--max_num_tracks",
        default=None,
        type=int,
        help="maximum number of tracks before bundle adjustment (None = unlimited)",
    )

    parser.add_argument(
        "--camera_model",
        default="SIMPLE_PINHOLE",
        type=str,
        help="camera model to use",
    )

    parser.add_argument(
        "--share_intrinsics",
        action="store_true",
        help="whether to share intrinsics among images with the same shape",
    )

    parser.add_argument(
        "--num_neighbors",
        default=100,
        type=int,
        help="number of neighbors to establish",
    )

    parser.add_argument(
        "--num_neighbors_sequential",
        default=30,
        type=int,
        help="number of neighbors to establish",
    )

    parser.add_argument(
        "--temp_path",
        default="./tmp",
        type=str,
        help="temp path, for collecting results from multiple GPUs",
    )
    parser.add_argument(
        "--save_result", default=True, type=bool, help="force to discard the results"
    )

    parser.add_argument(
        "--valid_pose_threshold",
        default=0.05,
        type=float,
        help="mininum threshold for valid pose. 0.05 means that if larger than 5 percent of points are covisible, it is valid",
    )

    parser.add_argument(
        "--num_workers", default=4, type=int, help="number of workers for data loading"
    )
    parser.add_argument(
        "--batch_size", default=30, type=int, help="batch size for two view inference"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--valid_dg_threshold",
        default=0.8,
        type=float,
        help="threshold for dg matching",
    )

    parser.add_argument(
        "--retrieval_batch_size", default=30, type=int, help="batch size for retrieval"
    )

    parser.add_argument(
        "--force_load", action="store_true", help="force load the precomputed results"
    )

    parser.add_argument(
        "--rerun_from",
        default=None,
        type=str,
        choices=["retrieval", "twoview", "star"],
        help="force rerun from a specific pipeline stage (deletes cached files from that stage onward)",
    )

    parser.add_argument(
        "--skip_refinement",
        action="store_true",
        help="whether to skip the refinement step",
    )

    parser.add_argument(
        "--coarse_only",
        action="store_true",
        help="only output coarse results, skip all refinement steps",
    )

    parser.add_argument(
        "--disable_tracking",
        action="store_true",
        help="skip VGGSfM tracking in star inference",
    )

    parser.add_argument(
        "--gt_intrinsics_path",
        default=None,
        type=str,
        help="path to a COLMAP reconstruction directory with GT intrinsics",
    )

    parser.add_argument(
        "--use_gt_intrinsics",
        action="store_true",
        help="use GT intrinsics from the ground truth reconstruction (requires gt_path in config)",
    )

    # Ablation flags
    parser.add_argument(
        "--skip_doppelgangers",
        action="store_true",
        help="Ablation: skip DG model, treat all pairs as valid (score=1.0)",
    )
    parser.add_argument(
        "--skip_back_and_forth",
        action="store_true",
        help="Ablation: set all pose_scores to 1.0 after star inference, disabling consistency filtering",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="path to a YAML configuration file (CLI arguments override YAML values)",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="force the entire pipeline to run on CPU (ignores available GPUs)",
    )

    return parser


def parse_args_with_config(parser):
    """Parse arguments with optional YAML config support.

    If --config is provided, YAML values are used as defaults;
    CLI arguments take precedence over YAML values.
    """
    args, _ = parser.parse_known_args()

    if args.config:
        with open(args.config) as f:
            yaml_config = yaml.safe_load(f) or {}
        parser.set_defaults(**yaml_config)

    args = parser.parse_args()

    if args.images_path is None:
        parser.error("the following arguments are required: --images_path")

    return args
