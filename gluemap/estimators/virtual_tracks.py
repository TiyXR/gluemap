import torch

from gluemap.math.virtual_tracks import (
    project_virtual_tracks,
    subsample_virtual_tracks,
)
from gluemap.utils.misc import (
    get_tracks_dict_indexes,
    collect_extrinsics_intrinsics_centered,
)

import logging
logger = logging.getLogger(__name__)


class VirtualTrackPreparation:
    def __init__(
        self,
        angle_threshold=5.0,
        num_desired_tracks=100,
        min_num=5,
        update_ratio=0.1,
    ):
        self.angle_threshold = angle_threshold
        self.num_desired_tracks = num_desired_tracks
        self.min_num = min_num
        self.update_ratio = update_ratio

    def main(
        self,
        predictions_dict,
        global_intrinsics,
        intrinsics_mapping,
        global_rotations,
        global_centers,
    ):
        """Prepare virtual tracks for bundle adjustment.

        Combines three steps:
        1. Project virtual 3D points to 2D using local extrinsics and global intrinsics.
        2. Subsample virtual tracks to a manageable number.
        3. Re-project a subset of virtual tracks using global poses, with extra
           updates for edges marked as pose-inconsistent.
        """
        indexes = get_tracks_dict_indexes(predictions_dict)

        self._update_virtual_tracks(
            predictions_dict, global_intrinsics, intrinsics_mapping, indexes
        )
        self._subsample_virtual_tracks(predictions_dict, indexes)
        self._update_virtual_tracks_global(
            predictions_dict,
            global_intrinsics,
            intrinsics_mapping,
            global_rotations,
            global_centers,
            indexes,
        )

        # Log valid virtual point count
        total_valid_virtual_points = 0
        for idx in indexes:
            total_valid_virtual_points += (
                predictions_dict["valid_virtual"][idx].sum().item()
            )
        logger.info(
            f"Total number of valid virtual points after global positioning: {total_valid_virtual_points}"
        )

    def _update_virtual_tracks(
        self, predictions_dict, global_intrinsics, intrinsics_mapping, indexes
    ):
        if "isnegative_virtual" not in predictions_dict:
            predictions_dict["isnegative_virtual"] = {}

        for idx in indexes:
            intrinsics = (
                torch.stack(
                    [
                        global_intrinsics[intrinsics_mapping[idx_inner]]
                        for idx_inner in predictions_dict["indexes"][idx]
                    ],
                    dim=1,
                )
                .cpu()
                .to(torch.float64)
            )
            extrinsics = predictions_dict["extrinsics"][idx].cpu().to(torch.float64)
            (
                predictions_dict["tracks_virtual"][idx],
                _,
                predictions_dict["valid_virtual"][idx],
                predictions_dict["isnegative_virtual"][idx],
            ) = project_virtual_tracks(
                predictions_dict["points3d_virtual"][idx].to(torch.float64),
                extrinsics,
                intrinsics,
                angle_threshold=self.angle_threshold,
            )

    def _subsample_virtual_tracks(self, predictions_dict, indexes):
        for idx in indexes:
            (
                predictions_dict["tracks_virtual"][idx],
                predictions_dict["points3d_virtual"][idx],
                predictions_dict["valid_virtual"][idx],
                predictions_dict["isnegative_virtual"][idx],
            ) = subsample_virtual_tracks(
                predictions_dict["tracks_virtual"][idx],
                predictions_dict["points3d_virtual"][idx],
                predictions_dict["valid_virtual"][idx],
                predictions_dict["isnegative_virtual"][idx],
                sampled_num=self.num_desired_tracks,
                min_num=self.min_num,
            )

    def _update_virtual_tracks_global(
        self,
        predictions_dict,
        global_intrinsics,
        intrinsics_mapping,
        global_rotations,
        global_centers,
        indexes,
    ):
        if "isnegative_virtual" not in predictions_dict:
            predictions_dict["isnegative_virtual"] = {}

        for idx in indexes:
            extrinsics, intrinsics = collect_extrinsics_intrinsics_centered(
                global_rotations,
                global_centers,
                global_intrinsics,
                intrinsics_mapping,
                predictions_dict["indexes"][idx],
            )
            num_virtual_points = predictions_dict["points3d_virtual"][idx].shape[-2]
            num_tracks_chosen = int(num_virtual_points * self.update_ratio)

            (
                predictions_dict["tracks_virtual"][idx][..., :num_tracks_chosen, :],
                _,
                predictions_dict["valid_virtual"][idx][..., :num_tracks_chosen],
                predictions_dict["isnegative_virtual"][idx][..., :num_tracks_chosen],
            ) = project_virtual_tracks(
                predictions_dict["points3d_virtual"][idx][:, :num_tracks_chosen].to(
                    torch.float64
                ),
                extrinsics,
                intrinsics,
                angle_threshold=self.angle_threshold,
            )

            if "pose_inconsistent" in predictions_dict:
                invalid_idx = predictions_dict["pose_inconsistent"][idx]
                tracks_insufficient = (
                    predictions_dict["scores"][idx].sum(dim=-1)[0] < num_virtual_points
                )
                if not invalid_idx.any():
                    continue
                predictions_dict["valid_virtual"][idx][
                    :, invalid_idx, num_tracks_chosen:
                ] = 0

                invalid_idx = invalid_idx * tracks_insufficient
                if not invalid_idx.any():
                    continue

                (
                    predictions_dict["tracks_virtual"][idx][
                        :, invalid_idx, : 2 * num_tracks_chosen, :
                    ],
                    _,
                    predictions_dict["valid_virtual"][idx][
                        :, invalid_idx, : 2 * num_tracks_chosen
                    ],
                    predictions_dict["isnegative_virtual"][idx][
                        :, invalid_idx, : 2 * num_tracks_chosen
                    ],
                ) = project_virtual_tracks(
                    predictions_dict["points3d_virtual"][idx][
                        :, : 2 * num_tracks_chosen
                    ].to(torch.float64),
                    extrinsics[:, invalid_idx, : 2 * num_tracks_chosen],
                    intrinsics[:, invalid_idx, : 2 * num_tracks_chosen],
                    angle_threshold=self.angle_threshold,
                )
