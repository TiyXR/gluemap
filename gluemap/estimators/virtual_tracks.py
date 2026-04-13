import logging

import numpy as np
import torch

from gluemap.math.geometry import project_tracks, restore_identity

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
        indexes = range(len(predictions_dict["indexes"]))

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

    def _collect_extrinsics_intrinsics_centered(
        self,
        global_rotations,
        global_centers,
        global_intrinsics,
        intrinsics_mapping,
        indexes,
    ):
        rotations = torch.from_numpy(
            np.stack([global_rotations[idx] for idx in indexes])
        ).unsqueeze(0)
        translations = torch.from_numpy(
            np.stack(
                [
                    -global_rotations[idx] @ global_centers[idx].reshape(3, 1)
                    for idx in indexes
                ]
            )
        ).unsqueeze(0)

        extrinsics = torch.cat(
            [rotations, translations], dim=-1
        )  # (B, N, 3, 4)
        intrinsics = (
            torch.stack(
                [global_intrinsics[intrinsics_mapping[idx]] for idx in indexes],
                dim=1,
            )
            .cpu()
            .to(torch.float64)
        )

        extrinsics = restore_identity(extrinsics)
        return extrinsics, intrinsics

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
            extrinsics = (
                predictions_dict["extrinsics"][idx].cpu().to(torch.float64)
            )
            (
                predictions_dict["tracks_virtual"][idx],
                _,
                predictions_dict["valid_virtual"][idx],
                predictions_dict["isnegative_virtual"][idx],
            ) = project_tracks(
                predictions_dict["points3d_virtual"][idx].to(torch.float64),
                extrinsics,
                intrinsics,
                angle_threshold=self.angle_threshold,
            )

    def _subsample_virtual_tracks(self, predictions_dict, indexes):
        for idx in indexes:
            track_virtual = predictions_dict["tracks_virtual"][idx]
            sampled_points = predictions_dict["points3d_virtual"][idx]
            valid_mask = predictions_dict["valid_virtual"][idx]
            is_negative = predictions_dict["isnegative_virtual"][idx]

            B, N = track_virtual.shape[:2]  # (B, N, K, 2)
            # First, select min_num valid tracks for each cameras
            selected_idx_all = []
            max_selected_num = 0
            for i in range(B):
                invalid_index = []
                selected_idx = set()
                for view_id in range(1, N):
                    valid_idx = torch.where(valid_mask[i, view_id])[0]
                    if len(valid_idx) == 0:
                        invalid_index.append(view_id)
                        continue
                    sample_num = min(self.min_num, len(valid_idx))
                    rand_columns = torch.randperm(len(valid_idx))[:sample_num]

                    selected_idx = selected_idx.union(
                        set(valid_idx[rand_columns].tolist())
                    )

                max_selected_num = max(max_selected_num, len(selected_idx))

                selected_idx_all.append(selected_idx)
                if len(invalid_index) > 0:
                    logger.info(
                        f"{len(invalid_index)} / {N - 1} images are not covered"
                    )

            selected_idx_all = [list(selected_idx_all[i]) for i in range(B)]
            max_selected_num = max(max_selected_num, self.num_desired_tracks)
            for i in range(B):
                # Then, we sample remaining number of points
                remaining_num = max_selected_num - len(selected_idx_all[i])

                if remaining_num > 0:
                    weights = valid_mask[i, 1:].float().sum(dim=0)
                    # Do not sampled the selected points
                    weights[list(selected_idx_all[i])] = 0

                    if weights.sum() == 0:
                        logger.warning("No valid points to sample")
                        continue

                    selected_idx_all[i] += torch.multinomial(
                        weights, remaining_num, replacement=False
                    ).tolist()

            selected_idx_list = (
                torch.Tensor([list(selected_idx_all[i]) for i in range(B)])
                .to(track_virtual.device)
                .long()
            )  # (B, K')

            predictions_dict["tracks_virtual"][idx] = torch.gather(
                track_virtual,
                2,
                selected_idx_list.unsqueeze(-1)
                .unsqueeze(1)
                .expand(-1, N, -1, 2),
            )  # (B, N, K', 2)
            predictions_dict["points3d_virtual"][idx] = torch.gather(
                sampled_points,
                1,
                selected_idx_list.unsqueeze(-1).expand(-1, -1, 3),
            )  # (B, K', 3)
            predictions_dict["valid_virtual"][idx] = torch.gather(
                valid_mask, 2, selected_idx_list.unsqueeze(1).expand(-1, N, -1)
            )  # (B, N, K')

            if is_negative is not None:
                predictions_dict["isnegative_virtual"][idx] = torch.gather(
                    is_negative,
                    2,
                    selected_idx_list.unsqueeze(1).expand(-1, N, -1),
                )  # (B, N, K')

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
            extrinsics, intrinsics = (
                self._collect_extrinsics_intrinsics_centered(
                    global_rotations,
                    global_centers,
                    global_intrinsics,
                    intrinsics_mapping,
                    predictions_dict["indexes"][idx],
                )
            )
            num_virtual_points = predictions_dict["points3d_virtual"][
                idx
            ].shape[-2]
            num_tracks_chosen = int(num_virtual_points * self.update_ratio)

            (
                predictions_dict["tracks_virtual"][idx][
                    ..., :num_tracks_chosen, :
                ],
                _,
                predictions_dict["valid_virtual"][idx][..., :num_tracks_chosen],
                predictions_dict["isnegative_virtual"][idx][
                    ..., :num_tracks_chosen
                ],
            ) = project_tracks(
                predictions_dict["points3d_virtual"][idx][
                    :, :num_tracks_chosen
                ].to(torch.float64),
                extrinsics,
                intrinsics,
                angle_threshold=self.angle_threshold,
            )

            if "pose_inconsistent" in predictions_dict:
                invalid_idx = predictions_dict["pose_inconsistent"][idx]
                tracks_insufficient = (
                    predictions_dict["scores"][idx].sum(dim=-1)[0]
                    < num_virtual_points
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
                ) = project_tracks(
                    predictions_dict["points3d_virtual"][idx][
                        :, : 2 * num_tracks_chosen
                    ].to(torch.float64),
                    extrinsics[:, invalid_idx, : 2 * num_tracks_chosen],
                    intrinsics[:, invalid_idx, : 2 * num_tracks_chosen],
                    angle_threshold=self.angle_threshold,
                )
