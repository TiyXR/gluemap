import einops
import networkx as nx
import torch

from gluemap.math.geometry import (
    bilinear_interpolate_value,
    project,
    project_tracks,
    unproject,
)
from gluemap.math.scaling import rescale_tracks_single


def calculate_index_mappings(query_index, S, device=None):
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    return [
        torch.index_select(tensor, dim, order) if tensor is not None else None
        for tensor in tensors
    ]


class CovisibilityExtraction:
    def __init__(self, check_consistency=False, include_track=True):
        self.check_consistency = check_consistency
        self.include_track = include_track

    def main(self, predictions, indexes, images_change):
        extrinsics = predictions["extrinsics"]
        intrinsics = predictions["intrinsics"]

        depth_transformed = self._convert_from_depth_to_world_points(
            predictions["depth"], extrinsics, intrinsics
        )

        if self.check_consistency:
            scores, valid_mask = self._verify_by_reprojection_n2(
                depth_transformed,
                extrinsics,
                intrinsics,
                conf=predictions["depth_conf"],
            )
        else:
            scores, valid_mask = self._verify_by_reprojection_n2(
                depth_transformed, extrinsics, intrinsics
            )

        tracks_virtual, points3d_virtual, isnegative_virtual, valid_virtual = (
            self._calculate_virtual_tracks(
                predictions["depth"], extrinsics, intrinsics, valid_mask
            )
        )

        if "track" in predictions and self.include_track:
            cam_points, cam_points_conf = self._extract_cam_points_depth(
                intrinsics, predictions
            )

            for i in range(indexes.shape[0]):
                for j, idx_inner in enumerate(indexes[i].tolist()):
                    tracks_virtual[i, j] = rescale_tracks_single(
                        tracks_virtual[i, j], images_change[i][j]
                    )

            return (
                extrinsics.cpu(),
                intrinsics.cpu(),
                scores.cpu(),
                tracks_virtual.cpu(),
                points3d_virtual.cpu(),
                valid_virtual.cpu(),
                cam_points.cpu(),
                cam_points_conf.cpu(),
                depth_transformed.cpu(),
            )
        else:
            return (
                extrinsics.cpu(),
                intrinsics.cpu(),
                scores.cpu(),
                tracks_virtual.cpu(),
                points3d_virtual.cpu(),
                valid_virtual.cpu(),
                None,
                None,
                depth_transformed.cpu(),
            )

    def _calculate_virtual_tracks(
        self, depth, extrinsics, intrinsic, valid_mask
    ):
        # Use the valid to obtain a distribution of the depth
        # Assuming the first image has the identity matrix
        B, N, H, W, _ = depth.shape

        valid_depth_mask = valid_mask[:, 1:].float().sum(dim=1) > 0  # (B, H, W)
        median_depth = []
        for i in range(B):
            depth_curr = depth[i, 0, :, :, 0][valid_depth_mask[i]]

            if depth_curr.numel() == 0:
                median_depth.append(torch.ones(1)[0].to(depth.device) * 1000)
                valid_depth_mask[i] = valid_depth_mask[i] * 0.0
                continue
            median_depth.append(torch.quantile(depth_curr, 0.5))

            valid_depth_mask[i] = (
                valid_depth_mask[i]
                * (depth[i, 0, :, :, 0] > torch.quantile(depth_curr, 0.25))
                * (depth[i, 0, :, :, 0] < torch.quantile(depth_curr, 0.75))
            )

        median_depth = (
            torch.stack(median_depth, dim=0).unsqueeze(-1).unsqueeze(-1)
        )  # (B, 1, 1)

        # Then, sample the depth on a coarser grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, W // 14), torch.arange(0, H // 14), indexing="xy"
        )
        grids_coarse = (
            torch.stack((grid_x, grid_y), dim=-1)
            .float()
            .to(depth.device)
            .unsqueeze(0)
            .expand(B, -1, -1, -1)
            * 14
            + 7
        )  # (H, W, 2)

        # For each grid point, check whether the depth is valid. If not, set the mean to be median
        image_rays = unproject(
            einops.rearrange(grids_coarse, "b h w d -> b (h w) d"),
            einops.rearrange(intrinsic[:, 0], "b d c -> b d c"),
        ).reshape(B, H // 14, W // 14, 3)  # (B, H, W, 3)

        world_points = torch.einsum(
            "b h w d, b d c -> b h w c",
            image_rays - extrinsics[:, 0, :3, 3].unsqueeze(1).unsqueeze(2),
            extrinsics[:, 0, :3, :3],
        )

        ori_depth = torch.where(
            valid_depth_mask, depth[:, 0][..., 0], median_depth
        ).unsqueeze(-1)  # (B, H, W, 1)
        selected_depth = ori_depth[:, 7::14, 7::14]
        # Add noise to the depth with 10% of the current depth as the noise
        noise = torch.randn_like(selected_depth) * 0.1 * selected_depth
        sampled_depth = selected_depth + noise

        camera_points = world_points * sampled_depth  # (B, H // 4, W // 14, 3)

        return project_tracks(camera_points, extrinsics, intrinsic)

    def _extract_cam_points_depth(self, intrinsics, prediction):
        tracks = prediction["track"].to(torch.float32)
        B, N, S = tracks.shape[:3]  # (B, N, S, 2)
        depth = prediction["depth"].to(torch.float32)
        depth_interp = einops.rearrange(
            bilinear_interpolate_value(
                einops.rearrange(depth, "b n h w c -> (b n) c h w"),
                tracks.flatten(0, 1),
            ),
            "(b n) s c -> b n s c",
            b=B,
            n=N,
        )

        image_rays = unproject(
            tracks.flatten(0, 1),
            einops.rearrange(intrinsics, "b n d c -> (b n) d c"),
        )  # (B*N, S, 3)
        pts_interp_cam = (
            image_rays.reshape(B, N, S, 3) * depth_interp
        )  # (B, N, S, 3)

        conf_interp = einops.rearrange(
            bilinear_interpolate_value(
                einops.rearrange(
                    prediction["depth_conf"].unsqueeze(-1),
                    "b n h w c -> (b n) c h w",
                ),
                tracks.flatten(0, 1),
            ),
            "(b n) s c -> b n s c",
            b=B,
            n=N,
        ).squeeze(-1)

        return pts_interp_cam, conf_interp

    def _project_point(
        self,
        world_points,
        extrinsics,
        intrinsic,
        from_0_to_j=True,
        images_change=None,
        images_shape_ori=None,
    ):
        # from_0_to_j: if True, project from frame 0 to frame j, otherwise project from frame j to frame 0
        B, N, H, W, _ = world_points.shape
        if from_0_to_j:
            # First, transform the points from the first frame to the other frames
            world_points_j = torch.einsum(
                "b h w c, b n d c -> b n h w d",
                world_points[:, 0],
                extrinsics[:, :, :3, :3],
            ) + extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3)

            # Project the points to the image plane
            image_points_j = project(
                einops.rearrange(world_points_j, "b n h w d -> (b n) (h w) d"),
                einops.rearrange(intrinsic, "b n d c -> (b n) d c"),
            )  # (B*N, H*W, 2)
        else:
            world_points_j = torch.einsum(
                "b n h w c, b d c -> b n h w d",
                world_points,
                extrinsics[:, 0, :3, :3],
            ) + extrinsics[:, :1, :3, 3].expand(-1, N, -1).unsqueeze(
                2
            ).unsqueeze(3)
            # Project the points to the image plane
            image_points_j = project(
                einops.rearrange(world_points_j, "b n h w d -> (b n) (h w) d"),
                einops.rearrange(
                    intrinsic[:, :1].expand(-1, N, -1, -1),
                    "b n d c -> (b n) d c",
                ),
            )

        invalid_i2j = (
            (image_points_j[..., 0] < 0)
            | (image_points_j[..., 1] < 0)
            | (image_points_j[..., 0] >= W)
            | (image_points_j[..., 1] >= H)
        ).reshape(B, N, H, W)

        image_points_j[..., 0] = torch.clamp(
            image_points_j[..., 0], min=0, max=W
        )
        image_points_j[..., 1] = torch.clamp(
            image_points_j[..., 1], min=0, max=H
        )

        # Transform the points from the other frames to the first frame
        if from_0_to_j:
            world_points_j_prime = bilinear_interpolate_value(
                world_points.flatten(0, 1).permute(0, 3, 1, 2), image_points_j
            )
            world_points_j_prime = einops.rearrange(
                world_points_j_prime,
                "(b n) (h w) d -> b n h w d",
                b=B,
                n=N,
                h=H,
                w=W,
            )

            world_points_i_prime = torch.einsum(
                "b n h w c, b d c -> b n h w d",
                world_points_j_prime,
                extrinsics[:, 0, :3, :3],
            ) + extrinsics[:, :1, :3, 3].expand(-1, N, -1).unsqueeze(
                2
            ).unsqueeze(3)
            # Project the points to the image plane
            image_points_i = project(
                einops.rearrange(
                    world_points_i_prime, "b n h w d -> (b n) (h w) d"
                ),
                einops.rearrange(
                    intrinsic[:, :1].expand(-1, N, -1, -1),
                    "b n d c -> (b n) d c",
                ),
            ).reshape(B, N, H, W, 2)

        else:
            world_points_j_prime = bilinear_interpolate_value(
                world_points[:, :1]
                .expand(-1, N, -1, -1, -1)
                .flatten(0, 1)
                .permute(0, 3, 1, 2),
                image_points_j,
            )
            world_points_j_prime = einops.rearrange(
                world_points_j_prime,
                "(b n) (h w) d -> b n h w d",
                b=B,
                n=N,
                h=H,
                w=W,
            )

            world_points_i_prime = torch.einsum(
                "b n h w c, b n d c -> b n h w d",
                world_points_j_prime,
                extrinsics[:, :, :3, :3],
            ) + extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3)

            # Project the points to the image plane
            image_points_i = project(
                einops.rearrange(
                    world_points_i_prime, "b n h w d -> (b n) (h w) d"
                ),
                einops.rearrange(intrinsic, "b n d c -> (b n) d c"),
            ).reshape(B, N, H, W, 2)

        # Check the different between the projected points and the original grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, W), torch.arange(0, H), indexing="xy"
        )
        grids = (
            torch.stack((grid_x, grid_y), dim=-1)
            .float()
            .to(world_points.device)
        )  # (H, W, 2)

        image_points_j = image_points_j.reshape(
            B, N, H, W, 2
        )  # (B, N, H, W, 2)
        if images_change is not None:
            images_change_temp = images_change.clone().unsqueeze(-2)
            corners_x = torch.stack(
                [
                    torch.zeros_like(images_shape_ori[:, :, 0]),
                    images_shape_ori[:, :, 1],
                ],
                dim=-1,
            )  # (B, N, 2)
            corners_y = torch.stack(
                [
                    torch.zeros_like(images_shape_ori[:, :, 0]),
                    images_shape_ori[:, :, 0],
                ],
                dim=-1,
            )  # (B, N, 2)
            corners_x_new = (
                (
                    corners_x * images_change_temp[..., 0]
                    + images_change_temp[..., 2]
                )
                .unsqueeze(-2)
                .unsqueeze(-2)
                .to(device=world_points.device)
            )
            corners_y_new = (
                (
                    corners_y * images_change_temp[..., 1]
                    + images_change_temp[..., 3]
                )
                .unsqueeze(-2)
                .unsqueeze(-2)
                .to(device=world_points.device)
            )

            # Initial points should be in the range of [0, W] and [0, H]
            invalid_i2j = (
                invalid_i2j
                | (
                    grids[..., 0].unsqueeze(0).unsqueeze(1)
                    < corners_x_new[..., 0]
                )
                | (
                    grids[..., 1].unsqueeze(0).unsqueeze(1)
                    < corners_y_new[..., 0]
                )
                | (
                    grids[..., 0].unsqueeze(0).unsqueeze(1)
                    >= corners_x_new[..., 1]
                )
                | (
                    grids[..., 1].unsqueeze(0).unsqueeze(1)
                    >= corners_y_new[..., 1]
                )
            )

            # Transformed points should be in the range of [0, W] and [0, H]
            invalid_i2j = (
                invalid_i2j
                | (image_points_j[..., 0] < corners_x_new[..., 0])
                | (image_points_j[..., 1] < corners_y_new[..., 0])
                | (image_points_j[..., 0] >= corners_x_new[..., 1])
                | (image_points_j[..., 1] >= corners_y_new[..., 1])
            )

            valid_number = (
                (corners_x_new[..., 1] - corners_x_new[..., 0])
                * (corners_y_new[..., 1] - corners_y_new[..., 0])
            ).flatten(-3, -1)  # (B, N)
        else:
            valid_number = H * W

        return image_points_i, world_points_i_prime, invalid_i2j, valid_number

    def _verify_by_reprojection(
        self,
        world_points,
        extrinsics,
        intrinsic,
        threshold_reproj=4.0,
        consistent_threshold=0.1,
        lambda_conf=0.05,
        return_seperate_scores=False,
        images_change=None,
        images_shape_ori=None,
    ):
        """
        Args:
            world_points (torch.Tensor): (B, N, H, W, 3)
            extrinsics (torch.Tensor): (B, N, 4, 4)
            intrinsic (torch.Tensor): (B, N, 3, 3)
        """

        B, N, H, W, _ = world_points.shape

        if images_shape_ori is not None:
            assert images_change is not None, (
                "If images_shape_ori is not None, images_change should not be None."
            )
            assert images_change.shape == (
                B,
                N,
                4,
            ), "images_change should be of shape (B, N, 4) if provided."
            assert images_shape_ori.shape == (
                B,
                N,
                2,
            ), "images_shape_ori should be of shape (B, N, 2) if provided."

        image_points_i, world_points_j_prime, invalid_j2i, valid_number = (
            self._project_point(
                world_points,
                extrinsics,
                intrinsic,
                from_0_to_j=True,
                images_change=images_change,
                images_shape_ori=images_shape_ori,
            )
        )
        image_points_j, world_points_i_prime, invalid_i2j, _ = (
            self._project_point(
                world_points,
                extrinsics,
                intrinsic,
                from_0_to_j=False,
                images_change=images_change,
                images_shape_ori=images_shape_ori,
            )
        )

        # Check the different between the projected points and the original grid
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, W), torch.arange(0, H), indexing="xy"
        )
        grids = (
            torch.stack((grid_x, grid_y), dim=-1)
            .float()
            .to(world_points.device)
        )  # (H, W, 2)

        errors_i = torch.norm(
            image_points_i - grids.unsqueeze(0).unsqueeze(1), dim=-1
        )

        valid = (errors_i < threshold_reproj) * (~invalid_j2i)
        scores = valid.flatten(-2, -1).sum(dim=-1) / valid_number

        return scores, valid

    def _verify_by_reprojection_n2(
        self,
        world_points,
        extrinsics,
        intrinsic,
        threshold_reproj=4.0,
        consistent_threshold=0.1,
        lambda_conf=0.05,
        images_change=None,
        images_shape_ori=None,
    ):

        B, N, H, W, _ = world_points.shape
        scores_all = []
        valid_mask_0 = None
        for j in range(N):
            # swap the order of the indexes
            order = calculate_index_mappings(j, N).to(world_points.device)
            world_points, extrinsics, intrinsic = switch_tensor_order(
                [world_points, extrinsics, intrinsic], order
            )

            if images_change is not None:
                images_change, images_shape_ori = switch_tensor_order(
                    [images_change, images_shape_ori], order
                )

            scores, valid_mask = self._verify_by_reprojection(
                world_points,
                extrinsics,
                intrinsic,
                threshold_reproj,
                consistent_threshold,
                lambda_conf,
                False,
                images_change,
                images_shape_ori,
            )

            # swap back the order of the indexes
            world_points, extrinsics, intrinsic, scores = switch_tensor_order(
                [world_points, extrinsics, intrinsic, scores], order
            )

            if images_change is not None:
                images_change, images_shape_ori = switch_tensor_order(
                    [images_change, images_shape_ori], order
                )

            scores_all.append(scores)
            if j == 0:
                valid_mask_0 = valid_mask

        scores_all = torch.stack(scores_all, dim=1)
        scores_all = torch.maximum(scores_all, scores_all.transpose(1, 2))
        scores_inner = torch.zeros(B, N).to(world_points.device)
        for i in range(B):
            valid_i, valid_j = torch.where(scores_all[i] > 0.0)

            valid_edges = set(
                [
                    (
                        valid_i[j].item(),
                        valid_j[j].item(),
                        -torch.log(
                            scores_all[i, valid_i[j], valid_j[j]]
                        ).item(),
                    )
                    for j in range(len(valid_i))
                ]
            )

            # Construct the graph
            G = nx.Graph()
            G.add_weighted_edges_from(list(valid_edges))

            # Get the shortest distance between every nodes to the center
            lengths = nx.single_source_dijkstra_path_length(
                G, 0, weight="weight"
            )
            scores_inner[i] = torch.exp(
                -torch.tensor(
                    [
                        lengths[j] if j in lengths else float("inf")
                        for j in range(N)
                    ],
                    device=world_points.device,
                )
            )

        return scores_inner, valid_mask_0

    def _convert_from_depth_to_world_points(self, depth, extrinsics, intrinsic):
        B, N, H, W, _ = depth.shape

        # Establish world points from depth
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, W), torch.arange(0, H), indexing="xy"
        )
        grids = (
            torch.stack((grid_x, grid_y), dim=-1)
            .float()
            .to(depth.device)
            .unsqueeze(0)
            .unsqueeze(1)
            .expand(B, N, -1, -1, -1)
        )  # (B, N, H, W, 2)

        image_rays = unproject(
            einops.rearrange(grids, "b n h w d -> (b n) (h w) d"),
            einops.rearrange(intrinsic, "b n d c -> (b n) d c"),
        )  # (B*N, H*W, 3)

        camera_points = (
            image_rays.reshape(B, N, H, W, 3) * depth
        )  # (B, N, H, W, 3)

        world_points = torch.einsum(
            "b n h w d, b n d c -> b n h w c",
            camera_points - extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3),
            extrinsics[:, :, :3, :3],
        )

        return world_points
