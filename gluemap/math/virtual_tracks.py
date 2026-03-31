import torch
import numpy as np
import einops

from gluemap.math.geometry import project, unproject, bilinear_interpolate_value


# Note: the coordinates of camera_points should correspond to extrinsics
def project_virtual_tracks(
    camera_points,
    extrinsics,
    intrinsic,
    angle_threshold=5,
):
    B, N = extrinsics.shape[:2]

    if camera_points.dim() == 3:
        camera_points = camera_points.unsqueeze(1)

    world_points_j = torch.einsum(
        "b h w c, b n d c -> b n h w d", camera_points, extrinsics[:, :, :3, :3]
    ) + extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3)

    image_points_j = project(
        einops.rearrange(
            world_points_j.to(torch.float64), "b n h w d -> (b n) (h w) d"
        ),
        einops.rearrange(intrinsic.to(torch.float64), "b n d c -> (b n) d c"),
    ).to(
        world_points_j.dtype
    )  # (B*N, H*W, 2)

    # Do not consider points outside the image as invalid. Instead, check the depth of the points, it the angle to the principle point is too large, then we consider it as invalid
    world_points_j_normalized = world_points_j / torch.clamp(
        world_points_j.norm(dim=-1, keepdim=True), min=1e-6
    )
    invalid_i2j = (
        world_points_j_normalized[..., 2] < np.sin(np.deg2rad(angle_threshold))
    ).reshape(
        B, N, -1
    )  # (B, N, K)
    invalid_j2i = (
        world_points_j_normalized[..., 2] > -np.sin(np.deg2rad(angle_threshold))
    ).reshape(
        B, N, -1
    )  # (B, N, K)

    # Use these poins as the tracks
    track_virtual = image_points_j.reshape(B, N, -1, 2)  # (B, N, grid_num, 2)

    # Return:
    # track_virtual: (B, N, K, 2)
    # tracks_points: (B, K, 3)
    # valid_mask: (B, N, K)
    # is_negative: (B, N, K)

    return (
        track_virtual,
        camera_points.flatten(1, 2),
        ~(invalid_i2j * invalid_j2i),
        ~invalid_j2i,
    )


def subsample_virtual_tracks(
    track_virtual,
    sampled_points,
    valid_mask,
    is_negative,
    sampled_num=100,
    min_num=3,
):
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
            sample_num = min(min_num, len(valid_idx))
            rand_columns = torch.randperm(len(valid_idx))[:sample_num]

            selected_idx = selected_idx.union(set(valid_idx[rand_columns].tolist()))

        max_selected_num = max(max_selected_num, len(selected_idx))

        selected_idx_all.append(selected_idx)
        if len(invalid_index) > 0:
            print(f"{len(invalid_index)} / {N-1} images are not covered")

    selected_idx_all = [list(selected_idx_all[i]) for i in range(B)]
    max_selected_num = max(max_selected_num, sampled_num)
    track_virtual_selected = []
    for i in range(B):
        # Then, we sample remaining number of points
        remaining_num = max_selected_num - len(selected_idx_all[i])

        if remaining_num > 0:
            weights = valid_mask[i, 1:].float().sum(dim=0)
            # Do not sampled the selected points
            weights[list(selected_idx_all[i])] = 0

            if weights.sum() == 0:
                print("No valid points to sample")
                continue

            selected_idx_all[i] += torch.multinomial(
                weights, remaining_num, replacement=False
            ).tolist()

    selected_idx_list = (
        torch.Tensor([list(selected_idx_all[i]) for i in range(B)])
        .to(track_virtual.device)
        .long()
    )  # (B, K')

    track_virtual_selected = torch.gather(
        track_virtual,
        2,
        selected_idx_list.unsqueeze(-1).unsqueeze(1).expand(-1, N, -1, 2),
    )  # (B, N, K', 2)
    sampled_points_selected = torch.gather(
        sampled_points, 1, selected_idx_list.unsqueeze(-1).expand(-1, -1, 3)
    )  # (B, K', 3)
    valid_mask_selected = torch.gather(
        valid_mask, 2, selected_idx_list.unsqueeze(1).expand(-1, N, -1)
    )  # (B, N, K')

    if not is_negative is None:
        is_negative_selected = torch.gather(
            is_negative, 2, selected_idx_list.unsqueeze(1).expand(-1, N, -1)
        )  # (B, N, K')

    return (
        track_virtual_selected,
        sampled_points_selected,
        valid_mask_selected,
        is_negative_selected,
    )


def calculate_virtual_tracks(depth, extrinsics, intrinsic, valid_mask):

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
        # Use the middle 50 percent of the depth range
        # median_depth.append(torch.percentile(depth_curr, 25), torch.percentile(depth_curr, 75), torch.percentile(depth_curr, 50))
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
    ).reshape(
        B, H // 14, W // 14, 3
    )  # (B, H, W, 3)

    world_points = torch.einsum(
        "b h w d, b d c -> b h w c",
        image_rays - extrinsics[:, 0, :3, 3].unsqueeze(1).unsqueeze(2),
        extrinsics[:, 0, :3, :3],
    )
    # # valid_depth = world_points[:, 0][..., 2][valid_depth_mask]

    ori_depth = torch.where(
        valid_depth_mask, depth[:, 0][..., 0], median_depth
    ).unsqueeze(
        -1
    )  # (B, H, W, 1)
    selected_depth = ori_depth[:, 7::14, 7::14]
    # Add noise to the depth with 10% of the current depth as the noise
    noise = torch.randn_like(selected_depth) * 0.1 * selected_depth
    sampled_depth = selected_depth + noise

    camera_points = world_points * sampled_depth  # (B, H // 4, W // 14, 3)

    return project_virtual_tracks(camera_points, extrinsics, intrinsic)


def extract_cam_points_depth(intrinsics, prediction):
    pr_pts_all = []
    tracks = prediction["track"].to(torch.float32)
    B, N, S = tracks.shape[:3]  # (B, N, S, 2)
    depth = prediction["depth"].to(torch.float32)
    depth_interp = einops.rearrange(
        bilinear_interpolate_value(
            einops.rearrange(depth, "b n h w c -> (b n) c h w"), tracks.flatten(0, 1)
        ),
        "(b n) s c -> b n s c",
        b=B,
        n=N,
    )

    image_rays = unproject(
        tracks.flatten(0, 1), einops.rearrange(intrinsics, "b n d c -> (b n) d c")
    )  # (B*N, S, 3)
    pts_interp_cam = image_rays.reshape(B, N, S, 3) * depth_interp  # (B, N, S, 3)

    conf_interp = einops.rearrange(
        bilinear_interpolate_value(
            einops.rearrange(
                prediction["depth_conf"].unsqueeze(-1), "b n h w c -> (b n) c h w"
            ),
            tracks.flatten(0, 1),
        ),
        "(b n) s c -> b n s c",
        b=B,
        n=N,
    ).squeeze(-1)

    return pts_interp_cam, conf_interp


def extract_world_points_depth(extrinsics, intrinsics, prediction):
    pts_interp_cam, conf_interp = extract_cam_points_depth(intrinsics, prediction)

    pts_interp = torch.einsum(
        "b n s d, b n d c -> b n s c",
        pts_interp_cam - extrinsics[:, :, :3, 3].unsqueeze(2),
        extrinsics[:, :, :3, :3],
    )

    return pts_interp, conf_interp


def convert_world_points_to_cam_points(extrinsics, cam_points):
    pts_world = torch.einsum(
        "b n s d, b n c d -> b n s c",
        cam_points,
        extrinsics[:, :, :3, :3],
    ) + extrinsics[:, :, :3, 3].unsqueeze(2)

    return pts_world
