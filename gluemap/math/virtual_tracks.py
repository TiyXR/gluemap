import logging
import torch
import einops

from gluemap.math.geometry import project_tracks, unproject, bilinear_interpolate_value

logger = logging.getLogger(__name__)


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

    return project_tracks(camera_points, extrinsics, intrinsic)


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


