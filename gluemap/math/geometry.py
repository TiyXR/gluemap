from scipy.spatial.transform import Rotation
import numpy as np
import torch
import torch.nn.functional as F
import einops

# from e2esfm.models.utils import unproject


def rotation_matrix_to_quaternion(R, w_first=True):
    if isinstance(R, torch.Tensor):
        R = R.cpu().numpy()

    if w_first:
        return Rotation.from_matrix(R).as_quat()[[3, 0, 1, 2]]
    else:
        return Rotation.from_matrix(R).as_quat()


def quaternion_to_rotation_matrix(q, w_first=True):
    if isinstance(q, torch.Tensor):
        q = q.cpu().numpy()

    if w_first:
        return Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    else:
        return Rotation.from_quat(q).as_matrix()


# def undistort_tracks(tracks, indexes, intrinsics):
#     # For each image, undistort the tracks
#     tracks_undistored = {}

#     rays = []
#     for i, idx in enumerate(indexes):
#         rays.append(unproject(tracks[:, i], intrinsics[idx]))

#     rays = torch.stack(rays, dim=1)

#     return rays / torch.clamp(torch.norm(rays, dim=-1, keepdim=True), min=1e-5)


# def get_tracks_dict_indexes(tracks_dict):
#     """
#     Args:
#         tracks_dict (dict): Dictionary containing the tracks
#     Returns:
#         indexes (list): List of indexes of the tracks
#     """
#     if isinstance(tracks_dict["indexes"], dict):
#         indexes = list(tracks_dict["indexes"].keys())
#     elif isinstance(tracks_dict["indexes"], list):
#         indexes = list(range(len(tracks_dict["indexes"])))


#     return indexes


def bilinear_interpolate_value(value_map, coords, align_corners=False):
    # Function of sampling N points on each value map with coordinate specified by coords
    # value_map:    B x C x H x W
    # coords:       B x N x 2
    # Return:       B x N x C
    assert value_map.dim() == 4, f"bad dimension for the value map"

    sizes = value_map.shape[2:]

    if align_corners:
        coords = coords * torch.tensor(
            [2 / max(size - 1, 1) for size in reversed(sizes)],
            device=coords.device,
        )
    else:
        coords = coords * torch.tensor(
            [2 / size for size in reversed(sizes)], device=coords.device
        )

    coords -= 1

    B, N, _ = coords.shape
    sampled = F.grid_sample(
        value_map, coords.unsqueeze(-2), align_corners=align_corners
    ).permute(0, 2, 3, 1)

    return sampled.view(B, N, -1)


def project(rays, intrinsics):
    # rays:         B x N x 3
    # intrinsics:   B x 3 x 3
    assert rays.dim() == 3, f"bad dimension for the coordinates"

    uv_homogeneous = torch.einsum("b t j, b n j -> b n t", intrinsics, rays)
    scale = torch.where(uv_homogeneous[..., 2:] < 0, -1, 1)

    uv = (
        uv_homogeneous[..., :2]
        / torch.clamp(torch.abs(uv_homogeneous[..., 2:]), min=1e-3)
        * scale
    )

    return uv


def unproject(uvs, intrinsics):
    # uvs:          B x N x 2
    # intrinsics:   B x 3 x 3
    uv_homogeneous = torch.cat([uvs, torch.ones_like(uvs)[..., :1]], dim=-1)

    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]

    intrinsics_inv = torch.zeros_like(intrinsics)
    intrinsics_inv[:, 0, 0] = 1.0 / fx
    intrinsics_inv[:, 1, 1] = 1.0 / fy
    intrinsics_inv[:, 0, 2] = -cx / fx
    intrinsics_inv[:, 1, 2] = -cy / fy
    intrinsics_inv[:, 2, 2] = 1.0

    rays = torch.einsum("b t j, b n j -> b n t", intrinsics_inv, uv_homogeneous)

    return rays


# Note: the coordinates of camera_points should correspond to extrinsics
def project_tracks(
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
