import torch
import networkx as nx
import einops

from gluemap.utils.misc import switch_tensor_order, calculate_index_mappings
from gluemap.math.geometry import unproject, project, bilinear_interpolate_value


def project_point(
    world_points,
    extrinsics,
    intrinsic,
    from_0_to_j=True,
    images_change=None,
    images_shape_ori=None,
    conf=None,
    conf_threshold=1.0,
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
        # # First, transform the points from the other frames to the first frame
        # world_points_j = torch.einsum('b n h w c, b n c d -> b n h w d', world_points - extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3), extrinsics[:, :, :3, :3])

        world_points_j = torch.einsum(
            "b n h w c, b d c -> b n h w d", world_points, extrinsics[:, 0, :3, :3]
        ) + extrinsics[:, :1, :3, 3].expand(-1, N, -1).unsqueeze(2).unsqueeze(3)
        # Project the points to the image plane
        image_points_j = project(
            einops.rearrange(world_points_j, "b n h w d -> (b n) (h w) d"),
            einops.rearrange(
                intrinsic[:, :1].expand(-1, N, -1, -1), "b n d c -> (b n) d c"
            ),
        )

    invalid_i2j = (
        (image_points_j[..., 0] < 0)
        | (image_points_j[..., 1] < 0)
        | (image_points_j[..., 0] >= W)
        | (image_points_j[..., 1] >= H)
    ).reshape(B, N, H, W)

    image_points_j[..., 0] = torch.clamp(image_points_j[..., 0], min=0, max=W)
    image_points_j[..., 1] = torch.clamp(image_points_j[..., 1], min=0, max=H)

    # Transform the points from the other frames to the first frame
    if from_0_to_j:
        # Perform bilinear interpolation to get the 3D points
        # Note that the 3D points are already in the world coordinates so do not need to transform them back
        world_points_j_prime = bilinear_interpolate_value(
            world_points.flatten(0, 1).permute(0, 3, 1, 2), image_points_j
        )
        world_points_j_prime = einops.rearrange(
            world_points_j_prime, "(b n) (h w) d -> b n h w d", b=B, n=N, h=H, w=W
        )

        world_points_i_prime = torch.einsum(
            "b n h w c, b d c -> b n h w d",
            world_points_j_prime,
            extrinsics[:, 0, :3, :3],
        ) + extrinsics[:, :1, :3, 3].expand(-1, N, -1).unsqueeze(2).unsqueeze(3)
        # Project the points to the image plane
        image_points_i = project(
            einops.rearrange(world_points_i_prime, "b n h w d -> (b n) (h w) d"),
            einops.rearrange(
                intrinsic[:, :1].expand(-1, N, -1, -1), "b n d c -> (b n) d c"
            ),
        ).reshape(B, N, H, W, 2)

    else:
        # Perform bilinear interpolation to get the 3D points
        # Note that the 3D points are already in the world coordinates so do not need to transform them back
        world_points_j_prime = bilinear_interpolate_value(
            world_points[:, :1]
            .expand(-1, N, -1, -1, -1)
            .flatten(0, 1)
            .permute(0, 3, 1, 2),
            image_points_j,
        )
        world_points_j_prime = einops.rearrange(
            world_points_j_prime, "(b n) (h w) d -> b n h w d", b=B, n=N, h=H, w=W
        )

        world_points_i_prime = torch.einsum(
            "b n h w c, b n d c -> b n h w d",
            world_points_j_prime,
            extrinsics[:, :, :3, :3],
        ) + extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3)

        # Project the points to the image plane
        image_points_i = project(
            einops.rearrange(world_points_i_prime, "b n h w d -> (b n) (h w) d"),
            einops.rearrange(intrinsic, "b n d c -> (b n) d c"),
        ).reshape(B, N, H, W, 2)

    # Check the different between the projected points and the original grid
    # Set up the grid
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W), torch.arange(0, H), indexing="xy"
    )
    grids = (
        torch.stack((grid_x, grid_y), dim=-1).float().to(world_points.device)
    )  # (H, W, 2)

    # invalid_j2i = (image_points_i[..., 0] < 0) | (image_points_i[..., 1] < 0) | (image_points_i[..., 0] >= W) | (image_points_i[..., 1] >= H)

    image_points_j = image_points_j.reshape(B, N, H, W, 2)  # (B, N, H, W, 2)
    if images_change is not None:
        images_change_temp = images_change.clone().unsqueeze(-2)
        corners_x = torch.stack(
            [torch.zeros_like(images_shape_ori[:, :, 0]), images_shape_ori[:, :, 1]],
            dim=-1,
        )  # (B, N, 2)
        corners_y = torch.stack(
            [torch.zeros_like(images_shape_ori[:, :, 0]), images_shape_ori[:, :, 0]],
            dim=-1,
        )  # (B, N, 2)
        corners_x_new = (
            (corners_x * images_change_temp[..., 0] + images_change_temp[..., 2])
            .unsqueeze(-2)
            .unsqueeze(-2)
            .to(device=world_points.device)
        )
        corners_y_new = (
            (corners_y * images_change_temp[..., 1] + images_change_temp[..., 3])
            .unsqueeze(-2)
            .unsqueeze(-2)
            .to(device=world_points.device)
        )

        # Initial points should be in the range of [0, W] and [0, H]
        invalid_i2j = (
            invalid_i2j
            | (grids[..., 0].unsqueeze(0).unsqueeze(1) < corners_x_new[..., 0])
            | (grids[..., 1].unsqueeze(0).unsqueeze(1) < corners_y_new[..., 0])
            | (grids[..., 0].unsqueeze(0).unsqueeze(1) >= corners_x_new[..., 1])
            | (grids[..., 1].unsqueeze(0).unsqueeze(1) >= corners_y_new[..., 1])
        )

        # Transformed points should be in the range of [0, W] and [0, H]
        invalid_i2j = (
            invalid_i2j
            | (image_points_j[..., 0] < corners_x_new[..., 0])
            | (image_points_j[..., 1] < corners_y_new[..., 0])
            | (image_points_j[..., 0] >= corners_x_new[..., 1])
            | (image_points_j[..., 1] >= corners_y_new[..., 1])
        )

        # # Transformed points should be in the range of [0, W] and [0, H]
        # invalid_j2i = invalid_j2i | (image_points_i[..., 0] < corners_x_new[:, :1, :, :, 0]) | (image_points_i[..., 1] < corners_y_new[:, :1, :, :, 0]) | (image_points_i[..., 0] >= corners_x_new[:, :1, :, :, 1]) | (image_points_i[..., 1] >= corners_y_new[:, :1, :, :, 1])

        valid_number = (
            (corners_x_new[..., 1] - corners_x_new[..., 0])
            * (corners_y_new[..., 1] - corners_y_new[..., 0])
        ).flatten(
            -3, -1
        )  # (B, N)
    else:
        valid_number = H * W

    # Return:
    #   - The two-way projected image point
    #   - The two-way transformed world points (in the frame coordinate)
    #   - The invalid points (in the two-way projection)
    #   - The valid number of points in the image
    return image_points_i, world_points_i_prime, invalid_i2j, valid_number


def verify_by_reprojection(
    world_points,
    extrinsics,
    intrinsic,
    threshold_reproj=4.0,
    # conf=None,
    # conf_threshold=1.0,
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

    # if conf is not None:
    #     conf_subtracted = conf
    # else:
    #     conf_subtracted = None

    B, N, H, W, _ = world_points.shape

    if not images_shape_ori is None:
        assert (
            not images_change is None
        ), "If images_shape_ori is not None, images_change should not be None."
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

    # Calculate the depth
    depths = torch.einsum(
        "b n h w c, b n d c -> b n h w d", world_points, extrinsics[:, :, :3, :3]
    ) + extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3)

    # if conf is not None:
    #     image_points_i, world_points_j_prime, invalid_j2i, valid_number, conf_j2i = (
    #         project_point(
    #             world_points,
    #             extrinsics,
    #             intrinsic,
    #             from_0_to_j=True,
    #             images_change=images_change,
    #             images_shape_ori=images_shape_ori,
    #             conf=conf_subtracted,
    #         )
    #     )
    #     image_points_j, world_points_i_prime, invalid_i2j, _, conf_i2j = project_point(
    #         world_points,
    #         extrinsics,
    #         intrinsic,
    #         from_0_to_j=False,
    #         images_change=images_change,
    #         images_shape_ori=images_shape_ori,
    #         conf=conf_subtracted,
    #     )
    # else:
    image_points_i, world_points_j_prime, invalid_j2i, valid_number = project_point(
        world_points,
        extrinsics,
        intrinsic,
        from_0_to_j=True,
        images_change=images_change,
        images_shape_ori=images_shape_ori,
    )
    image_points_j, world_points_i_prime, invalid_i2j, _ = project_point(
        world_points,
        extrinsics,
        intrinsic,
        from_0_to_j=False,
        images_change=images_change,
        images_shape_ori=images_shape_ori,
    )

    # Check the different between the projected points and the original grid
    # Set up the grid
    grid_x, grid_y = torch.meshgrid(
        torch.arange(0, W), torch.arange(0, H), indexing="xy"
    )
    grids = (
        torch.stack((grid_x, grid_y), dim=-1).float().to(world_points.device)
    )  # (H, W, 2)

    errors_i = torch.norm(image_points_i - grids.unsqueeze(0).unsqueeze(1), dim=-1)
    errors_j = torch.norm(image_points_j - grids.unsqueeze(0).unsqueeze(1), dim=-1)

    # # Count the number of points that are within the threshold
    # # For the visual overlap, we only consider the projection from the first frame to the other frames
    # valid = ((errors_i < threshold_reproj) * (~(invalid_i2j)) * (errors_j < threshold_reproj) * (~(invalid_i2j + invalid_j2i)))
    valid = (errors_i < threshold_reproj) * (~invalid_j2i)
    scores = valid.flatten(-2, -1).sum(dim=-1) / valid_number

    # if conf is not None:
    #     # For points that are projected behind the predicted depth, it should be considered as occuluded, not necessarily inconsistent

    #     # Here, to avoid over emphasizing the depth difference, we only check the ratio if the depth is larger than 1
    #     depths_clamped = torch.clamp(depths, min=1.0)  # (B, N, H, W)

    #     inconsistent_i = (
    #         torch.clamp(
    #             (
    #                 world_points_j_prime[..., -1]
    #                 - depths[:, :1, :, :, -1].expand(-1, N, -1, -1)
    #             )
    #             / (
    #                 (1 / conf_subtracted[:, :1].expand(-1, N, -1, -1) + 1 / conf_j2i)
    #                 * depths_clamped[:, :1, :, :, -1].expand(-1, N, -1, -1)
    #             )
    #             + 1e-6,
    #             min=0,
    #         )
    #         > lambda_conf
    #     ) * (~invalid_j2i)
    #     inconsistent_j = (
    #         torch.clamp(
    #             (world_points_i_prime[..., -1] - depths[..., -1])
    #             / (
    #                 (1 / conf_subtracted + 1 / conf_i2j) * depths_clamped[..., -1]
    #                 + 1e-6
    #             ),
    #             min=0,
    #         )
    #         > lambda_conf
    #     ) * (~invalid_i2j)

    #     # A point is considered inconsistent if it is inconsistent in either direction and
    #     inconsistent = inconsistent_i + inconsistent_j
    #     scores_inconsistent = inconsistent.flatten(-2, -1).sum(dim=-1) / valid_number

    #     inconsistent_pairs = (scores_inconsistent > scores) + (
    #         scores_inconsistent > consistent_threshold
    #     )

    #     if return_seperate_scores:
    #         # errors[(invalid_i2j + invalid_j2i)] = -1
    #         return scores, scores_inconsistent, valid, inconsistent

    #     scores = torch.where(
    #         inconsistent_pairs, -1.0, scores
    #     )  # If there are more inconsistent points than consistent points, set the score to 0
    #     valid = torch.where(
    #         (inconsistent_pairs).unsqueeze(-1).unsqueeze(-1).expand(-1, -1, H, W),
    #         False,
    #         valid,
    #     )  # If there are more inconsistent points than consistent points, set the valid to False

    return scores, valid


def verify_by_reprojection_n2(
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
        # order = order
        world_points, extrinsics, intrinsic = switch_tensor_order(
            [world_points, extrinsics, intrinsic], order
        )

        if images_change is not None:
            images_change, images_shape_ori = switch_tensor_order(
                [images_change, images_shape_ori], order
            )

        # if conf is not None:
        #     conf = switch_tensor_order([conf], order)[0]

        scores, valid_mask = verify_by_reprojection(
            world_points,
            extrinsics,
            intrinsic,
            threshold_reproj,
            # conf,
            # conf_threshold,
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

        # if conf is not None:
        #     conf = switch_tensor_order([conf], order)[0]

        scores_all.append(scores)
        if j == 0:
            valid_mask_0 = valid_mask

    # If image 0 do not see image j but image 0 sees image i and image i sees image j, then we can consider the distance from image 0 to image j as the distance from image 0 to image i multiply the distance from image i to image j.
    # Go through the scores and find the valid pairs
    scores_all = torch.stack(scores_all, dim=1)
    scores_all = torch.maximum(scores_all, scores_all.transpose(1, 2))
    scores_inner = torch.zeros(B, N).to(world_points.device)  # (B, N, N)
    for i in range(B):
        valid_i, valid_j = torch.where(scores_all[i] > 0.0)  # (N, N)

        valid_edges = set(
            [
                (
                    valid_i[j].item(),
                    valid_j[j].item(),
                    -torch.log(scores_all[i, valid_i[j], valid_j[j]]).item(),
                )
                for j in range(len(valid_i))
            ]
        )

        # Construct the graph
        G = nx.Graph()
        G.add_weighted_edges_from(list(valid_edges))

        # Get the shortest distance between every nodes to the center
        lengths = nx.single_source_dijkstra_path_length(G, 0, weight="weight")
        scores_inner[i] = torch.exp(
            -torch.tensor(
                [lengths[j] if j in lengths else float("inf") for j in range(N)],
                device=world_points.device,
            )
        )

    return scores_inner, valid_mask_0


def convert_from_depth_to_world_points(depth, extrinsics, intrinsic):
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

    camera_points = image_rays.reshape(B, N, H, W, 3) * depth  # (B, N, H, W, 3)

    world_points = torch.einsum(
        "b n h w d, b n d c -> b n h w c",
        camera_points - extrinsics[:, :, :3, 3].unsqueeze(2).unsqueeze(3),
        extrinsics[:, :, :3, :3],
    )

    return world_points
