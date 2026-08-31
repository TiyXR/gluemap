import hashlib

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


def _calculate_index_mappings(
    query_index: int,
    S: int,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build an index permutation that swaps slot 0 with ``query_index``.

    Args:
        query_index: Position to move to slot 0.
        S: Total number of slots.
        device: Optional torch device to move the result onto.

    Returns:
        Long tensor of shape ``(S,)`` defining the swap permutation.
    """
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def _switch_tensor_order(
    tensors: list[torch.Tensor | None],
    order: torch.Tensor,
    dim: int = 1,
) -> list[torch.Tensor | None]:
    """
    Apply an index permutation along a given axis to a list of tensors.

    Args:
        tensors: Tensors to permute. ``None`` entries are passed through
            unchanged.
        order: Long index tensor produced by :func:`_calculate_index_mappings`.
        dim: Axis along which to apply the permutation.

    Returns:
        List of permuted tensors with ``None`` slots preserved.
    """
    return [
        torch.index_select(tensor, dim, order) if tensor is not None else None
        for tensor in tensors
    ]


class CovisibilityExtraction:
    def __init__(
        self,
        check_consistency: bool = False,
        include_track: bool = True,
        return_cpu: bool = True,
        graph_policy: str = "dense-transitive",
        virtual_track_depth_noise_ratio: float = 0.1,
        virtual_track_noise_seed: int = 0,
    ) -> None:
        if graph_policy not in {"dense-transitive", "planned-star"}:
            raise ValueError(
                f"unsupported covisibility graph policy: {graph_policy}"
            )
        if virtual_track_depth_noise_ratio < 0:
            raise ValueError(
                "virtual track depth noise ratio must be non-negative"
            )
        if virtual_track_noise_seed < 0:
            raise ValueError("virtual track noise seed must be non-negative")
        self.check_consistency = check_consistency
        self.include_track = include_track
        self.return_cpu = return_cpu
        self.graph_policy = graph_policy
        self.virtual_track_depth_noise_ratio = virtual_track_depth_noise_ratio
        self.virtual_track_noise_seed = virtual_track_noise_seed
        self.last_report: dict[str, int | float | str] = {}

    def main(
        self,
        predictions: dict,
        indexes: torch.Tensor,
        images_change: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Run reprojection-consistency verification and extract virtual tracks.

        Converts depth to world points, scores pairwise consistency, and
        samples virtual tracks on a coarse grid.

        Args:
            predictions: Two-view inference output with at least
                ``"extrinsics"``, ``"intrinsics"``, ``"depth"``,
                ``"depth_conf"`` and optionally ``"track"``.
            indexes: Per-batch image indices used to rescale tracks back to
                the original image resolution.
            images_change: Per-image (scale_x, scale_y, offset_x, offset_y)
                used to rescale virtual tracks.

        Returns:
            Tuple of CPU tensors
            ``(extrinsics, intrinsics, scores, tracks_virtual,
            points3d_virtual, valid_virtual)``.
        """
        extrinsics = predictions["extrinsics"]
        intrinsics = predictions["intrinsics"]

        depth_transformed = self._convert_from_depth_to_world_points(
            predictions["depth"], extrinsics, intrinsics
        )

        verify = (
            self._verify_by_reprojection_n2
            if self.graph_policy == "dense-transitive"
            else self._verify_by_reprojection_star
        )
        if self.check_consistency:
            scores, valid_mask = verify(
                depth_transformed,
                extrinsics,
                intrinsics,
                conf=predictions["depth_conf"],
            )
        else:
            scores, valid_mask = verify(
                depth_transformed, extrinsics, intrinsics
            )

        tracks_virtual, points3d_virtual, valid_virtual, _isnegative_virtual = (
            self._calculate_virtual_tracks(
                predictions["depth"],
                extrinsics,
                intrinsics,
                valid_mask,
                indexes,
            )
        )

        batch_count, view_count = indexes.shape[:2]
        dense_pair_count = int(batch_count * view_count * view_count)
        evaluated_pair_count = (
            dense_pair_count
            if self.graph_policy == "dense-transitive"
            else int(batch_count * (1 + 2 * max(0, view_count - 1)))
        )
        self.last_report = {
            "contractId": "jarailsense.gluemap-covisibility/v1",
            "graphPolicy": self.graph_policy,
            "batchCount": int(batch_count),
            "viewCount": int(view_count),
            "evaluatedDirectedPairCount": evaluated_pair_count,
            "denseEquivalentDirectedPairCount": dense_pair_count,
            "virtualTrackDepthNoiseRatio": float(
                self.virtual_track_depth_noise_ratio
            ),
            "virtualTrackNoiseSeed": int(self.virtual_track_noise_seed),
            "virtualTrackNoiseSeedPolicy": (
                "sha256-base-seed-and-star-indexes/v1"
            ),
        }

        if "track" in predictions and self.include_track:
            for i in range(indexes.shape[0]):
                for j, _idx_inner in enumerate(indexes[i].tolist()):
                    tracks_virtual[i, j] = rescale_tracks_single(
                        tracks_virtual[i, j], images_change[i][j]
                    )

        outputs = (
            extrinsics,
            intrinsics,
            scores,
            tracks_virtual,
            points3d_virtual,
            valid_virtual,
        )
        if self.return_cpu:
            return tuple(value.cpu() for value in outputs)
        return outputs

    def _calculate_virtual_tracks(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsic: torch.Tensor,
        valid_mask: torch.Tensor,
        indexes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample a coarse grid of virtual tracks from the reference depth map.

        Filters the reference-view depth to its inter-quartile range, draws
        reproducibly perturbed depth samples on a 14-pixel-stride grid,
        unprojects to camera/world points, and projects through
        ``project_tracks`` to obtain virtual track positions across all views.

        Args:
            depth: Per-view depth map of shape ``(B, N, H, W, 1)``.
            extrinsics: Per-view extrinsics ``(B, N, 4, 4)``.
            intrinsic: Per-view intrinsics ``(B, N, 3, 3)``.
            valid_mask: Pairwise validity mask from
                :meth:`_verify_by_reprojection_n2`.

        Returns:
            ``(tracks_virtual, points3d_virtual, valid_virtual,
            isnegative_virtual)`` as returned by ``project_tracks``.
        """
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

        # For each grid point, check whether the depth is valid.
        # If not, set the mean to be median
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
        sampled_depth = selected_depth
        if self.virtual_track_depth_noise_ratio > 0:
            noise_rows = []
            for batch_index in range(B):
                ordered_indexes = ",".join(
                    str(int(value))
                    for value in indexes[batch_index].detach().cpu().tolist()
                )
                identity = (
                    f"{self.virtual_track_noise_seed}:{ordered_indexes}"
                ).encode()
                seed = int.from_bytes(
                    hashlib.sha256(identity).digest()[:8], "little"
                ) % (2**63 - 1)
                generator = torch.Generator(device=selected_depth.device)
                generator.manual_seed(seed)
                noise_rows.append(
                    torch.randn(
                        selected_depth[batch_index].shape,
                        dtype=selected_depth.dtype,
                        device=selected_depth.device,
                        generator=generator,
                    )
                )
            noise = torch.stack(noise_rows, dim=0)
            sampled_depth = selected_depth + (
                noise * self.virtual_track_depth_noise_ratio * selected_depth
            )

        camera_points = world_points * sampled_depth  # (B, H // 4, W // 14, 3)

        return project_tracks(camera_points, extrinsics, intrinsic)

    def _project_point(
        self,
        world_points: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsic: torch.Tensor,
        from_0_to_j: bool = True,
        images_change: torch.Tensor | None = None,
        images_shape_ori: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | int]:
        """
        Forward/back-project per-pixel world points between frame 0 and frame j.

        When ``from_0_to_j`` is True, projects frame-0 world points into
        every frame, samples each frame's world-point map at those
        projections, then projects the resampled points back into frame 0
        for round-trip comparison. When False, the roles are reversed.

        Args:
            world_points: Per-view world-point grids ``(B, N, H, W, 3)``.
            extrinsics: Per-view extrinsics ``(B, N, 4, 4)``.
            intrinsic: Per-view intrinsics ``(B, N, 3, 3)``.
            from_0_to_j: Direction of the projection round-trip.
            images_change: Optional ``(B, N, 4)`` rescale params for the
                original image domain; restricts validity checks to the
                in-image region of each frame.
            images_shape_ori: Optional ``(B, N, 2)`` original image shapes
                paired with ``images_change``.

        Returns:
            ``(image_points_i, world_points_i_prime, invalid_i2j,
            valid_number)`` describing the round-trip projection result and
            its validity mask.
        """
        B, N, H, W, _ = world_points.shape
        if from_0_to_j:
            # First, transform the points from the first frame to the
            # other frames
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
        world_points: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsic: torch.Tensor,
        threshold_reproj: float = 4.0,
        consistent_threshold: float = 0.1,
        lambda_conf: float = 0.05,
        return_seperate_scores: bool = False,
        images_change: torch.Tensor | None = None,
        images_shape_ori: torch.Tensor | None = None,
        symmetric_scores: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Score reference-frame consistency via round-trip reprojection error.

        Args:
            world_points: Per-view world-point grids ``(B, N, H, W, 3)``.
            extrinsics: Per-view extrinsics ``(B, N, 4, 4)``.
            intrinsic: Per-view intrinsics ``(B, N, 3, 3)``.
            threshold_reproj: Pixel reprojection-error inlier threshold.
            consistent_threshold: Reserved for future world-point
                consistency check (currently unused).
            lambda_conf: Reserved for future depth-confidence weighting
                (currently unused).
            return_seperate_scores: Reserved flag (currently unused).
            images_change: Optional ``(B, N, 4)`` rescale params; passed
                through to :meth:`_project_point`.
            images_shape_ori: Optional ``(B, N, 2)`` original image shapes.

        Returns:
            ``(scores, valid)`` where ``scores`` is per-view inlier ratio of
            shape ``(B, N)`` and ``valid`` is the per-pixel inlier mask of
            shape ``(B, N, H, W)``.
        """

        B, N, H, W, _ = world_points.shape

        if images_shape_ori is not None:
            assert images_change is not None, (
                "If images_shape_ori is not None, "
                "images_change should not be None."
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
        (
            image_points_j,
            world_points_i_prime,
            invalid_i2j,
            valid_number_reverse,
        ) = self._project_point(
                world_points,
                extrinsics,
                intrinsic,
                from_0_to_j=False,
                images_change=images_change,
                images_shape_ori=images_shape_ori,
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

        if symmetric_scores:
            errors_j = torch.norm(
                image_points_j - grids.unsqueeze(0).unsqueeze(1), dim=-1
            )
            valid_reverse = (errors_j < threshold_reproj) * (~invalid_i2j)
            scores_reverse = (
                valid_reverse.flatten(-2, -1).sum(dim=-1)
                / valid_number_reverse
            )
            scores = torch.maximum(scores, scores_reverse)

        return scores, valid

    def _verify_by_reprojection_star(
        self,
        world_points: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsic: torch.Tensor,
        threshold_reproj: float = 4.0,
        consistent_threshold: float = 0.1,
        lambda_conf: float = 0.05,
        images_change: torch.Tensor | None = None,
        images_shape_ori: torch.Tensor | None = None,
        **_unused: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Score only center-neighbor edges while preserving both directions.

        The explicit pair plan already defines every edge in a streaming Star.
        One batched center sweep therefore evaluates the center-to-neighbor and
        neighbor-to-center round trips without constructing the dense neighbor
        graph or running a CPU shortest-path pass.
        """

        return self._verify_by_reprojection(
            world_points,
            extrinsics,
            intrinsic,
            threshold_reproj,
            consistent_threshold,
            lambda_conf,
            False,
            images_change,
            images_shape_ori,
            symmetric_scores=True,
        )

    def _verify_by_reprojection_n2(
        self,
        world_points: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsic: torch.Tensor,
        threshold_reproj: float = 4.0,
        consistent_threshold: float = 0.1,
        lambda_conf: float = 0.05,
        images_change: torch.Tensor | None = None,
        images_shape_ori: torch.Tensor | None = None,
        conf: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        N^2 covisibility scoring across every pair of views.

        Repeatedly swaps each view to the reference slot, calls
        :meth:`_verify_by_reprojection`, then propagates pairwise scores to
        a global per-view score via shortest-path on the negative-log-score
        graph rooted at view 0.

        Args:
            world_points: Per-view world-point grids ``(B, N, H, W, 3)``.
            extrinsics: Per-view extrinsics ``(B, N, 4, 4)``.
            intrinsic: Per-view intrinsics ``(B, N, 3, 3)``.
            threshold_reproj: Pixel reprojection-error inlier threshold.
            consistent_threshold: Forwarded to :meth:`_verify_by_reprojection`.
            lambda_conf: Forwarded to :meth:`_verify_by_reprojection`.
            images_change: Optional ``(B, N, 4)`` rescale params.
            images_shape_ori: Optional ``(B, N, 2)`` original image shapes.

        Returns:
            ``(scores_inner, valid_mask_0)``: per-view aggregated score
            ``(B, N)`` plus the per-pixel inlier mask from the j=0 sweep.
        """
        B, N, H, W, _ = world_points.shape
        scores_all = []
        valid_mask_0 = None
        for j in range(N):
            # swap the order of the indexes
            order = _calculate_index_mappings(j, N).to(world_points.device)
            world_points, extrinsics, intrinsic = _switch_tensor_order(
                [world_points, extrinsics, intrinsic], order
            )

            if images_change is not None:
                images_change, images_shape_ori = _switch_tensor_order(
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
            world_points, extrinsics, intrinsic, scores = _switch_tensor_order(
                [world_points, extrinsics, intrinsic, scores], order
            )

            if images_change is not None:
                images_change, images_shape_ori = _switch_tensor_order(
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

    def _convert_from_depth_to_world_points(
        self,
        depth: torch.Tensor,
        extrinsics: torch.Tensor,
        intrinsic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Unproject per-pixel depths into world-frame 3D point grids.

        Args:
            depth: Per-view depth maps ``(B, N, H, W, 1)``.
            extrinsics: Per-view extrinsics ``(B, N, 4, 4)``.
            intrinsic: Per-view intrinsics ``(B, N, 3, 3)``.

        Returns:
            World-frame point grid ``(B, N, H, W, 3)``.
        """
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
