"""CUDA-vectorized calibrated PnP for dense pose-only localization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


class CudaPoseOnlyPnpError(ValueError):
    """Raised when pose-only PnP inputs or the CUDA backend are invalid."""


@dataclass(frozen=True)
class CudaPoseOnlyPnpResult:
    rotation_world_to_camera: torch.Tensor
    translation_world_to_camera: torch.Tensor
    inlier_mask: torch.Tensor
    inlier_count: int
    inlier_ratio: float
    positive_depth_fraction: float
    reprojection_p95_pixels: float
    condition_estimate: float
    hypothesis_count: int
    refinement_iterations: int
    backend: str = "torch-cuda-batched-dlt-ransac-gn/v1"

    def metrics(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "inlierCount": self.inlier_count,
            "inlierRatio": self.inlier_ratio,
            "positiveDepthFraction": self.positive_depth_fraction,
            "reprojectionP95Pixels": self.reprojection_p95_pixels,
            "conditionEstimate": self.condition_estimate,
            "hypothesisCount": self.hypothesis_count,
            "refinementIterations": self.refinement_iterations,
            "gpuUsed": True,
        }


def _skew(values: torch.Tensor) -> torch.Tensor:
    zeros = torch.zeros_like(values[..., 0])
    x, y, z = values.unbind(dim=-1)
    return torch.stack(
        (zeros, -z, y, z, zeros, -x, -y, x, zeros), dim=-1
    ).reshape(*values.shape[:-1], 3, 3)


def _so3_exp(rotation_vectors: torch.Tensor) -> torch.Tensor:
    theta = torch.linalg.vector_norm(rotation_vectors, dim=-1, keepdim=True)
    axis = rotation_vectors / theta.clamp_min(1e-12)
    skew = _skew(axis)
    identity = torch.eye(
        3, dtype=rotation_vectors.dtype, device=rotation_vectors.device
    ).expand(*rotation_vectors.shape[:-1], 3, 3)
    sin = torch.sin(theta)[..., None]
    cosine = torch.cos(theta)[..., None]
    rodrigues = identity + sin * skew + (1.0 - cosine) * (skew @ skew)
    small = theta[..., 0] < 1e-7
    first_order = identity + _skew(rotation_vectors)
    return torch.where(small[..., None, None], first_order, rodrigues)


def _project(
    points_world: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    intrinsics: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    camera = torch.einsum("...ij,nj->...ni", rotation, points_world)
    camera = camera + translation[..., None, :]
    depth = camera[..., 2]
    safe_depth = depth.clamp_min(1e-8)
    x = intrinsics[0, 0] * camera[..., 0] / safe_depth + intrinsics[0, 2]
    y = intrinsics[1, 1] * camera[..., 1] / safe_depth + intrinsics[1, 2]
    return torch.stack((x, y), dim=-1), depth


def _batched_dlt(
    points_world: torch.Tensor,
    points_image: torch.Tensor,
    intrinsics: torch.Tensor,
    samples: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world = points_world[samples]
    image = points_image[samples]
    homogeneous = torch.cat(
        (world, torch.ones_like(world[..., :1])), dim=-1
    )
    normalized_x = (image[..., 0] - intrinsics[0, 2]) / intrinsics[0, 0]
    normalized_y = (image[..., 1] - intrinsics[1, 2]) / intrinsics[1, 1]
    zeros = torch.zeros_like(homogeneous)
    rows_x = torch.cat(
        (homogeneous, zeros, -normalized_x[..., None] * homogeneous), dim=-1
    )
    rows_y = torch.cat(
        (zeros, homogeneous, -normalized_y[..., None] * homogeneous), dim=-1
    )
    design = torch.stack((rows_x, rows_y), dim=-2).flatten(-3, -2)
    _, singular, right = torch.linalg.svd(design, full_matrices=False)
    projection = right[..., -1, :].reshape(-1, 3, 4)
    matrix = projection[..., :3]
    determinant = torch.linalg.det(matrix)
    scale = torch.sign(determinant) * determinant.abs().clamp_min(1e-18).pow(1 / 3)
    normalized_matrix = matrix / scale[:, None, None]
    u, _, vh = torch.linalg.svd(normalized_matrix)
    orientation = u @ vh
    improper = torch.linalg.det(orientation) < 0
    if improper.any():
        u = u.clone()
        u[improper, :, -1] *= -1
        orientation = u @ vh
    translation = projection[..., 3] / scale[:, None]
    condition = singular[..., 0] / singular[..., -2].clamp_min(1e-12)
    return orientation, translation, condition


def _refine_pose(
    points_world: torch.Tensor,
    points_image: torch.Tensor,
    intrinsics: torch.Tensor,
    rotation: torch.Tensor,
    translation: torch.Tensor,
    inliers: torch.Tensor,
    *,
    iterations: int,
    huber_pixels: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    condition = float("inf")
    identity = torch.eye(6, dtype=points_world.dtype, device=points_world.device)
    for _ in range(iterations):
        selected_world = points_world[inliers]
        selected_image = points_image[inliers]
        camera = (rotation @ selected_world.T).T + translation
        depth = camera[:, 2].clamp_min(1e-8)
        predicted = torch.stack(
            (
                intrinsics[0, 0] * camera[:, 0] / depth + intrinsics[0, 2],
                intrinsics[1, 1] * camera[:, 1] / depth + intrinsics[1, 2],
            ),
            dim=-1,
        )
        residual = selected_image - predicted
        residual_norm = torch.linalg.vector_norm(residual, dim=-1)
        weights = torch.where(
            residual_norm <= huber_pixels,
            torch.ones_like(residual_norm),
            huber_pixels / residual_norm.clamp_min(1e-8),
        )
        projection_jacobian = torch.zeros(
            (camera.shape[0], 2, 3),
            dtype=points_world.dtype,
            device=points_world.device,
        )
        projection_jacobian[:, 0, 0] = intrinsics[0, 0] / depth
        projection_jacobian[:, 0, 2] = (
            -intrinsics[0, 0] * camera[:, 0] / depth.square()
        )
        projection_jacobian[:, 1, 1] = intrinsics[1, 1] / depth
        projection_jacobian[:, 1, 2] = (
            -intrinsics[1, 1] * camera[:, 1] / depth.square()
        )
        motion_jacobian = torch.cat(
            (-_skew(camera), torch.eye(3, device=camera.device, dtype=camera.dtype).expand(camera.shape[0], 3, 3)),
            dim=-1,
        )
        jacobian = projection_jacobian @ motion_jacobian
        weighted_jacobian = jacobian * weights.sqrt()[:, None, None]
        weighted_residual = residual * weights.sqrt()[:, None]
        normal = torch.einsum("nij,nik->jk", weighted_jacobian, weighted_jacobian)
        gradient = torch.einsum("nij,ni->j", weighted_jacobian, weighted_residual)
        eigenvalues = torch.linalg.eigvalsh(normal)
        condition = float(
            (eigenvalues[-1] / eigenvalues[0].clamp_min(1e-12)).item()
        )
        delta = torch.linalg.solve(normal + identity * 1e-6, gradient)
        increment = _so3_exp(delta[:3])
        rotation = increment @ rotation
        translation = increment @ translation + delta[3:]
        if float(torch.linalg.vector_norm(delta).item()) < 1e-7:
            break
    return rotation, translation, condition


def solve_pose_only_pnp_cuda(
    points_world: torch.Tensor,
    points_image: torch.Tensor,
    intrinsics: torch.Tensor,
    *,
    hypothesis_count: int = 128,
    sample_size: int = 8,
    reprojection_threshold_pixels: float = 3.0,
    refinement_iterations: int = 8,
    random_seed: int = 0,
) -> CudaPoseOnlyPnpResult:
    """Solve one calibrated pose with batched CUDA RANSAC and GPU refinement."""
    if not torch.cuda.is_available():
        raise CudaPoseOnlyPnpError("CUDA is required for pose-only PnP")
    if points_world.ndim != 2 or points_world.shape[1] != 3:
        raise CudaPoseOnlyPnpError("world points must have shape N,3")
    if points_image.shape != (points_world.shape[0], 2):
        raise CudaPoseOnlyPnpError("image points must have shape N,2")
    if intrinsics.shape != (3, 3):
        raise CudaPoseOnlyPnpError("intrinsics must have shape 3,3")
    if points_world.shape[0] < sample_size or sample_size < 6:
        raise CudaPoseOnlyPnpError("insufficient PnP correspondences")
    if hypothesis_count < 1 or refinement_iterations < 0:
        raise CudaPoseOnlyPnpError("PnP iteration budget is invalid")
    if reprojection_threshold_pixels <= 0:
        raise CudaPoseOnlyPnpError("PnP reprojection threshold is invalid")
    device = torch.device("cuda")
    # Image-scale DLT is sufficiently conditioned in FP32 after calibrated
    # normalization.  Consumer GPUs execute its many small SVDs much faster
    # than FP64, while the final quality gates remain in pixel units.
    dtype = torch.float32
    world = points_world.to(device=device, dtype=dtype)
    image = points_image.to(device=device, dtype=dtype)
    camera = intrinsics.to(device=device, dtype=dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(random_seed)
    samples = torch.stack(
        [
            torch.randperm(world.shape[0], generator=generator, device=device)[
                :sample_size
            ]
            for _ in range(hypothesis_count)
        ]
    )
    rotations, translations, hypothesis_conditions = _batched_dlt(
        world, image, camera, samples
    )
    projected, depth = _project(world, rotations, translations, camera)
    errors = torch.linalg.vector_norm(projected - image[None, :, :], dim=-1)
    inliers = (errors <= reprojection_threshold_pixels) & (depth > 0)
    counts = inliers.sum(dim=-1)
    clipped_error = torch.where(
        inliers, errors, torch.full_like(errors, reprojection_threshold_pixels)
    ).sum(dim=-1)
    score = counts.to(dtype) * (world.shape[0] + 1) - clipped_error
    best = int(torch.argmax(score).item())
    best_inliers = inliers[best]
    if int(best_inliers.sum().item()) < 6:
        raise CudaPoseOnlyPnpError("PnP RANSAC found fewer than six inliers")
    rotation, translation, refined_condition = _refine_pose(
        world,
        image,
        camera,
        rotations[best],
        translations[best],
        best_inliers,
        iterations=refinement_iterations,
        huber_pixels=reprojection_threshold_pixels,
    )
    final_projected, final_depth = _project(world, rotation, translation, camera)
    final_errors = torch.linalg.vector_norm(final_projected - image, dim=-1)
    final_inliers = (
        (final_errors <= reprojection_threshold_pixels) & (final_depth > 0)
    )
    inlier_errors = final_errors[final_inliers]
    if inlier_errors.numel() == 0:
        raise CudaPoseOnlyPnpError("refined PnP pose has no inliers")
    return CudaPoseOnlyPnpResult(
        rotation_world_to_camera=rotation,
        translation_world_to_camera=translation,
        inlier_mask=final_inliers,
        inlier_count=int(final_inliers.sum().item()),
        inlier_ratio=float(final_inliers.double().mean().item()),
        positive_depth_fraction=float((final_depth > 0).double().mean().item()),
        reprojection_p95_pixels=float(torch.quantile(inlier_errors, 0.95).item()),
        condition_estimate=max(
            refined_condition, float(hypothesis_conditions[best].item())
        ),
        hypothesis_count=hypothesis_count,
        refinement_iterations=refinement_iterations,
    )
