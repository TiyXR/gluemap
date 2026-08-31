from __future__ import annotations

import math

import pytest
import torch

from gluemap.estimators.cuda_pose_only_pnp import (
    CudaPoseOnlyPnpError,
    solve_pose_only_pnp_cuda,
)


def _rotation_y(degrees: float) -> torch.Tensor:
    radians = math.radians(degrees)
    cosine = math.cos(radians)
    sine = math.sin(radians)
    return torch.tensor(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=torch.float64,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_pose_only_pnp_recovers_pose_with_outliers() -> None:
    generator = torch.Generator().manual_seed(17)
    points = torch.rand((256, 3), generator=generator, dtype=torch.float64)
    points[:, :2] = (points[:, :2] - 0.5) * 6.0
    points[:, 2] = points[:, 2] * 8.0 + 12.0
    rotation = _rotation_y(4.0)
    translation = torch.tensor([0.4, -0.2, 1.0], dtype=torch.float64)
    intrinsics = torch.tensor(
        [[1200.0, 0.0, 960.0], [0.0, 1180.0, 540.0], [0.0, 0.0, 1.0]],
        dtype=torch.float64,
    )
    camera = (rotation @ points.T).T + translation
    image = torch.stack(
        (
            intrinsics[0, 0] * camera[:, 0] / camera[:, 2] + intrinsics[0, 2],
            intrinsics[1, 1] * camera[:, 1] / camera[:, 2] + intrinsics[1, 2],
        ),
        dim=-1,
    )
    image += torch.randn(image.shape, generator=generator) * 0.15
    image[:32] = torch.rand((32, 2), generator=generator) * torch.tensor(
        [1920.0, 1080.0]
    )

    result = solve_pose_only_pnp_cuda(
        points,
        image,
        intrinsics,
        hypothesis_count=128,
        reprojection_threshold_pixels=2.0,
        random_seed=29,
    )

    recovered_rotation = result.rotation_world_to_camera.cpu().to(rotation)
    recovered_translation = result.translation_world_to_camera.cpu().to(translation)
    relative = recovered_rotation @ rotation.T
    angle = torch.acos(((torch.trace(relative) - 1.0) / 2.0).clamp(-1.0, 1.0))
    assert math.degrees(float(angle)) < 0.1
    assert torch.linalg.vector_norm(
        recovered_translation - translation
    ) < 0.03
    assert result.inlier_count >= 220
    assert result.reprojection_p95_pixels < 0.5
    assert result.metrics()["gpuUsed"] is True


def test_cuda_pose_only_pnp_rejects_insufficient_correspondences() -> None:
    with pytest.raises(CudaPoseOnlyPnpError, match="insufficient"):
        solve_pose_only_pnp_cuda(
            torch.zeros((5, 3)),
            torch.zeros((5, 2)),
            torch.eye(3),
        )
