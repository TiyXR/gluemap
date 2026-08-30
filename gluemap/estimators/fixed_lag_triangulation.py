"""GPU-batched DLT triangulation for selected fixed-lag tracks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gluemap.estimators.active_track_store import SelectedTrackState


class FixedLagTriangulationError(ValueError):
    """Raised when selected tracks cannot form a bounded DLT problem."""


@dataclass(frozen=True)
class TriangulatedTrackState:
    track_uid: str
    xyz: tuple[float, float, float]
    observations: tuple[Any, ...]
    positive_depth_fraction: float | None
    maximum_reprojection_error_pixels: float | None
    initialization_source: str = "dlt"


def _intrinsics_matrix(value: Any, device: str) -> torch.Tensor:
    matrix = torch.as_tensor(value, dtype=torch.float64, device=device).squeeze()
    if matrix.shape != (3, 3) or not torch.isfinite(matrix).all():
        raise FixedLagTriangulationError("shared intrinsics matrix is invalid")
    return matrix


def triangulate_selected_tracks(
    tracks: list[SelectedTrackState],
    rotations: dict[int, np.ndarray],
    centers: dict[int, np.ndarray],
    intrinsics: Any,
    *,
    device_policy: str = "cuda-preferred",
    microbatch_tracks: int = 4096,
    solver_policy: str = "homogeneous-svd",
) -> tuple[list[TriangulatedTrackState], dict[str, Any]]:
    """Triangulate selected tracks without a COLMAP database or image IO."""
    if device_policy not in {"cuda-required", "cuda-preferred", "cpu"}:
        raise FixedLagTriangulationError("triangulation device policy is invalid")
    if microbatch_tracks < 1:
        raise FixedLagTriangulationError("triangulation microbatch is invalid")
    if solver_policy not in {
        "homogeneous-svd",
        "homogeneous-gram-eigh",
    }:
        raise FixedLagTriangulationError("triangulation solver policy is invalid")
    if device_policy == "cuda-required" and not torch.cuda.is_available():
        raise FixedLagTriangulationError("CUDA triangulation is unavailable")
    device = (
        "cuda"
        if device_policy != "cpu" and torch.cuda.is_available()
        else "cpu"
    )
    if not tracks:
        raise FixedLagTriangulationError("selected track set is empty")
    camera_ids = set(rotations) & set(centers)
    if not camera_ids:
        raise FixedLagTriangulationError("camera pose set is empty")

    started = time.perf_counter()
    matrix_k = _intrinsics_matrix(intrinsics, device)
    ordered_camera_ids = sorted(camera_ids)
    camera_index_by_frame = {
        frame_id: index for index, frame_id in enumerate(ordered_camera_ids)
    }
    projections = []
    for frame_id in ordered_camera_ids:
        rotation = torch.as_tensor(
            rotations[frame_id], dtype=torch.float64, device=device
        )
        center = torch.as_tensor(
            centers[frame_id], dtype=torch.float64, device=device
        )
        if rotation.shape != (3, 3) or center.shape != (3,):
            raise FixedLagTriangulationError("camera pose shape is invalid")
        extrinsics = torch.cat((rotation, (-rotation @ center)[:, None]), dim=1)
        projections.append(matrix_k @ extrinsics)
    projection_stack = torch.stack(projections)

    usable = []
    maximum_views = 0
    for track in tracks:
        observations = tuple(
            value
            for value in track.observations
            if value.geometry_ordinal in camera_index_by_frame
        )
        if len(observations) < 2:
            continue
        usable.append((track.track_uid, observations))
        maximum_views = max(maximum_views, len(observations))
    if not usable:
        raise FixedLagTriangulationError("no selected track has two camera views")

    results: list[TriangulatedTrackState] = []
    all_reprojection_errors: list[torch.Tensor] = []
    invalid_count = 0
    microbatch_count = 0
    for start in range(0, len(usable), microbatch_tracks):
        batch = usable[start : start + microbatch_tracks]
        microbatch_count += 1
        batch_maximum_views = max(len(observations) for _, observations in batch)
        frame_indices = np.zeros(
            (len(batch), batch_maximum_views), dtype=np.int64
        )
        targets = np.zeros(
            (len(batch), batch_maximum_views, 2), dtype=np.float64
        )
        observation_mask = np.zeros(
            (len(batch), batch_maximum_views), dtype=np.bool_
        )
        for track_index, (_, observations) in enumerate(batch):
            for view_index, observation in enumerate(observations):
                frame_indices[track_index, view_index] = camera_index_by_frame[
                    observation.geometry_ordinal
                ]
                targets[track_index, view_index] = (
                    float(observation.x),
                    float(observation.y),
                )
                observation_mask[track_index, view_index] = True

        frame_index_tensor = torch.as_tensor(frame_indices, device=device)
        target_tensor = torch.as_tensor(targets, device=device)
        mask_tensor = torch.as_tensor(observation_mask, device=device)
        projection_batch = projection_stack[frame_index_tensor]
        row_x = (
            target_tensor[..., 0, None] * projection_batch[..., 2, :]
            - projection_batch[..., 0, :]
        )
        row_y = (
            target_tensor[..., 1, None] * projection_batch[..., 2, :]
            - projection_batch[..., 1, :]
        )
        design = torch.stack((row_x, row_y), dim=2)
        design = design.masked_fill(~mask_tensor[..., None, None], 0.0)
        design = design.reshape(len(batch), batch_maximum_views * 2, 4)
        if solver_policy == "homogeneous-svd":
            _, _, right = torch.linalg.svd(design, full_matrices=False)
            homogeneous = right[:, -1, :]
        else:
            gram = design.transpose(1, 2) @ design
            _, eigenvectors = torch.linalg.eigh(gram)
            homogeneous = eigenvectors[:, :, 0]
        valid_w = homogeneous[:, 3].abs() > 1e-12
        denominator = torch.where(
            valid_w[:, None],
            homogeneous[:, 3:4],
            torch.ones_like(homogeneous[:, 3:4]),
        )
        points = homogeneous[:, :3] / denominator
        valid_track = valid_w & torch.isfinite(points).all(dim=1)

        homogeneous_points = torch.cat(
            (
                points,
                torch.ones((len(batch), 1), dtype=torch.float64, device=device),
            ),
            dim=1,
        )
        projected = torch.einsum(
            "bvij,bj->bvi", projection_batch, homogeneous_points
        )
        depth = projected[..., 2]
        valid_depth = depth.abs() > 1e-12
        safe_depth = torch.where(valid_depth, depth, torch.ones_like(depth))
        predicted_xy = projected[..., :2] / safe_depth[..., None]
        reprojection_errors = torch.linalg.vector_norm(
            predicted_xy - target_tensor, dim=2
        )
        reprojection_errors = torch.where(
            valid_depth,
            reprojection_errors,
            torch.full_like(reprojection_errors, float("inf")),
        )
        maximum_errors = reprojection_errors.masked_fill(
            ~mask_tensor, -float("inf")
        ).max(dim=1).values
        view_counts = mask_tensor.sum(dim=1)
        positive_depth_fractions = (
            ((depth > 0) & mask_tensor).sum(dim=1) / view_counts
        )
        all_reprojection_errors.append(
            reprojection_errors[valid_track[:, None] & mask_tensor]
        )

        points_cpu = points.detach().cpu().numpy()
        maximum_errors_cpu = maximum_errors.detach().cpu().numpy()
        positive_depth_cpu = positive_depth_fractions.detach().cpu().numpy()
        valid_track_cpu = valid_track.detach().cpu().numpy()
        for track_index, (track_uid, observations) in enumerate(batch):
            if not bool(valid_track_cpu[track_index]):
                invalid_count += 1
                continue
            results.append(
                TriangulatedTrackState(
                    track_uid=track_uid,
                    xyz=tuple(float(value) for value in points_cpu[track_index]),
                    observations=observations,
                    positive_depth_fraction=float(positive_depth_cpu[track_index]),
                    maximum_reprojection_error_pixels=float(
                        maximum_errors_cpu[track_index]
                    ),
                )
            )

    if not results:
        raise FixedLagTriangulationError("every DLT track is degenerate")
    errors = torch.cat(all_reprojection_errors)
    quantiles = torch.quantile(
        errors,
        torch.tensor((0.5, 0.95), dtype=torch.float64, device=device),
    ).cpu()
    report = {
        "contractId": "jarailsense.gluemap-fixed-lag-triangulation/v1",
        "status": "passed",
        "publishable": False,
        "backend": device,
        "gpuUsed": device == "cuda",
        "inputTrackCount": len(tracks),
        "usableTrackCount": len(usable),
        "triangulatedTrackCount": len(results),
        "degenerateTrackCount": invalid_count,
        "maximumViewsPerTrack": maximum_views,
        "microbatchTracks": microbatch_tracks,
        "microbatchCount": microbatch_count,
        "solverPolicy": solver_policy,
        "tensorLayout": "padded-contiguous-batch",
        "reprojectionErrorP50Pixels": float(quantiles[0]),
        "reprojectionErrorP95Pixels": float(quantiles[1]),
        "wallSeconds": time.perf_counter() - started,
    }
    return results, report
