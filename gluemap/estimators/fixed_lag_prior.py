"""GPU-batched Schur marginalization for a persistent fixed-lag FEJ prior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import time

import numpy as np
import pyceres
import torch

from gluemap.estimators.fixed_lag_ceres_linearization import (
    CeresProblemLinearization,
)


class FixedLagPriorError(ValueError):
    """Raised when fixed-lag linearization or prior identity is invalid."""


@dataclass(frozen=True)
class FejPriorState:
    """Pose-only normal form and square-root factor at frozen FEJ points."""

    camera_ids: tuple[int, ...]
    linearization_points: torch.Tensor
    hessian: torch.Tensor
    gradient: torch.Tensor
    factor: torch.Tensor
    factor_residual: torch.Tensor
    report: dict[str, Any]

    def cpu(self) -> "FejPriorState":
        """Return a detached CPU snapshot suitable for a durable checkpoint."""
        return FejPriorState(
            camera_ids=self.camera_ids,
            linearization_points=self.linearization_points.detach().cpu(),
            hessian=self.hessian.detach().cpu(),
            gradient=self.gradient.detach().cpu(),
            factor=self.factor.detach().cpu(),
            factor_residual=self.factor_residual.detach().cpu(),
            report=dict(self.report),
        )


def _quaternion_product_xyzw(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_xyz = left[:3]
    right_xyz = right[:3]
    return np.concatenate(
        (
            left[3] * right_xyz
            + right[3] * left_xyz
            + np.cross(left_xyz, right_xyz),
            np.asarray(
                [left[3] * right[3] - np.dot(left_xyz, right_xyz)],
                dtype=np.float64,
            ),
        )
    )


def _eigen_quaternion_minus(current: np.ndarray, origin: np.ndarray) -> np.ndarray:
    origin_conjugate = origin.copy()
    origin_conjugate[:3] *= -1.0
    difference = _quaternion_product_xyzw(current, origin_conjugate)
    norm = float(np.linalg.norm(difference[:3]))
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    theta = np.arctan2(norm, float(difference[3]))
    return difference[:3] * (theta / norm)


def _eigen_quaternion_plus_jacobian(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = quaternion
    return np.asarray(
        (
            (w, z, -y),
            (-z, w, x),
            (y, -x, w),
            (-x, -y, -z),
        ),
        dtype=np.float64,
    )


class FejPosePriorCostFunction(pyceres.CostFunction):
    """One dense pose-only square-root prior over Ceres pose manifolds."""

    def __init__(self, prior: FejPriorState) -> None:
        super().__init__()
        factor = prior.factor.detach().cpu().numpy().astype(np.float64)
        factor_residual = (
            prior.factor_residual.detach().cpu().numpy().astype(np.float64)
        )
        linearization = (
            prior.linearization_points.detach().cpu().numpy().astype(np.float64)
        )
        if factor.ndim != 2 or factor.shape[1] != len(prior.camera_ids) * 6:
            raise FixedLagPriorError("FEJ factor shape is invalid")
        if factor_residual.shape != (factor.shape[0],):
            raise FixedLagPriorError("FEJ factor residual shape is invalid")
        if linearization.shape != (len(prior.camera_ids), 7):
            raise FixedLagPriorError("FEJ ambient linearization shape is invalid")
        self.factor = factor
        self.factor_residual = factor_residual
        self.linearization = linearization
        self.set_num_residuals(factor.shape[0])
        self.set_parameter_block_sizes([7] * len(prior.camera_ids))

    def Evaluate(self, parameters, residuals, jacobians):  # noqa: N802
        deltas = []
        for current_value, origin in zip(
            parameters, self.linearization, strict=True
        ):
            current = np.asarray(current_value, dtype=np.float64)
            if current.shape != (7,):
                return False
            deltas.append(
                np.concatenate(
                    (
                        _eigen_quaternion_minus(current[:4], origin[:4]),
                        current[4:] - origin[4:],
                    )
                )
            )
        residuals[:] = self.factor @ np.concatenate(deltas) + self.factor_residual
        if jacobians is not None:
            for index, current_value in enumerate(parameters):
                if jacobians[index] is None:
                    continue
                current = np.asarray(current_value, dtype=np.float64)
                plus = np.zeros((7, 6), dtype=np.float64)
                plus[:4, :3] = _eigen_quaternion_plus_jacobian(current[:4])
                plus[4:, 3:] = np.eye(3, dtype=np.float64)
                tangent = self.factor[:, index * 6 : (index + 1) * 6]
                np.asarray(jacobians[index]).reshape(self.factor.shape[0], 7)[:] = (
                    tangent @ plus.T
                )
        return True


def _resolve_device(policy: str) -> torch.device:
    if policy not in {"cuda-required", "cuda-preferred", "cpu"}:
        raise FixedLagPriorError("fixed-lag prior device policy is invalid")
    cuda_available = torch.cuda.is_available()
    if policy == "cuda-required" and not cuda_available:
        raise FixedLagPriorError("CUDA fixed-lag prior backend is unavailable")
    return torch.device(
        "cuda" if cuda_available and policy != "cpu" else "cpu"
    )


def _as_tensor(
    value: Any,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype, device=device).contiguous()


def _scatter_vector_blocks(
    output: torch.Tensor,
    camera_indexes: torch.Tensor,
    values: torch.Tensor,
) -> None:
    valid = camera_indexes >= 0
    if not bool(valid.any()):
        return
    indexes = camera_indexes[valid]
    blocks = values[valid]
    for row in range(6):
        output.index_add_(0, indexes * 6 + row, blocks[:, row])


def _scatter_matrix_blocks(
    output: torch.Tensor,
    row_indexes: torch.Tensor,
    column_indexes: torch.Tensor,
    values: torch.Tensor,
) -> None:
    valid = (row_indexes >= 0) & (column_indexes >= 0)
    if not bool(valid.any()):
        return
    rows = row_indexes[valid]
    columns = column_indexes[valid]
    blocks = values[valid]
    dimension = output.shape[0]
    flattened = output.view(-1)
    for row in range(6):
        for column in range(6):
            indexes = (rows * 6 + row) * dimension + columns * 6 + column
            flattened.index_add_(0, indexes, blocks[:, row, column])


def _runtime_point_schur_batch_size(
    point_count: int,
    maximum_views: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> int:
    """Use at most 20% of currently free VRAM for one dense view-pair batch."""
    if point_count < 1:
        return 1
    element_bytes = torch.empty((), dtype=dtype).element_size()
    bytes_per_point = max(
        1,
        maximum_views
        * maximum_views
        * 36
        * (element_bytes + torch.empty((), dtype=torch.int64).element_size()),
    )
    if device.type != "cuda":
        return point_count
    free_bytes, _ = torch.cuda.mem_get_info(device)
    working_budget = max(bytes_per_point, int(free_bytes * 0.2))
    return max(1, min(point_count, working_budget // bytes_per_point))


def _apply_point_schur_correction(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    camera_indexes: torch.Tensor,
    camera_point: torch.Tensor,
    point_inverse: torch.Tensor,
    point_gradient: torch.Tensor,
) -> tuple[int, int]:
    """Accumulate every point/view correction with one kernel per microbatch."""
    point_count, maximum_views = camera_indexes.shape
    dimension = hessian.shape[0]
    batch_size = _runtime_point_schur_batch_size(
        point_count,
        maximum_views,
        dtype=hessian.dtype,
        device=hessian.device,
    )
    tangent = torch.arange(6, device=hessian.device)
    batch_count = 0
    for start in range(0, point_count, batch_size):
        stop = min(point_count, start + batch_size)
        indexes = camera_indexes[start:stop]
        blocks = camera_point[start:stop]
        inverse = point_inverse[start:stop]
        point_g = point_gradient[start:stop]
        valid_views = indexes >= 0
        gradient_correction = torch.einsum(
            "pvij,pjk,pk->pvi", blocks, inverse, point_g
        )
        gradient_targets = indexes[:, :, None] * 6 + tangent
        gradient.index_add_(
            0,
            gradient_targets[valid_views].reshape(-1),
            -gradient_correction[valid_views].reshape(-1),
        )

        matrix_correction = torch.einsum(
            "pvij,pjk,pwlk->pvwil", blocks, inverse, blocks
        )
        valid_pairs = valid_views[:, :, None] & valid_views[:, None, :]
        row_targets = (
            indexes[:, :, None, None, None] * 6
            + tangent[None, None, None, :, None]
        )
        column_targets = (
            indexes[:, None, :, None, None] * 6
            + tangent[None, None, None, None, :]
        )
        flattened_targets = row_targets * dimension + column_targets
        hessian.view(-1).index_add_(
            0,
            flattened_targets.expand_as(matrix_correction)[valid_pairs].reshape(-1),
            -matrix_correction[valid_pairs].reshape(-1),
        )
        batch_count += 1
    return batch_size, batch_count


def _symmetric_pseudoinverse(
    matrix: torch.Tensor,
    relative_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    values, vectors = torch.linalg.eigh((matrix + matrix.transpose(-1, -2)) * 0.5)
    maximum = values.amax(dim=-1, keepdim=True).clamp_min(
        torch.finfo(values.dtype).eps
    )
    accepted = values > maximum * relative_threshold
    inverse_values = torch.where(
        accepted, values.reciprocal(), torch.zeros_like(values)
    )
    inverse = (vectors * inverse_values.unsqueeze(-2)) @ vectors.transpose(-1, -2)
    return inverse, values, accepted


def _camera_block_jacobi_whitening(
    hessian: torch.Tensor,
    relative_rank_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise FixedLagPriorError("block-Jacobi Hessian shape is invalid")
    if hessian.shape[0] % 6 != 0:
        raise FixedLagPriorError("block-Jacobi Hessian is not camera aligned")
    camera_count = hessian.shape[0] // 6
    blocks = torch.stack(
        tuple(
            hessian[
                index * 6 : (index + 1) * 6,
                index * 6 : (index + 1) * 6,
            ]
            for index in range(camera_count)
        )
    )
    blocks = (blocks + blocks.transpose(-1, -2)) * 0.5
    block_values, block_vectors = torch.linalg.eigh(blocks)
    block_maximum = block_values.amax(dim=-1).clamp_min(
        torch.finfo(hessian.dtype).eps
    )
    block_floor = block_maximum * relative_rank_threshold
    inverse_square_root_values = torch.rsqrt(
        torch.maximum(block_values, block_floor[:, None])
    )
    square_root_values = torch.sqrt(
        torch.maximum(block_values, block_floor[:, None])
    )
    whiteners = (
        block_vectors * inverse_square_root_values[:, None, :]
    ) @ block_vectors.transpose(-1, -2)
    unwhiteners = (
        block_vectors * square_root_values[:, None, :]
    ) @ block_vectors.transpose(-1, -2)
    block_accepted = block_values > block_floor[:, None]
    block_minimum = torch.where(
        block_accepted,
        block_values,
        torch.full_like(block_values, float("inf")),
    ).amin(dim=-1)
    maximum_block_condition = float(
        (block_maximum / block_minimum).max().item()
    )
    return (
        torch.block_diag(*whiteners.unbind()),
        torch.block_diag(*unwhiteners.unbind()),
        maximum_block_condition,
    )


def _camera_block_jacobi_pseudoinverse(
    matrix: torch.Tensor,
    relative_rank_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    whitening, _, maximum_block_condition = (
        _camera_block_jacobi_whitening(matrix, relative_rank_threshold)
    )
    equilibrated = whitening @ matrix @ whitening.T
    equilibrated = (equilibrated + equilibrated.T) * 0.5
    inverse, eigenvalues, accepted = _symmetric_pseudoinverse(
        equilibrated, relative_rank_threshold
    )
    return (
        whitening.T @ inverse @ whitening,
        eigenvalues,
        accepted,
        maximum_block_condition,
    )


def _prior_factorization(
    hessian: torch.Tensor,
    gradient: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    raw_accepted: torch.Tensor,
    *,
    relative_rank_threshold: float,
    policy: str,
    minimum_nullity: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Build a square-root prior in the selected conditioning coordinates.

    The raw eigenvalue ratio is coordinate-scale dependent.  In particular,
    pose translation Jacobians acquire a growing rotation/translation lever arm
    as a forward-only trajectory moves away from the global origin.  A block
    Jacobi congruence normalizes each 6-DoF camera block before measuring and
    truncating the spectrum.  The resulting factor is mapped back to the
    original coordinates, so the stored Hessian, gradient and FEJ points remain
    unchanged while the numerical rank no longer depends on pose units.
    """
    if policy not in {"raw-eigenvalue", "camera-block-jacobi"}:
        raise FixedLagPriorError("prior condition estimate policy is invalid")
    if hessian.ndim != 2 or hessian.shape[0] != hessian.shape[1]:
        raise FixedLagPriorError("prior Hessian shape is invalid")
    if hessian.shape[0] % 6 != 0:
        raise FixedLagPriorError("prior Hessian is not camera-block aligned")

    raw_positive_values = eigenvalues[raw_accepted]
    raw_positive_vectors = eigenvectors[:, raw_accepted]
    raw_rank = int(raw_accepted.sum().item())
    raw_nullity = int(len(eigenvalues) - raw_rank)
    raw_condition = (
        float(
            (raw_positive_values.max() / raw_positive_values.min()).item()
        )
        if raw_rank
        else float("inf")
    )
    result: dict[str, Any] = {
        "policy": policy,
        "rawRank": raw_rank,
        "rawNullity": raw_nullity,
        "rawConditionEstimate": raw_condition,
        "selectedRank": raw_rank,
        "selectedNullity": raw_nullity,
        "equilibratedRank": None,
        "equilibratedNullity": None,
        "equilibratedConditionEstimate": None,
        "maximumCameraBlockConditionEstimate": None,
    }
    if policy == "raw-eigenvalue":
        factor = (
            raw_positive_values.sqrt()[:, None] * raw_positive_vectors.T
        )
        factor_residual = (
            raw_positive_vectors.T @ gradient
        ) / raw_positive_values.sqrt()
        result["selectedConditionEstimate"] = raw_condition
        return factor, factor_residual, result

    whitening, unwhitening, maximum_block_condition = (
        _camera_block_jacobi_whitening(hessian, relative_rank_threshold)
    )
    equilibrated = whitening @ hessian @ whitening.T
    equilibrated = (equilibrated + equilibrated.T) * 0.5
    equilibrated_values, equilibrated_vectors = torch.linalg.eigh(equilibrated)
    equilibrated_maximum = equilibrated_values.max().clamp_min(
        torch.finfo(hessian.dtype).eps
    )
    equilibrated_accepted = (
        equilibrated_values
        > equilibrated_maximum * relative_rank_threshold
    )
    if minimum_nullity:
        equilibrated_accepted[:minimum_nullity] = False
    equilibrated_positive = equilibrated_values[equilibrated_accepted]
    equilibrated_rank = int(equilibrated_accepted.sum().item())
    equilibrated_nullity = int(len(equilibrated_values) - equilibrated_rank)
    equilibrated_condition = (
        float(
            (
                equilibrated_positive.max()
                / equilibrated_positive.min()
            ).item()
        )
        if equilibrated_rank
        else float("inf")
    )
    equilibrated_positive_vectors = equilibrated_vectors[
        :, equilibrated_accepted
    ]
    factor = (
        equilibrated_positive.sqrt()[:, None]
        * equilibrated_positive_vectors.T
    ) @ unwhitening
    factor_residual = (
        equilibrated_positive_vectors.T @ (whitening @ gradient)
    ) / equilibrated_positive.sqrt()
    result.update(
        {
            "equilibratedRank": equilibrated_rank,
            "equilibratedNullity": equilibrated_nullity,
            "equilibratedConditionEstimate": equilibrated_condition,
            "selectedRank": equilibrated_rank,
            "selectedNullity": equilibrated_nullity,
            "maximumCameraBlockConditionEstimate": (
                maximum_block_condition
            ),
            "selectedConditionEstimate": equilibrated_condition,
        }
    )
    return factor, factor_residual, result


def _canonical_normal_from_factor(
    factor: torch.Tensor,
    factor_residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the exact normal form represented by the solver prior.

    Spectrum truncation is part of the prior contract.  Propagating the
    pre-truncation normal form would reintroduce modes that Ceres never saw and
    lets round-off-level negative eigenvalues accumulate across long runs.
    """
    hessian = factor.T @ factor
    hessian = (hessian + hessian.T) * 0.5
    return hessian, factor.T @ factor_residual


def _validate_prior_identity(
    previous: FejPriorState,
    camera_ids: tuple[int, ...],
    linearization_points: torch.Tensor,
) -> list[int]:
    index_by_id = {camera_id: index for index, camera_id in enumerate(camera_ids)}
    if any(camera_id not in index_by_id for camera_id in previous.camera_ids):
        raise FixedLagPriorError("previous FEJ prior camera identity is absent")
    indexes = [index_by_id[camera_id] for camera_id in previous.camera_ids]
    current = linearization_points[indexes].detach().cpu()
    previous_points = previous.linearization_points.detach().cpu()
    if current.shape != previous_points.shape or not torch.equal(
        current, previous_points
    ):
        raise FixedLagPriorError("previous FEJ linearization point changed")
    return indexes


def marginalize_pose_prior(
    prior: FejPriorState,
    *,
    eliminate_camera_id: int,
    device_policy: str = "cuda-preferred",
    relative_rank_threshold: float = 1e-10,
    maximum_condition_estimate: float | None = None,
    condition_estimate_policy: str = "raw-eigenvalue",
    expected_nullity: int | None = None,
) -> FejPriorState:
    """Schur-eliminate one pose directly from an existing FEJ prior.

    A sealed forward-only stream has no reason to reconstruct new landmarks
    merely to release its retained tail poses.  The pose-only prior already
    contains the complete history needed for that release, so this path keeps
    the small dense normal form on the resolved torch device and performs one
    six-dimensional Schur complement without triangulation or Ceres BA.
    """
    return marginalize_pose_prior_batch(
        prior,
        eliminate_camera_ids=(eliminate_camera_id,),
        device_policy=device_policy,
        relative_rank_threshold=relative_rank_threshold,
        maximum_condition_estimate=maximum_condition_estimate,
        condition_estimate_policy=condition_estimate_policy,
        expected_nullity=expected_nullity,
    )


def marginalize_pose_prior_batch(
    prior: FejPriorState,
    *,
    eliminate_camera_ids: tuple[int, ...] | list[int],
    device_policy: str = "cuda-preferred",
    relative_rank_threshold: float = 1e-10,
    maximum_condition_estimate: float | None = None,
    condition_estimate_policy: str = "raw-eigenvalue",
    expected_nullity: int | None = None,
) -> FejPriorState:
    """Schur-eliminate multiple retained poses with one dense solve.

    A forward-only scheduler may ingest every keyframe while delaying the
    expensive nonlinear BA for a small bounded number of advances.  Once BA
    runs, all pending oldest poses share one FEJ linearization and can be
    removed as one block instead of repeating eigendecomposition and Schur
    factorization for every pose.
    """
    started = time.perf_counter()
    camera_ids = tuple(int(value) for value in prior.camera_ids)
    eliminated_ids = tuple(int(value) for value in eliminate_camera_ids)
    if (
        len(camera_ids) < 2
        or len(set(camera_ids)) != len(camera_ids)
        or not eliminated_ids
        or len(set(eliminated_ids)) != len(eliminated_ids)
        or any(value not in camera_ids for value in eliminated_ids)
        or len(eliminated_ids) >= len(camera_ids)
        or not 0 < relative_rank_threshold < 1
    ):
        raise FixedLagPriorError("pose-only prior marginalization identity is invalid")
    device = _resolve_device(device_policy)
    dtype = torch.float64
    camera_count = len(camera_ids)
    dimension = camera_count * 6
    hessian = _as_tensor(prior.hessian, dtype=dtype, device=device)
    gradient = _as_tensor(prior.gradient, dtype=dtype, device=device)
    linearization = _as_tensor(
        prior.linearization_points, dtype=dtype, device=device
    )
    if (
        hessian.shape != (dimension, dimension)
        or gradient.shape != (dimension,)
        or linearization.shape != (camera_count, 7)
        or not all(
            bool(torch.isfinite(value).all())
            for value in (hessian, gradient, linearization)
        )
    ):
        raise FixedLagPriorError("pose-only prior marginalization shape is invalid")

    hessian = (hessian + hessian.T) * 0.5
    eliminate_indexes = tuple(camera_ids.index(value) for value in eliminated_ids)
    retained_indexes = tuple(
        index for index in range(camera_count) if index not in eliminate_indexes
    )
    eliminate_columns = torch.cat(
        tuple(
            torch.arange(index * 6, (index + 1) * 6, device=device)
            for index in eliminate_indexes
        )
    )
    retained_columns = torch.cat(
        tuple(
            torch.arange(index * 6, (index + 1) * 6, device=device)
            for index in retained_indexes
        )
    )
    h_mm = hessian[eliminate_columns[:, None], eliminate_columns]
    h_mr = hessian[eliminate_columns[:, None], retained_columns]
    h_rr = hessian[retained_columns[:, None], retained_columns]
    g_m = gradient[eliminate_columns]
    g_r = gradient[retained_columns]
    raw_marginal_eigenvalues = torch.linalg.eigvalsh(
        (h_mm + h_mm.T) * 0.5
    )
    marginal_block_condition = None
    if condition_estimate_policy == "camera-block-jacobi":
        (
            inverse_mm,
            marginal_eigenvalues,
            marginal_rank_mask,
            marginal_block_condition,
        ) = _camera_block_jacobi_pseudoinverse(
            h_mm, relative_rank_threshold
        )
        marginal_condition_policy = "camera-block-jacobi"
    else:
        inverse_mm, marginal_eigenvalues, marginal_rank_mask = (
            _symmetric_pseudoinverse(h_mm, relative_rank_threshold)
        )
        marginal_condition_policy = "raw-eigenvalue"
    retained_hessian = h_rr - h_mr.T @ inverse_mm @ h_mr
    retained_gradient = g_r - h_mr.T @ inverse_mm @ g_m
    retained_hessian = (retained_hessian + retained_hessian.T) * 0.5

    retained_camera_ids = tuple(
        camera_ids[index] for index in retained_indexes
    )
    retained_points = linearization[
        torch.as_tensor(retained_indexes, dtype=torch.int64, device=device)
    ].contiguous()
    scale_gauge = torch.zeros_like(retained_gradient).view(-1, 6)
    scale_gauge[:, 3:] = retained_points[:, 4:]
    scale_gauge = scale_gauge.reshape(-1)
    scale_gauge_norm = torch.linalg.vector_norm(scale_gauge)
    scale_projection_applied = bool(
        expected_nullity == 1
        and scale_gauge_norm > torch.finfo(dtype).eps
    )
    removed_scale_gradient = 0.0
    if scale_projection_applied:
        scale_gauge /= scale_gauge_norm
        hessian_scale = retained_hessian @ scale_gauge
        scale_information = scale_gauge @ hessian_scale
        retained_hessian = (
            retained_hessian
            - torch.outer(hessian_scale, scale_gauge)
            - torch.outer(scale_gauge, hessian_scale)
            + scale_information * torch.outer(scale_gauge, scale_gauge)
        )
        removed_scale_gradient = float(
            (scale_gauge @ retained_gradient).detach().cpu().item()
        )
        retained_gradient = (
            retained_gradient - scale_gauge * (scale_gauge @ retained_gradient)
        )
        retained_hessian = (retained_hessian + retained_hessian.T) * 0.5

    eigenvalues, eigenvectors = torch.linalg.eigh(retained_hessian)
    maximum_eigenvalue = eigenvalues.max().clamp_min(torch.finfo(dtype).eps)
    accepted = eigenvalues > maximum_eigenvalue * relative_rank_threshold
    if scale_projection_applied and expected_nullity:
        accepted[:expected_nullity] = False
    factor, factor_residual, condition_metrics = _prior_factorization(
        retained_hessian,
        retained_gradient,
        eigenvalues,
        eigenvectors,
        accepted,
        relative_rank_threshold=relative_rank_threshold,
        policy=condition_estimate_policy,
        minimum_nullity=(expected_nullity or 0) if scale_projection_applied else 0,
    )
    canonical_hessian, canonical_gradient = _canonical_normal_from_factor(
        factor, factor_residual
    )
    hessian_error = float(
        torch.max(torch.abs(canonical_hessian - retained_hessian)).item()
    )
    gradient_error = float(
        torch.max(torch.abs(canonical_gradient - retained_gradient)).item()
    )
    rank = condition_metrics["selectedRank"]
    nullity = condition_metrics["selectedNullity"]
    condition = condition_metrics["selectedConditionEstimate"]
    status = "passed"
    reasons: list[str] = []
    if expected_nullity is not None and nullity != expected_nullity:
        status = "failed"
        reasons.append("unexpected-prior-nullity")
    if (
        maximum_condition_estimate is not None
        and condition > maximum_condition_estimate
    ):
        status = "failed"
        reasons.append("prior-condition-exceeded")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    wall = time.perf_counter() - started
    single_pose = len(eliminated_ids) == 1
    terminal_solve_mode = (
        "prior-only-schur" if single_pose else "prior-only-batch-schur"
    )
    report = {
        "contractId": "jarailsense.gluemap-schur-fej-prior/v1",
        "status": status,
        "sourceLinearizationContractId": "pose-only-fej-prior",
        "terminalSolveMode": terminal_solve_mode,
        "backend": device.type,
        "gpuUsed": device.type == "cuda",
        "cameraCountBefore": camera_count,
        "cameraCountAfter": len(retained_camera_ids),
        "eliminatedCameraId": eliminated_ids[0] if single_pose else None,
        "eliminatedCameraIds": list(eliminated_ids),
        "eliminatedCameraCount": len(eliminated_ids),
        "pointCount": 0,
        "pointWithoutVariableCameraCount": 0,
        "residualCount": int(prior.factor.shape[0]),
        "maximumPointCameraCount": 0,
        "degeneratePointCount": 0,
        "marginalPoseRank": int(marginal_rank_mask.sum().item()),
        "marginalPoseConditionPolicy": marginal_condition_policy,
        "marginalPoseMaximumCameraBlockConditionEstimate": (
            marginal_block_condition
        ),
        "marginalPoseEigenvalues": [
            float(value) for value in marginal_eigenvalues.detach().cpu().tolist()
        ],
        "marginalPoseRawEigenvalues": [
            float(value)
            for value in raw_marginal_eigenvalues.detach().cpu().tolist()
        ],
        "priorRank": rank,
        "priorNullity": nullity,
        "priorConditionEstimate": condition,
        "priorConditionEstimatePolicy": condition_metrics["policy"],
        "priorRawRank": condition_metrics["rawRank"],
        "priorRawNullity": condition_metrics["rawNullity"],
        "priorRawConditionEstimate": condition_metrics[
            "rawConditionEstimate"
        ],
        "priorEquilibratedRank": condition_metrics["equilibratedRank"],
        "priorEquilibratedNullity": condition_metrics[
            "equilibratedNullity"
        ],
        "priorEquilibratedConditionEstimate": condition_metrics[
            "equilibratedConditionEstimate"
        ],
        "priorMaximumCameraBlockConditionEstimate": condition_metrics[
            "maximumCameraBlockConditionEstimate"
        ],
        "factorGradientMaximumAbsoluteError": gradient_error,
        "factorHessianMaximumAbsoluteError": hessian_error,
        "normalFormCanonicalizedFromFactor": True,
        "scaleGaugeProjectionApplied": scale_projection_applied,
        "scaleGaugeRemovedGradient": removed_scale_gradient,
        "cpuSparseNormalWallSeconds": 0.0,
        "resolvedSchurWallSeconds": wall,
        "pointSchurMicrobatchPoints": 0,
        "pointSchurMicrobatchCount": 0,
        "reasonCodes": reasons,
    }
    return FejPriorState(
        camera_ids=retained_camera_ids,
        linearization_points=retained_points,
        hessian=canonical_hessian,
        gradient=canonical_gradient,
        factor=factor,
        factor_residual=factor_residual,
        report=report,
    )


def marginalize_linearized_tracks(
    *,
    camera_ids: list[int] | tuple[int, ...],
    linearization_points: Any,
    observation_camera_indexes: Any,
    residuals: Any,
    camera_jacobians: Any,
    point_jacobians: Any,
    eliminate_camera_id: int,
    previous_prior: FejPriorState | None = None,
    device_policy: str = "cuda-preferred",
    relative_rank_threshold: float = 1e-10,
    maximum_condition_estimate: float | None = None,
    condition_estimate_policy: str = "raw-eigenvalue",
    expected_nullity: int | None = None,
) -> FejPriorState:
    """Eliminate every point and one oldest pose without forming a point Hessian.

    Inputs are padded per-track observation blocks. Camera Jacobians use the
    six-dimensional pose tangent ordering ``[rotation, translation]`` and point
    Jacobians use XYZ. ``-1`` camera indexes mark padding. The implementation
    batches every point inverse and Schur update on the resolved torch device;
    only the final report scalars synchronize to the host.
    """
    ordered_camera_ids = tuple(int(value) for value in camera_ids)
    if (
        not ordered_camera_ids
        or len(set(ordered_camera_ids)) != len(ordered_camera_ids)
        or eliminate_camera_id not in ordered_camera_ids
    ):
        raise FixedLagPriorError("fixed-lag camera ordering is invalid")
    if not 0 < relative_rank_threshold < 1:
        raise FixedLagPriorError("fixed-lag rank threshold is invalid")
    device = _resolve_device(device_policy)
    dtype = torch.float64
    points = _as_tensor(
        linearization_points, dtype=dtype, device=device
    )
    camera_indexes = _as_tensor(
        observation_camera_indexes, dtype=torch.int64, device=device
    )
    residual = _as_tensor(residuals, dtype=dtype, device=device)
    camera_jacobian = _as_tensor(
        camera_jacobians, dtype=dtype, device=device
    )
    point_jacobian = _as_tensor(
        point_jacobians, dtype=dtype, device=device
    )
    point_count, maximum_views = camera_indexes.shape
    if points.shape != (len(ordered_camera_ids), 7):
        raise FixedLagPriorError("FEJ pose linearization shape is invalid")
    quaternion_norms = torch.linalg.vector_norm(points[:, :4], dim=1)
    if not bool(
        torch.allclose(
            quaternion_norms,
            torch.ones_like(quaternion_norms),
            rtol=1e-10,
            atol=1e-10,
        )
    ):
        raise FixedLagPriorError("FEJ quaternion linearization is not normalized")
    if residual.shape != (point_count, maximum_views, 2):
        raise FixedLagPriorError("linearized residual shape is invalid")
    if camera_jacobian.shape != (point_count, maximum_views, 2, 6):
        raise FixedLagPriorError("camera Jacobian shape is invalid")
    if point_jacobian.shape != (point_count, maximum_views, 2, 3):
        raise FixedLagPriorError("point Jacobian shape is invalid")
    if point_count == 0 or maximum_views < 2:
        raise FixedLagPriorError("fixed-lag track batch is empty")
    if bool((camera_indexes >= len(ordered_camera_ids)).any()):
        raise FixedLagPriorError("observation camera index exceeds ordering")
    mask = camera_indexes >= 0
    if bool((mask.sum(dim=1) < 2).any()):
        raise FixedLagPriorError("marginalized track has fewer than two views")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (points, residual, camera_jacobian, point_jacobian)
    ):
        raise FixedLagPriorError("fixed-lag linearization is non-finite")

    mask_scalar = mask.to(dtype)
    masked_residual = residual * mask_scalar[:, :, None]
    masked_camera_jacobian = camera_jacobian * mask_scalar[:, :, None, None]
    masked_point_jacobian = point_jacobian * mask_scalar[:, :, None, None]
    point_hessian = torch.einsum(
        "pvri,pvrj->pij", masked_point_jacobian, masked_point_jacobian
    )
    point_gradient = torch.einsum(
        "pvri,pvr->pi", masked_point_jacobian, masked_residual
    )
    point_inverse, point_eigenvalues, point_rank_mask = _symmetric_pseudoinverse(
        point_hessian, relative_rank_threshold
    )
    camera_point = torch.einsum(
        "pvri,pvrj->pvij", masked_camera_jacobian, masked_point_jacobian
    )

    dimension = len(ordered_camera_ids) * 6
    camera_hessian = torch.zeros((dimension, dimension), dtype=dtype, device=device)
    camera_gradient = torch.zeros(dimension, dtype=dtype, device=device)
    for view in range(maximum_views):
        indexes = camera_indexes[:, view]
        diagonal = torch.einsum(
            "pri,prj->pij",
            masked_camera_jacobian[:, view],
            masked_camera_jacobian[:, view],
        )
        gradient = torch.einsum(
            "pri,pr->pi",
            masked_camera_jacobian[:, view],
            masked_residual[:, view],
        )
        _scatter_matrix_blocks(camera_hessian, indexes, indexes, diagonal)
        correction_gradient = torch.einsum(
            "pij,pjk,pk->pi",
            camera_point[:, view],
            point_inverse,
            point_gradient,
        )
        _scatter_vector_blocks(
            camera_gradient, indexes, gradient - correction_gradient
        )
        for other_view in range(maximum_views):
            correction = torch.einsum(
                "pij,pjk,plk->pil",
                camera_point[:, view],
                point_inverse,
                camera_point[:, other_view],
            )
            _scatter_matrix_blocks(
                camera_hessian,
                indexes,
                camera_indexes[:, other_view],
                -correction,
            )

    if previous_prior is not None:
        prior_indexes = _validate_prior_identity(
            previous_prior, ordered_camera_ids, points
        )
        prior_hessian = previous_prior.hessian.to(device=device, dtype=dtype)
        prior_gradient = previous_prior.gradient.to(device=device, dtype=dtype)
        if prior_hessian.shape != (len(prior_indexes) * 6,) * 2:
            raise FixedLagPriorError("previous FEJ prior Hessian shape is invalid")
        if prior_gradient.shape != (len(prior_indexes) * 6,):
            raise FixedLagPriorError("previous FEJ prior gradient shape is invalid")
        for row, camera_row in enumerate(prior_indexes):
            row_slice = slice(row * 6, (row + 1) * 6)
            target_row = slice(camera_row * 6, (camera_row + 1) * 6)
            camera_gradient[target_row] += prior_gradient[row_slice]
            for column, camera_column in enumerate(prior_indexes):
                column_slice = slice(column * 6, (column + 1) * 6)
                target_column = slice(camera_column * 6, (camera_column + 1) * 6)
                camera_hessian[target_row, target_column] += prior_hessian[
                    row_slice, column_slice
                ]

    camera_hessian = (camera_hessian + camera_hessian.T) * 0.5
    eliminate_index = ordered_camera_ids.index(eliminate_camera_id)
    eliminate_columns = torch.arange(
        eliminate_index * 6,
        (eliminate_index + 1) * 6,
        device=device,
    )
    retained_columns = torch.cat(
        (
            torch.arange(0, eliminate_index * 6, device=device),
            torch.arange((eliminate_index + 1) * 6, dimension, device=device),
        )
    )
    h_mm = camera_hessian[eliminate_columns[:, None], eliminate_columns]
    h_mr = camera_hessian[eliminate_columns[:, None], retained_columns]
    h_rr = camera_hessian[retained_columns[:, None], retained_columns]
    g_m = camera_gradient[eliminate_columns]
    g_r = camera_gradient[retained_columns]
    inverse_mm, marginal_eigenvalues, marginal_rank_mask = (
        _symmetric_pseudoinverse(h_mm, relative_rank_threshold)
    )
    retained_hessian = h_rr - h_mr.T @ inverse_mm @ h_mr
    retained_gradient = g_r - h_mr.T @ inverse_mm @ g_m
    retained_hessian = (retained_hessian + retained_hessian.T) * 0.5

    eigenvalues, eigenvectors = torch.linalg.eigh(retained_hessian)
    maximum_eigenvalue = eigenvalues.max().clamp_min(torch.finfo(dtype).eps)
    accepted = eigenvalues > maximum_eigenvalue * relative_rank_threshold
    factor, factor_residual, condition_metrics = _prior_factorization(
        retained_hessian,
        retained_gradient,
        eigenvalues,
        eigenvectors,
        accepted,
        relative_rank_threshold=relative_rank_threshold,
        policy=condition_estimate_policy,
    )
    canonical_hessian, canonical_gradient = _canonical_normal_from_factor(
        factor, factor_residual
    )
    hessian_error = float(
        torch.max(torch.abs(canonical_hessian - retained_hessian)).item()
    )
    gradient_error = float(
        torch.max(torch.abs(canonical_gradient - retained_gradient)).item()
    )
    rank = condition_metrics["selectedRank"]
    nullity = condition_metrics["selectedNullity"]
    condition = condition_metrics["selectedConditionEstimate"]
    status = "passed"
    reasons: list[str] = []
    if expected_nullity is not None and nullity != expected_nullity:
        status = "failed"
        reasons.append("unexpected-prior-nullity")
    if (
        maximum_condition_estimate is not None
        and condition > maximum_condition_estimate
    ):
        status = "failed"
        reasons.append("prior-condition-exceeded")
    report = {
        "contractId": "jarailsense.gluemap-schur-fej-prior/v1",
        "status": status,
        "backend": device.type,
        "gpuUsed": device.type == "cuda",
        "cameraCountBefore": len(ordered_camera_ids),
        "cameraCountAfter": len(ordered_camera_ids) - 1,
        "eliminatedCameraId": eliminate_camera_id,
        "pointCount": point_count,
        "observationCount": int(mask.sum().item()),
        "maximumTrackViews": maximum_views,
        "degeneratePointCount": int(
            (point_rank_mask.sum(dim=1) < 3).sum().item()
        ),
        "marginalPoseRank": int(marginal_rank_mask.sum().item()),
        "marginalPoseEigenvalues": [
            float(value) for value in marginal_eigenvalues.detach().cpu().tolist()
        ],
        "priorRank": rank,
        "priorNullity": nullity,
        "priorConditionEstimate": condition,
        "priorConditionEstimatePolicy": condition_metrics["policy"],
        "priorRawRank": condition_metrics["rawRank"],
        "priorRawNullity": condition_metrics["rawNullity"],
        "priorRawConditionEstimate": condition_metrics[
            "rawConditionEstimate"
        ],
        "priorEquilibratedRank": condition_metrics["equilibratedRank"],
        "priorEquilibratedNullity": condition_metrics[
            "equilibratedNullity"
        ],
        "priorEquilibratedConditionEstimate": condition_metrics[
            "equilibratedConditionEstimate"
        ],
        "priorMaximumCameraBlockConditionEstimate": condition_metrics[
            "maximumCameraBlockConditionEstimate"
        ],
        "factorGradientMaximumAbsoluteError": gradient_error,
        "factorHessianMaximumAbsoluteError": hessian_error,
        "normalFormCanonicalizedFromFactor": True,
        "reasonCodes": reasons,
    }
    retained_camera_ids = tuple(
        value for value in ordered_camera_ids if value != eliminate_camera_id
    )
    retained_points = torch.cat(
        (points[:eliminate_index], points[eliminate_index + 1 :]), dim=0
    )
    return FejPriorState(
        camera_ids=retained_camera_ids,
        linearization_points=retained_points,
        hessian=canonical_hessian,
        gradient=canonical_gradient,
        factor=factor,
        factor_residual=factor_residual,
        report=report,
    )


def marginalize_ceres_linearization(
    linearization: CeresProblemLinearization,
    *,
    eliminate_camera_id: int,
    previous_prior: FejPriorState | None = None,
    device_policy: str = "cuda-preferred",
    relative_rank_threshold: float = 1e-10,
    maximum_condition_estimate: float | None = None,
    condition_estimate_policy: str = "raw-eigenvalue",
    expected_nullity: int | None = None,
) -> FejPriorState:
    """Convert a solved Ceres CRS directly into the next pose-only FEJ prior."""
    from scipy.sparse import csr_matrix

    camera_ids = linearization.camera_ids
    if eliminate_camera_id not in camera_ids:
        raise FixedLagPriorError("eliminated Ceres camera identity is absent")
    if not 0 < relative_rank_threshold < 1:
        raise FixedLagPriorError("fixed-lag rank threshold is invalid")
    camera_count = len(camera_ids)
    point_count = len(linearization.point3d_ids)
    camera_dimension = camera_count * 6
    point_dimension = point_count * 3
    expected_columns = camera_dimension + point_dimension
    if linearization.report.get("columnCount") != expected_columns:
        raise FixedLagPriorError("Ceres linearization column identity differs")
    if linearization.pose_ambient_values.shape != (camera_count, 7):
        raise FixedLagPriorError("Ceres pose ambient values differ")

    sparse_started = time.perf_counter()
    sparse_cpu_started = time.process_time()
    native_normal_values = (
        linearization.camera_hessian,
        linearization.camera_gradient,
        linearization.point_hessian,
        linearization.point_gradient,
        linearization.block_point_indexes,
        linearization.block_camera_indexes,
        linearization.camera_point_hessian,
    )
    has_native_normals = all(value is not None for value in native_normal_values)
    if (
        any(value is not None for value in native_normal_values)
        and not has_native_normals
    ):
        raise FixedLagPriorError("native Ceres normal block set is incomplete")
    if has_native_normals:
        camera_hessian = np.asarray(
            linearization.camera_hessian, dtype=np.float64
        )
        camera_gradient = np.asarray(
            linearization.camera_gradient, dtype=np.float64
        )
        point_hessian = np.asarray(
            linearization.point_hessian, dtype=np.float64
        )
        point_gradient = np.asarray(
            linearization.point_gradient, dtype=np.float64
        )
        block_points = np.asarray(
            linearization.block_point_indexes, dtype=np.int64
        )
        block_cameras = np.asarray(
            linearization.block_camera_indexes, dtype=np.int64
        )
        camera_point_blocks = np.asarray(
            linearization.camera_point_hessian, dtype=np.float64
        )
        block_count = len(block_points)
        if (
            camera_hessian.shape != (camera_dimension, camera_dimension)
            or camera_gradient.shape != (camera_dimension,)
            or point_hessian.shape != (point_count, 3, 3)
            or point_gradient.shape != (point_count, 3)
            or block_cameras.shape != (block_count,)
            or camera_point_blocks.shape != (block_count, 6, 3)
            or np.any(block_points < 0)
            or np.any(block_points >= point_count)
            or np.any(block_cameras < 0)
            or np.any(block_cameras >= camera_count)
            or np.any(np.diff(block_points) < 0)
        ):
            raise FixedLagPriorError("native Ceres normal block layout is invalid")
    else:
        jacobian = csr_matrix(
            (
                linearization.jacobian_values,
                linearization.column_indices,
                linearization.row_offsets,
            ),
            shape=(len(linearization.residuals), expected_columns),
        )
        camera_jacobian = jacobian[:, :camera_dimension]
        point_jacobian = jacobian[:, camera_dimension:]
        residual = linearization.residuals
        camera_hessian = (camera_jacobian.T @ camera_jacobian).toarray()
        camera_gradient = np.asarray(camera_jacobian.T @ residual).reshape(-1)
        camera_point_hessian = (camera_jacobian.T @ point_jacobian).tocoo()
        point_hessian_sparse = (point_jacobian.T @ point_jacobian).tocoo()
        point_gradient = np.asarray(point_jacobian.T @ residual).reshape(
            point_count, 3
        )

        point_rows = point_hessian_sparse.row // 3
        point_columns = point_hessian_sparse.col // 3
        if np.any(point_rows != point_columns):
            raise FixedLagPriorError("Ceres point Hessian is not block diagonal")
        point_hessian = np.zeros((point_count, 3, 3), dtype=np.float64)
        np.add.at(
            point_hessian,
            (
                point_rows,
                point_hessian_sparse.row % 3,
                point_hessian_sparse.col % 3,
            ),
            point_hessian_sparse.data,
        )

        block_camera_indexes = camera_point_hessian.row // 6
        block_point_indexes = camera_point_hessian.col // 3
        block_keys = block_point_indexes * camera_count + block_camera_indexes
        unique_keys, inverse_keys = np.unique(block_keys, return_inverse=True)
        camera_point_blocks = np.zeros(
            (len(unique_keys), 6, 3), dtype=np.float64
        )
        np.add.at(
            camera_point_blocks,
            (
                inverse_keys,
                camera_point_hessian.row % 6,
                camera_point_hessian.col % 3,
            ),
            camera_point_hessian.data,
        )
        block_points = unique_keys // camera_count
        block_cameras = unique_keys % camera_count
    block_counts = np.bincount(block_points, minlength=point_count)
    points_without_variable_camera = int(np.count_nonzero(block_counts == 0))
    maximum_views = max(1, int(block_counts.max()))
    first_block = np.repeat(
        np.cumsum(block_counts) - block_counts, block_counts
    )
    block_positions = np.arange(len(block_points)) - first_block
    padded_camera_indexes = np.full(
        (point_count, maximum_views), -1, dtype=np.int64
    )
    padded_camera_point = np.zeros(
        (point_count, maximum_views, 6, 3), dtype=np.float64
    )
    padded_camera_indexes[block_points, block_positions] = block_cameras
    padded_camera_point[block_points, block_positions] = camera_point_blocks
    sparse_wall = time.perf_counter() - sparse_started
    sparse_cpu = time.process_time() - sparse_cpu_started

    device = _resolve_device(device_policy)
    dtype = torch.float64
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    schur_started = time.perf_counter()
    schur_cpu_started = time.process_time()
    hessian = _as_tensor(camera_hessian, dtype=dtype, device=device)
    gradient = _as_tensor(camera_gradient, dtype=dtype, device=device)
    camera_indexes = _as_tensor(
        padded_camera_indexes, dtype=torch.int64, device=device
    )
    camera_point = _as_tensor(
        padded_camera_point, dtype=dtype, device=device
    )
    point_hessian_tensor = _as_tensor(
        point_hessian, dtype=dtype, device=device
    )
    point_gradient_tensor = _as_tensor(
        point_gradient, dtype=dtype, device=device
    )
    point_inverse, point_eigenvalues, point_rank_mask = _symmetric_pseudoinverse(
        point_hessian_tensor, relative_rank_threshold
    )
    schur_microbatch_points, schur_microbatch_count = (
        _apply_point_schur_correction(
            hessian,
            gradient,
            camera_indexes,
            camera_point,
            point_inverse,
            point_gradient_tensor,
        )
    )
    hessian = (hessian + hessian.T) * 0.5

    if previous_prior is not None:
        index_by_camera_id = {
            camera_id: index for index, camera_id in enumerate(camera_ids)
        }
        if any(
            camera_id not in index_by_camera_id
            for camera_id in previous_prior.camera_ids
        ):
            raise FixedLagPriorError("previous FEJ prior camera identity is absent")
        prior_hessian = previous_prior.hessian.to(device=device, dtype=dtype)
        prior_gradient = previous_prior.gradient.to(device=device, dtype=dtype)
        prior_count = len(previous_prior.camera_ids)
        if (
            prior_hessian.shape != (prior_count * 6, prior_count * 6)
            or prior_gradient.shape != (prior_count * 6,)
            or previous_prior.linearization_points.shape != (prior_count, 7)
        ):
            raise FixedLagPriorError("previous FEJ prior shape is invalid")
        prior_targets = previous_prior.linearization_points.detach().cpu().numpy()
        prior_delta = []
        for camera_id, target in zip(
            previous_prior.camera_ids, prior_targets, strict=True
        ):
            current = linearization.pose_ambient_values[
                index_by_camera_id[camera_id]
            ]
            prior_delta.extend(
                _eigen_quaternion_minus(current[:4], target[:4]).tolist()
            )
            prior_delta.extend((current[4:] - target[4:]).tolist())
        prior_gradient_current = prior_gradient + prior_hessian @ _as_tensor(
            prior_delta, dtype=dtype, device=device
        )
        for row, camera_id_row in enumerate(previous_prior.camera_ids):
            source_row = slice(row * 6, (row + 1) * 6)
            target_row_index = index_by_camera_id[camera_id_row]
            target_row = slice(target_row_index * 6, (target_row_index + 1) * 6)
            gradient[target_row] += prior_gradient_current[source_row]
            for column, camera_id_column in enumerate(previous_prior.camera_ids):
                source_column = slice(column * 6, (column + 1) * 6)
                target_column_index = index_by_camera_id[camera_id_column]
                target_column = slice(
                    target_column_index * 6, (target_column_index + 1) * 6
                )
                hessian[target_row, target_column] += prior_hessian[
                    source_row, source_column
                ]
        hessian = (hessian + hessian.T) * 0.5

    eliminate_index = camera_ids.index(eliminate_camera_id)
    eliminate_columns = torch.arange(
        eliminate_index * 6,
        (eliminate_index + 1) * 6,
        device=device,
    )
    retained_columns = torch.cat(
        (
            torch.arange(0, eliminate_index * 6, device=device),
            torch.arange(
                (eliminate_index + 1) * 6,
                camera_dimension,
                device=device,
            ),
        )
    )
    h_mm = hessian[eliminate_columns[:, None], eliminate_columns]
    h_mr = hessian[eliminate_columns[:, None], retained_columns]
    h_rr = hessian[retained_columns[:, None], retained_columns]
    g_m = gradient[eliminate_columns]
    g_r = gradient[retained_columns]
    inverse_mm, marginal_eigenvalues, marginal_rank_mask = (
        _symmetric_pseudoinverse(h_mm, relative_rank_threshold)
    )
    retained_hessian = h_rr - h_mr.T @ inverse_mm @ h_mr
    retained_gradient_current = g_r - h_mr.T @ inverse_mm @ g_m
    retained_hessian = (retained_hessian + retained_hessian.T) * 0.5

    retained_camera_ids = tuple(
        value for value in camera_ids if value != eliminate_camera_id
    )
    current_points = np.concatenate(
        (
            linearization.pose_ambient_values[:eliminate_index],
            linearization.pose_ambient_values[eliminate_index + 1 :],
        )
    )
    target_points = current_points.copy()
    if previous_prior is not None:
        previous_by_id = {
            camera_id: value
            for camera_id, value in zip(
                previous_prior.camera_ids,
                previous_prior.linearization_points.detach().cpu().numpy(),
                strict=True,
            )
        }
        for index, camera_id in enumerate(retained_camera_ids):
            if camera_id in previous_by_id:
                target_points[index] = previous_by_id[camera_id]
    delta_from_target = []
    for current, target in zip(current_points, target_points, strict=True):
        delta_from_target.extend(
            _eigen_quaternion_minus(current[:4], target[:4]).tolist()
        )
        delta_from_target.extend((current[4:] - target[4:]).tolist())
    delta_tensor = _as_tensor(
        delta_from_target, dtype=dtype, device=device
    )
    retained_gradient = (
        retained_gradient_current - retained_hessian @ delta_tensor
    )

    unconstrained_eigenvalues, unconstrained_eigenvectors = torch.linalg.eigh(
        retained_hessian
    )
    unconstrained_maximum_eigenvalue = unconstrained_eigenvalues.max().clamp_min(
        torch.finfo(dtype).eps
    )
    scale_gauge = torch.zeros_like(retained_gradient).view(-1, 6)
    scale_gauge[:, 3:] = _as_tensor(
        target_points[:, 4:], dtype=dtype, device=device
    )
    scale_gauge = scale_gauge.reshape(-1)
    scale_gauge_norm = torch.linalg.vector_norm(scale_gauge)
    if bool(scale_gauge_norm > torch.finfo(dtype).eps):
        scale_gauge /= scale_gauge_norm
        scale_relative_rayleigh = float(
            (
                scale_gauge
                @ retained_hessian
                @ scale_gauge
                / unconstrained_maximum_eigenvalue
            )
            .detach()
            .cpu()
            .item()
        )
        scale_smallest_eigenvector_cosine = float(
            torch.abs(scale_gauge @ unconstrained_eigenvectors[:, 0])
            .detach()
            .cpu()
            .item()
        )
    else:
        scale_relative_rayleigh = float("inf")
        scale_smallest_eigenvector_cosine = 0.0
    scale_projection_applied = bool(
        previous_prior is not None
        and expected_nullity == 1
        and scale_gauge_norm > torch.finfo(dtype).eps
    )
    removed_scale_gradient = 0.0
    if scale_projection_applied:
        hessian_scale = retained_hessian @ scale_gauge
        scale_information = scale_gauge @ hessian_scale
        retained_hessian = (
            retained_hessian
            - torch.outer(hessian_scale, scale_gauge)
            - torch.outer(scale_gauge, hessian_scale)
            + scale_information * torch.outer(scale_gauge, scale_gauge)
        )
        removed_scale_gradient = float(
            (scale_gauge @ retained_gradient).detach().cpu().item()
        )
        retained_gradient = (
            retained_gradient - scale_gauge * (scale_gauge @ retained_gradient)
        )
        retained_hessian = (retained_hessian + retained_hessian.T) * 0.5

    eigenvalues, eigenvectors = torch.linalg.eigh(retained_hessian)
    maximum_eigenvalue = eigenvalues.max().clamp_min(torch.finfo(dtype).eps)
    accepted = eigenvalues > maximum_eigenvalue * relative_rank_threshold
    if scale_projection_applied and expected_nullity:
        # The Ceres CRS path can retain a tiny projected gauge eigenvalue just
        # above the relative cutoff after merging the previous FEJ factor.
        accepted[:expected_nullity] = False
    factor, factor_residual, condition_metrics = _prior_factorization(
        retained_hessian,
        retained_gradient,
        eigenvalues,
        eigenvectors,
        accepted,
        relative_rank_threshold=relative_rank_threshold,
        policy=condition_estimate_policy,
        minimum_nullity=(expected_nullity or 0) if scale_projection_applied else 0,
    )
    canonical_hessian, canonical_gradient = _canonical_normal_from_factor(
        factor, factor_residual
    )
    hessian_error = float(
        torch.max(torch.abs(canonical_hessian - retained_hessian)).item()
    )
    gradient_error = float(
        torch.max(torch.abs(canonical_gradient - retained_gradient)).item()
    )
    rank = condition_metrics["selectedRank"]
    nullity = condition_metrics["selectedNullity"]
    condition = condition_metrics["selectedConditionEstimate"]
    status = "passed"
    reasons: list[str] = []
    if expected_nullity is not None and nullity != expected_nullity:
        status = "failed"
        reasons.append("unexpected-prior-nullity")
    if (
        maximum_condition_estimate is not None
        and condition > maximum_condition_estimate
    ):
        status = "failed"
        reasons.append("prior-condition-exceeded")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    schur_wall = time.perf_counter() - schur_started
    schur_cpu = time.process_time() - schur_cpu_started
    report = {
        "contractId": "jarailsense.gluemap-schur-fej-prior/v1",
        "status": status,
        "sourceLinearizationContractId": linearization.report.get("contractId"),
        "backend": device.type,
        "gpuUsed": device.type == "cuda",
        "cameraCountBefore": camera_count,
        "cameraCountAfter": camera_count - 1,
        "eliminatedCameraId": eliminate_camera_id,
        "pointCount": point_count,
        "pointWithoutVariableCameraCount": points_without_variable_camera,
        "residualCount": int(
            linearization.report.get(
                "residualCount", len(linearization.residuals)
            )
        ),
        "linearizationRepresentation": linearization.report.get(
            "representation", "ceres-crs"
        ),
        "maximumPointCameraCount": maximum_views,
        "degeneratePointCount": int(
            (point_rank_mask.sum(dim=1) < 3).sum().item()
        ),
        "marginalPoseRank": int(marginal_rank_mask.sum().item()),
        "marginalPoseEigenvalues": [
            float(value) for value in marginal_eigenvalues.detach().cpu().tolist()
        ],
        "priorRank": rank,
        "priorNullity": nullity,
        "priorConditionEstimate": condition,
        "priorConditionEstimatePolicy": condition_metrics["policy"],
        "priorRawRank": condition_metrics["rawRank"],
        "priorRawNullity": condition_metrics["rawNullity"],
        "priorRawConditionEstimate": condition_metrics[
            "rawConditionEstimate"
        ],
        "priorEquilibratedRank": condition_metrics["equilibratedRank"],
        "priorEquilibratedNullity": condition_metrics[
            "equilibratedNullity"
        ],
        "priorEquilibratedConditionEstimate": condition_metrics[
            "equilibratedConditionEstimate"
        ],
        "priorMaximumCameraBlockConditionEstimate": condition_metrics[
            "maximumCameraBlockConditionEstimate"
        ],
        "scaleGaugeProjectionApplied": scale_projection_applied,
        "scaleGaugeUnconstrainedRelativeRayleigh": scale_relative_rayleigh,
        "scaleGaugeUnconstrainedSmallestEigenvectorCosine": (
            scale_smallest_eigenvector_cosine
        ),
        "scaleGaugeRemovedGradient": removed_scale_gradient,
        "factorGradientMaximumAbsoluteError": gradient_error,
        "factorHessianMaximumAbsoluteError": hessian_error,
        "normalFormCanonicalizedFromFactor": True,
        "priorUnconstrainedSmallestRelativeEigenvalues": [
            float(value)
            for value in (
                unconstrained_eigenvalues[:8]
                / unconstrained_maximum_eigenvalue
            )
            .detach()
            .cpu()
            .tolist()
        ],
        "cpuSparseNormalWallSeconds": sparse_wall,
        "cpuSparseNormalCpuSeconds": sparse_cpu,
        "resolvedSchurWallSeconds": schur_wall,
        "resolvedSchurCpuSeconds": schur_cpu,
        "pointSchurMicrobatchPoints": schur_microbatch_points,
        "pointSchurMicrobatchCount": schur_microbatch_count,
        "previousPriorMerged": previous_prior is not None,
        "reasonCodes": reasons,
    }
    return FejPriorState(
        camera_ids=retained_camera_ids,
        linearization_points=_as_tensor(
            target_points, dtype=dtype, device=device
        ),
        hessian=canonical_hessian,
        gradient=canonical_gradient,
        factor=factor,
        factor_residual=factor_residual,
        report=report,
    )
