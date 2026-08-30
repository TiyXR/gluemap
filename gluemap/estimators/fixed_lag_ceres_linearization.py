"""Capture one solved Ceres BA linearization without rebuilding the problem."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyceres
import pycolmap

from gluemap.utils.runtime_capacity import resolve_native_thread_count


class FixedLagCeresLinearizationError(ValueError):
    """Raised when Ceres parameter ordering cannot be captured exactly."""


@dataclass(frozen=True)
class CeresProblemLinearization:
    """CRS Jacobian and residuals in explicit pose-then-point ordering."""

    camera_ids: tuple[int, ...]
    image_ids: tuple[int, ...]
    point3d_ids: tuple[int, ...]
    pose_ambient_values: np.ndarray
    point_values: np.ndarray
    residuals: np.ndarray
    row_offsets: np.ndarray
    column_indices: np.ndarray
    jacobian_values: np.ndarray
    report: dict[str, Any]

    @property
    def pose_tangent_dimension(self) -> int:
        return len(self.camera_ids) * 6

    @property
    def point_dimension(self) -> int:
        return len(self.point3d_ids) * 3


def capture_ceres_problem_linearization(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    image_id_by_camera_id: dict[int, int],
    *,
    point3d_ids: list[int] | tuple[int, ...] | None = None,
    apply_loss_function: bool = True,
) -> CeresProblemLinearization:
    """Evaluate residual/J once with deterministic local parameter ordering."""
    started = time.perf_counter()
    camera_ids = tuple(int(value) for value in image_id_by_camera_id)
    image_ids = tuple(int(image_id_by_camera_id[value]) for value in camera_ids)
    if (
        not camera_ids
        or len(set(camera_ids)) != len(camera_ids)
        or len(set(image_ids)) != len(image_ids)
        or any(image_id not in reconstruction.images for image_id in image_ids)
    ):
        raise FixedLagCeresLinearizationError("Ceres pose ordering is invalid")
    resolved_point_ids = tuple(
        sorted(reconstruction.point3D_ids())
        if point3d_ids is None
        else (int(value) for value in point3d_ids)
    )
    if (
        not resolved_point_ids
        or len(set(resolved_point_ids)) != len(resolved_point_ids)
        or any(value not in reconstruction.points3D for value in resolved_point_ids)
    ):
        raise FixedLagCeresLinearizationError("Ceres point ordering is invalid")

    pose_parameters = [
        reconstruction.frames[image_id].rig_from_world.params
        for image_id in image_ids
    ]
    point_parameters = [
        reconstruction.points3D[point3d_id].xyz
        for point3d_id in resolved_point_ids
    ]
    parameters = [*pose_parameters, *point_parameters]
    if any(not problem.has_parameter_block(value) for value in parameters):
        raise FixedLagCeresLinearizationError(
            "requested Ceres parameter block is absent"
        )
    if any(problem.is_parameter_block_constant(value) for value in parameters):
        raise FixedLagCeresLinearizationError(
            "constant Ceres parameter block entered prior ordering"
        )
    tangent_sizes = [
        problem.parameter_block_tangent_size(value) for value in parameters
    ]
    expected_sizes = [6] * len(pose_parameters) + [3] * len(point_parameters)
    if tangent_sizes != expected_sizes:
        raise FixedLagCeresLinearizationError(
            "Ceres tangent parameter layout differs"
        )

    options = pyceres.EvaluateOptions()
    options.apply_loss_function = apply_loss_function
    options.num_threads = resolve_native_thread_count()
    options.set_parameter_blocks(parameters)
    residuals = np.asarray(
        problem.evaluate_residuals(options), dtype=np.float64
    )
    jacobian = problem.evaluate_jacobian(options)
    expected_columns = len(pose_parameters) * 6 + len(point_parameters) * 3
    if jacobian.num_cols != expected_columns:
        raise FixedLagCeresLinearizationError(
            "Ceres Jacobian column ordering differs"
        )
    if jacobian.num_rows != len(residuals):
        raise FixedLagCeresLinearizationError(
            "Ceres residual/Jacobian row count differs"
        )
    row_offsets = np.asarray(jacobian.rows, dtype=np.int64)
    column_indices = np.asarray(jacobian.cols, dtype=np.int64)
    jacobian_values = np.asarray(jacobian.values, dtype=np.float64)
    if (
        row_offsets.shape != (len(residuals) + 1,)
        or row_offsets[0] != 0
        or row_offsets[-1] != len(jacobian_values)
        or len(column_indices) != len(jacobian_values)
    ):
        raise FixedLagCeresLinearizationError("Ceres CRS layout is invalid")
    wall = time.perf_counter() - started
    report = {
        "contractId": "jarailsense.gluemap-ceres-linearization/v1",
        "status": "passed",
        "applyLossFunction": apply_loss_function,
        "cameraCount": len(camera_ids),
        "pointCount": len(resolved_point_ids),
        "residualCount": len(residuals),
        "columnCount": expected_columns,
        "jacobianNonzeroCount": len(jacobian_values),
        "nativeThreadCount": options.num_threads,
        "captureWallSeconds": wall,
    }
    return CeresProblemLinearization(
        camera_ids=camera_ids,
        image_ids=image_ids,
        point3d_ids=resolved_point_ids,
        pose_ambient_values=np.stack(
            [np.asarray(value, dtype=np.float64).copy() for value in pose_parameters]
        ),
        point_values=np.stack(
            [np.asarray(value, dtype=np.float64).copy() for value in point_parameters]
        ),
        residuals=residuals,
        row_offsets=row_offsets,
        column_indices=column_indices,
        jacobian_values=jacobian_values,
        report=report,
    )
