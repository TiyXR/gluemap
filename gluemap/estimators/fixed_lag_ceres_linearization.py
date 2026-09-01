"""Capture one solved Ceres BA linearization without rebuilding the problem."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pyceres
import pycolmap
import pygluemap

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
    camera_hessian: np.ndarray | None = None
    camera_gradient: np.ndarray | None = None
    point_hessian: np.ndarray | None = None
    point_gradient: np.ndarray | None = None
    block_point_indexes: np.ndarray | None = None
    block_camera_indexes: np.ndarray | None = None
    camera_point_hessian: np.ndarray | None = None

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
    residual_seed_point3d_ids: list[int] | tuple[int, ...] | None = None,
    apply_loss_function: bool = True,
    build_normal_blocks: bool = False,
) -> CeresProblemLinearization:
    """Evaluate residual/J once with deterministic local parameter ordering."""
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
    seed_parameters = None
    if residual_seed_point3d_ids is not None:
        seed_ids = tuple(int(value) for value in residual_seed_point3d_ids)
        if (
            not seed_ids
            or len(set(seed_ids)) != len(seed_ids)
            or any(value not in resolved_point_ids for value in seed_ids)
        ):
            raise FixedLagCeresLinearizationError(
                "connected residual seed identity is invalid"
            )
        seed_parameters = [
            reconstruction.points3D[point3d_id].xyz for point3d_id in seed_ids
        ]
    return capture_explicit_ceres_problem_linearization(
        problem,
        camera_ids=camera_ids,
        image_ids=image_ids,
        point3d_ids=resolved_point_ids,
        pose_parameters=pose_parameters,
        point_parameters=point_parameters,
        residual_seed_parameters=seed_parameters,
        apply_loss_function=apply_loss_function,
        build_normal_blocks=build_normal_blocks,
    )


def capture_explicit_ceres_problem_linearization(
    problem: pyceres.Problem,
    *,
    camera_ids: tuple[int, ...],
    image_ids: tuple[int, ...],
    point3d_ids: tuple[int, ...],
    pose_parameters: list[np.ndarray],
    point_parameters: list[np.ndarray],
    residual_seed_parameters: list[np.ndarray] | None = None,
    apply_loss_function: bool = True,
    build_normal_blocks: bool = False,
    native_thread_count_override: int | None = None,
) -> CeresProblemLinearization:
    """Capture CRS from explicit stable blocks owned by a persistent problem."""
    started = time.perf_counter()
    if (
        not camera_ids
        or len(camera_ids) != len(image_ids)
        or len(camera_ids) != len(pose_parameters)
        or len(set(camera_ids)) != len(camera_ids)
        or len(set(image_ids)) != len(image_ids)
        or not point3d_ids
        or len(point3d_ids) != len(point_parameters)
        or len(set(point3d_ids)) != len(point3d_ids)
    ):
        raise FixedLagCeresLinearizationError(
            "explicit Ceres parameter identity is invalid"
        )
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

    if (
        native_thread_count_override is not None
        and (
            isinstance(native_thread_count_override, bool)
            or native_thread_count_override < 1
        )
    ):
        raise FixedLagCeresLinearizationError(
            "native linearization thread count is invalid"
        )
    native_thread_count = (
        resolve_native_thread_count()
        if native_thread_count_override is None
        else int(native_thread_count_override)
    )
    connected_residual_block_count = None
    if residual_seed_parameters is None:
        options = pyceres.EvaluateOptions()
        options.apply_loss_function = apply_loss_function
        options.num_threads = native_thread_count
        options.set_parameter_blocks(parameters)
        residuals = np.asarray(
            problem.evaluate_residuals(options), dtype=np.float64
        )
        jacobian = problem.evaluate_jacobian(options)
        row_offsets = np.asarray(jacobian.rows, dtype=np.int64)
        column_indices = np.asarray(jacobian.cols, dtype=np.int64)
        jacobian_values = np.asarray(jacobian.values, dtype=np.float64)
        jacobian_num_rows = jacobian.num_rows
        jacobian_num_cols = jacobian.num_cols
    elif not build_normal_blocks:
        if not residual_seed_parameters or any(
            not problem.has_parameter_block(value)
            or problem.is_parameter_block_constant(value)
            for value in residual_seed_parameters
        ):
            raise FixedLagCeresLinearizationError(
                "connected residual seed parameter is invalid"
            )
        evaluated = pygluemap.evaluate_connected_crs(
            problem,
            [int(np.asarray(value).ctypes.data) for value in parameters],
            [
                int(np.asarray(value).ctypes.data)
                for value in residual_seed_parameters
            ],
            apply_loss_function,
            native_thread_count,
        )
        residuals = np.asarray(evaluated["residuals"], dtype=np.float64)
        row_offsets = np.asarray(evaluated["rowOffsets"], dtype=np.int64)
        column_indices = np.asarray(
            evaluated["columnIndices"], dtype=np.int64
        )
        jacobian_values = np.asarray(
            evaluated["jacobianValues"], dtype=np.float64
        )
        jacobian_num_rows = int(evaluated["residualCount"])
        jacobian_num_cols = int(evaluated["columnCount"])
        connected_residual_block_count = int(
            evaluated["residualBlockCount"]
        )
    else:
        if not residual_seed_parameters or any(
            not problem.has_parameter_block(value)
            or problem.is_parameter_block_constant(value)
            for value in residual_seed_parameters
        ):
            raise FixedLagCeresLinearizationError(
                "connected residual seed parameter is invalid"
            )
        evaluated = pygluemap.evaluate_connected_normal_blocks(
            problem,
            [int(np.asarray(value).ctypes.data) for value in parameters],
            [
                int(np.asarray(value).ctypes.data)
                for value in residual_seed_parameters
            ],
            len(pose_parameters),
            len(point_parameters),
            apply_loss_function,
            native_thread_count,
        )
        residuals = np.empty((0,), dtype=np.float64)
        row_offsets = np.empty((0,), dtype=np.int64)
        column_indices = np.empty((0,), dtype=np.int64)
        jacobian_values = np.empty((0,), dtype=np.float64)
        jacobian_num_rows = int(evaluated["residualCount"])
        jacobian_num_cols = int(evaluated["columnCount"])
        connected_residual_block_count = int(
            evaluated["residualBlockCount"]
        )
    expected_columns = len(pose_parameters) * 6 + len(point_parameters) * 3
    if jacobian_num_cols != expected_columns:
        raise FixedLagCeresLinearizationError(
            "Ceres Jacobian column ordering differs"
        )
    if not build_normal_blocks and jacobian_num_rows != len(residuals):
        raise FixedLagCeresLinearizationError(
            "Ceres residual/Jacobian row count differs"
        )
    if (
        not build_normal_blocks
        and (
        row_offsets.shape != (len(residuals) + 1,)
        or row_offsets[0] != 0
        or row_offsets[-1] != len(jacobian_values)
        or len(column_indices) != len(jacobian_values)
        )
    ):
        raise FixedLagCeresLinearizationError("Ceres CRS layout is invalid")
    wall = time.perf_counter() - started
    report = {
        "contractId": "jarailsense.gluemap-ceres-linearization/v1",
        "status": "passed",
        "applyLossFunction": apply_loss_function,
        "cameraCount": len(camera_ids),
        "pointCount": len(point3d_ids),
        "residualCount": jacobian_num_rows,
        "columnCount": expected_columns,
        "jacobianNonzeroCount": (
            int(evaluated["jacobianNonzeroCount"])
            if build_normal_blocks
            else len(jacobian_values)
        ),
        "nativeThreadCount": native_thread_count,
        "residualSelection": (
            "all-problem-residuals"
            if residual_seed_parameters is None
            else "seed-point-connected-residuals"
        ),
        "connectedResidualBlockCount": connected_residual_block_count,
        "representation": (
            "native-normal-blocks" if build_normal_blocks else "ceres-crs"
        ),
        "nativeSelectionWallSeconds": (
            float(evaluated["selectionWallSeconds"])
            if build_normal_blocks
            else 0.0
        ),
        "nativeEvaluationWallSeconds": (
            float(evaluated["evaluationWallSeconds"])
            if build_normal_blocks
            else 0.0
        ),
        "nativeNormalBuildWallSeconds": (
            float(evaluated["normalBuildWallSeconds"])
            if build_normal_blocks
            else 0.0
        ),
        "captureWallSeconds": wall,
    }
    return CeresProblemLinearization(
        camera_ids=camera_ids,
        image_ids=image_ids,
        point3d_ids=point3d_ids,
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
        camera_hessian=(
            np.asarray(evaluated["cameraHessian"], dtype=np.float64)
            if build_normal_blocks
            else None
        ),
        camera_gradient=(
            np.asarray(evaluated["cameraGradient"], dtype=np.float64)
            if build_normal_blocks
            else None
        ),
        point_hessian=(
            np.asarray(evaluated["pointHessian"], dtype=np.float64)
            if build_normal_blocks
            else None
        ),
        point_gradient=(
            np.asarray(evaluated["pointGradient"], dtype=np.float64)
            if build_normal_blocks
            else None
        ),
        block_point_indexes=(
            np.asarray(evaluated["blockPointIndexes"], dtype=np.int64)
            if build_normal_blocks
            else None
        ),
        block_camera_indexes=(
            np.asarray(evaluated["blockCameraIndexes"], dtype=np.int64)
            if build_normal_blocks
            else None
        ),
        camera_point_hessian=(
            np.asarray(evaluated["cameraPointHessian"], dtype=np.float64)
            if build_normal_blocks
            else None
        ),
    )
