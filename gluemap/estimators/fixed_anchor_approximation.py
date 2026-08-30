"""Warm-start coarse window solves in one canonical fixed-anchor gauge."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gluemap.controllers.global_merger import GlobalGluer
from gluemap.utils.runtime_capacity import resolve_native_thread_count


_MEMBER_TENSOR_KEYS = frozenset(
    {"extrinsics", "intrinsics", "pose_scores", "vis"}
)
_STAR_TENSOR_KEYS = frozenset({"points3d_virtual"})


class FixedAnchorApproximationError(ValueError):
    """Raised when a coarse window cannot preserve its declared identity."""


@dataclass(frozen=True)
class FixedAnchorWindowSolution:
    frame_ids: tuple[int, ...]
    rotations: dict[int, np.ndarray]
    centers: dict[int, np.ndarray]
    intrinsics: list[np.ndarray | None]
    report: dict[str, Any]


def slice_star_predictions(
    predictions: dict[str, list[Any]], frame_ids: list[int]
) -> tuple[dict[str, list[Any]], dict[int, int], dict[int, int]]:
    """Copy only one window and remap its global frame ids to dense ids."""
    ordered_ids = list(dict.fromkeys(frame_ids))
    if len(ordered_ids) < 3 or ordered_ids != sorted(ordered_ids):
        raise FixedAnchorApproximationError(
            "window frame ids must be sorted, unique, and contain at least three frames"
        )
    required = {"indexes", *_MEMBER_TENSOR_KEYS, *_STAR_TENSOR_KEYS}
    if any(key not in predictions for key in required):
        raise FixedAnchorApproximationError("star prediction fields are incomplete")

    global_to_local = {value: index for index, value in enumerate(ordered_ids)}
    local_to_global = {value: key for key, value in global_to_local.items()}
    star_by_center = {
        int(indexes[0]): star_index
        for star_index, indexes in enumerate(predictions["indexes"])
    }
    if any(value not in star_by_center for value in ordered_ids):
        raise FixedAnchorApproximationError("window center star is missing")

    sliced: dict[str, list[Any]] = {key: [] for key in required}
    for center in ordered_ids:
        star_index = star_by_center[center]
        members = predictions["indexes"][star_index]
        positions = [
            position
            for position, member in enumerate(members)
            if member in global_to_local
        ]
        if not positions or positions[0] != 0:
            raise FixedAnchorApproximationError("star center was not preserved")
        sliced["indexes"].append(
            [global_to_local[members[position]] for position in positions]
        )
        tensor_positions = torch.tensor(positions, dtype=torch.int64)
        for key in _MEMBER_TENSOR_KEYS:
            value = predictions[key][star_index]
            if not isinstance(value, torch.Tensor) or value.ndim < 2:
                raise FixedAnchorApproximationError(
                    f"star prediction field {key} is not a member tensor"
                )
            sliced[key].append(value.index_select(1, tensor_positions).clone())
        for key in _STAR_TENSOR_KEYS:
            value = predictions[key][star_index]
            if not isinstance(value, torch.Tensor):
                raise FixedAnchorApproximationError(
                    f"star prediction field {key} is not a tensor"
                )
            sliced[key].append(value.clone())

    return sliced, global_to_local, local_to_global


class FixedAnchorApproximationSolver:
    """Run one bounded Ceres coarse solve with canonical overlap anchors."""

    def __init__(
        self,
        *,
        valid_pose_threshold: float = 0.05,
        sequential_neighbor_distance: int = 1,
        camera_model: str = "SIMPLE_PINHOLE",
    ) -> None:
        if not 0 <= valid_pose_threshold <= 1:
            raise FixedAnchorApproximationError("valid pose threshold is invalid")
        if sequential_neighbor_distance < 1:
            raise FixedAnchorApproximationError(
                "sequential neighbor distance is invalid"
            )
        self.valid_pose_threshold = valid_pose_threshold
        self.sequential_neighbor_distance = sequential_neighbor_distance
        self.camera_model = camera_model

    def solve(
        self,
        predictions: dict[str, list[Any]],
        frame_ids: list[int],
        *,
        initial_rotations: dict[int, np.ndarray] | None = None,
        initial_centers: dict[int, np.ndarray] | None = None,
        fixed_pose_ids: set[int] | None = None,
    ) -> FixedAnchorWindowSolution:
        sliced, global_to_local, local_to_global = slice_star_predictions(
            predictions, frame_ids
        )
        fixed_global = set(fixed_pose_ids or set())
        if fixed_global - set(frame_ids):
            raise FixedAnchorApproximationError("fixed pose is outside the window")
        if fixed_global and (
            initial_rotations is None
            or initial_centers is None
            or fixed_global - set(initial_rotations)
            or fixed_global - set(initial_centers)
        ):
            raise FixedAnchorApproximationError("fixed pose warm start is incomplete")

        local_initial_rotations = (
            {
                global_to_local[key]: value.copy()
                for key, value in initial_rotations.items()
                if key in global_to_local
            }
            if initial_rotations
            else None
        )
        local_initial_centers = (
            {
                global_to_local[key]: value.copy()
                for key, value in initial_centers.items()
                if key in global_to_local
            }
            if initial_centers
            else None
        )
        local_fixed = {global_to_local[key] for key in fixed_global}
        args = argparse.Namespace(
            valid_pose_threshold=self.valid_pose_threshold,
            is_sequential=True,
            use_ceres_rotation_averaging=True,
        )
        gluer = GlobalGluer(args)
        gluer.sequential_edges = {
            (first_local, second_local)
            for first_local, first_global in local_to_global.items()
            for second_local, second_global in local_to_global.items()
            if first_local < second_local
            and second_global - first_global <= self.sequential_neighbor_distance
        }

        started = time.perf_counter()
        (
            local_rotations,
            local_centers,
            intrinsics,
            valid_edges,
            _,
        ) = gluer.main(
            sliced,
            {index: 0 for index in range(len(frame_ids))},
            self.camera_model,
            len(frame_ids),
            initial_rotations=local_initial_rotations,
            initial_centers=local_initial_centers,
            fixed_pose_ids=local_fixed,
        )
        wall = time.perf_counter() - started
        rotations = {
            local_to_global[key]: value for key, value in local_rotations.items()
        }
        centers = {
            local_to_global[key]: value for key, value in local_centers.items()
        }
        if set(rotations) != set(frame_ids) or set(centers) != set(frame_ids):
            raise FixedAnchorApproximationError("coarse window pose set is incomplete")

        fixed_rotation_delta = 0.0
        fixed_center_delta = 0.0
        for frame_id in fixed_global:
            fixed_rotation_delta = max(
                fixed_rotation_delta,
                float(
                    np.max(
                        np.abs(rotations[frame_id] - initial_rotations[frame_id])
                    )
                ),
            )
            fixed_center_delta = max(
                fixed_center_delta,
                float(np.max(np.abs(centers[frame_id] - initial_centers[frame_id]))),
            )
        report = {
            "contractId": "jarailsense.gluemap-fixed-anchor-window/v1",
            "status": "passed",
            "publishable": False,
            "diagnosticMode": "fixed-anchor-approximation",
            "backend": "ceres-native",
            "nativeThreadCount": resolve_native_thread_count(),
            "frameCount": len(frame_ids),
            "firstFrameId": frame_ids[0],
            "lastFrameId": frame_ids[-1],
            "fixedPoseCount": len(fixed_global),
            "validEdgeCount": len(valid_edges),
            "intrinsicsGroupCount": len(intrinsics),
            "solveWallSeconds": wall,
            "fixedAnchorMaximumRotationMatrixDelta": fixed_rotation_delta,
            "fixedAnchorMaximumCenterDelta": fixed_center_delta,
        }
        return FixedAnchorWindowSolution(
            frame_ids=tuple(frame_ids),
            rotations=rotations,
            centers=centers,
            intrinsics=intrinsics,
            report=report,
        )
