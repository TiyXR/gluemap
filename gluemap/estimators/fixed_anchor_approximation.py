"""Warm-start coarse window solves in one canonical fixed-anchor gauge."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from gluemap.controllers.global_merger import GlobalGluer
from gluemap.estimators.rotation_averaging import rotation_averaging
from gluemap.estimators.similarity_averaging import similarity_averaging
from gluemap.math.mst_initialization import initialize_mst_structures
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
        sequential_edges: set[tuple[int, int]] | None = None,
        camera_model: str = "SIMPLE_PINHOLE",
        incremental_entering_pose: bool = False,
        minimum_incremental_edges: int = 2,
        maximum_incremental_rotation_residual_degrees: float = 5.0,
        incremental_center_condition_limit: float = 1e6,
        incremental_center_regularization: float = 1e-3,
        incremental_full_solve_interval: int | None = None,
    ) -> None:
        if not 0 <= valid_pose_threshold <= 1:
            raise FixedAnchorApproximationError("valid pose threshold is invalid")
        if sequential_neighbor_distance < 1:
            raise FixedAnchorApproximationError(
                "sequential neighbor distance is invalid"
            )
        self.valid_pose_threshold = valid_pose_threshold
        self.sequential_neighbor_distance = sequential_neighbor_distance
        self.sequential_edges = (
            {
                (min(first, second), max(first, second))
                for first, second in sequential_edges
            }
            if sequential_edges is not None
            else None
        )
        self.camera_model = camera_model
        if minimum_incremental_edges < 2:
            raise FixedAnchorApproximationError(
                "minimum incremental edge count is invalid"
            )
        if maximum_incremental_rotation_residual_degrees <= 0:
            raise FixedAnchorApproximationError(
                "incremental rotation residual threshold is invalid"
            )
        if incremental_center_condition_limit <= 1:
            raise FixedAnchorApproximationError(
                "incremental center condition limit is invalid"
            )
        if incremental_center_regularization <= 0:
            raise FixedAnchorApproximationError(
                "incremental center regularization is invalid"
            )
        if (
            incremental_full_solve_interval is not None
            and incremental_full_solve_interval < 2
        ):
            raise FixedAnchorApproximationError(
                "incremental full solve interval is invalid"
            )
        self.incremental_entering_pose = bool(incremental_entering_pose)
        self.minimum_incremental_edges = int(minimum_incremental_edges)
        self.maximum_incremental_rotation_residual_degrees = float(
            maximum_incremental_rotation_residual_degrees
        )
        self.incremental_center_condition_limit = float(
            incremental_center_condition_limit
        )
        self.incremental_center_regularization = float(
            incremental_center_regularization
        )
        self.incremental_full_solve_interval = (
            None
            if incremental_full_solve_interval is None
            else int(incremental_full_solve_interval)
        )
        self._cached_intrinsics: list[np.ndarray | None] | None = None
        self._incremental_windows_since_full_solve = 0

    @staticmethod
    def _copy_intrinsics(
        values: list[np.ndarray | None],
    ) -> list[np.ndarray | None]:
        return [
            None if value is None else np.asarray(value).copy()
            for value in values
        ]

    def _solve_incremental_entering_pose(
        self,
        predictions: dict[str, list[Any]],
        sliced: dict[str, list[Any]],
        global_to_local: dict[int, int],
        frame_ids: list[int],
        initial_rotations: dict[int, np.ndarray],
        initial_centers: dict[int, np.ndarray],
        fixed_pose_ids: set[int],
        entering_frame_id: int,
    ) -> FixedAnchorWindowSolution | None:
        """Estimate one entering pose from all fixed-to-entering star edges."""
        local_to_global = {
            local_id: global_id
            for global_id, local_id in global_to_local.items()
        }
        gluer_args = argparse.Namespace(
            valid_pose_threshold=self.valid_pose_threshold,
            is_sequential=True,
            use_ceres_rotation_averaging=True,
        )
        gluer = GlobalGluer(gluer_args)
        gluer.N = len(frame_ids)
        if self.sequential_edges is None:
            gluer.sequential_edges = {
                (first_local, second_local)
                for first_local, first_global in local_to_global.items()
                for second_local, second_global in local_to_global.items()
                if first_local < second_local
                and second_global - first_global
                <= self.sequential_neighbor_distance
            }
        else:
            gluer.sequential_edges = {
                (global_to_local[first], global_to_local[second])
                for first, second in self.sequential_edges
                if first in global_to_local and second in global_to_local
            }
        sliced, valid_edges = gluer._refine_graph_structure(sliced)
        gluer._boost_sequential_edges(sliced, boost_factor=2.0)

        edge_records: list[tuple[int, bool, torch.Tensor, torch.Tensor, torch.Tensor]] = []
        frame_id_set = set(frame_ids)
        for star_index, indexes in enumerate(sliced["indexes"]):
            center_id = local_to_global[int(indexes[0])]
            if center_id not in frame_id_set:
                continue
            scores = sliced["pose_scores"][star_index][0]
            extrinsics = sliced["extrinsics"][star_index][0]
            for member_index in range(1, len(indexes)):
                member_id = local_to_global[int(indexes[member_index])]
                if member_id not in frame_id_set:
                    continue
                score = scores[member_index]
                if float(score.item()) <= self.valid_pose_threshold:
                    continue
                center_is_entering = center_id == entering_frame_id
                member_is_entering = member_id == entering_frame_id
                if center_is_entering == member_is_entering:
                    continue
                known_id = member_id if center_is_entering else center_id
                if known_id not in fixed_pose_ids:
                    continue
                edge_records.append(
                    (
                        known_id,
                        center_is_entering,
                        extrinsics[member_index, :3, :3],
                        extrinsics[member_index, :3, 3],
                        score,
                    )
                )
        if len(edge_records) < self.minimum_incremental_edges:
            return None

        source_device = edge_records[0][2].device
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else source_device
        )
        dtype = torch.float64
        weights = torch.stack([value[4] for value in edge_records]).to(
            device=device, dtype=dtype
        )
        weights = weights.clamp_min(torch.finfo(dtype).eps)
        weights /= weights.sum()
        relative_rotations = torch.stack([value[2] for value in edge_records]).to(
            device=device, dtype=dtype
        )
        known_rotations = torch.stack(
            [
                torch.as_tensor(
                    initial_rotations[value[0]], device=device, dtype=dtype
                )
                for value in edge_records
            ]
        )
        center_is_entering = torch.tensor(
            [value[1] for value in edge_records],
            device=device,
            dtype=torch.bool,
        )
        rotation_candidates = torch.where(
            center_is_entering[:, None, None],
            relative_rotations.transpose(1, 2) @ known_rotations,
            relative_rotations @ known_rotations,
        )
        moment = (weights[:, None, None] * rotation_candidates).sum(dim=0)
        left, _, right_h = torch.linalg.svd(moment)
        correction = torch.eye(3, device=device, dtype=dtype)
        correction[-1, -1] = torch.where(
            torch.det(left @ right_h) < 0,
            torch.tensor(-1.0, device=device, dtype=dtype),
            torch.tensor(1.0, device=device, dtype=dtype),
        )
        entering_rotation = left @ correction @ right_h
        rotation_delta = rotation_candidates @ entering_rotation.transpose(0, 1)
        rotation_cosines = (
            torch.diagonal(rotation_delta, dim1=1, dim2=2).sum(dim=1) - 1.0
        ) / 2.0
        maximum_rotation_residual_degrees = float(
            torch.rad2deg(
                torch.acos(rotation_cosines.clamp(-1.0, 1.0))
            ).max().item()
        )
        if (
            maximum_rotation_residual_degrees
            > self.maximum_incremental_rotation_residual_degrees
        ):
            return None
        rotation_refinement_started = time.perf_counter()
        local_rotations = {
            global_to_local[frame_id]: (
                entering_rotation.cpu().numpy()
                if frame_id == entering_frame_id
                else np.asarray(initial_rotations[frame_id]).copy()
            )
            for frame_id in frame_ids
        }
        local_fixed_pose_ids = {
            global_to_local[frame_id] for frame_id in fixed_pose_ids
        }
        local_rotations = rotation_averaging(
            sliced,
            init_rotations=local_rotations,
            fixed_rotation_ids=local_fixed_pose_ids,
            num_threads=1,
        )
        gluer._filter_invalid_edges(sliced, local_rotations)
        gluer._prune_invisible_pairs(sliced)
        entering_rotation = torch.as_tensor(
            local_rotations[global_to_local[entering_frame_id]],
            device=device,
            dtype=dtype,
        )
        rotation_refinement_wall = (
            time.perf_counter() - rotation_refinement_started
        )

        metric_center_candidates: list[torch.Tensor] = []
        metric_center_weights: list[torch.Tensor] = []
        for star_index, indexes in enumerate(sliced["indexes"]):
            center_id = local_to_global[int(indexes[0])]
            if center_id not in frame_id_set:
                continue
            scores = sliced["pose_scores"][star_index][0].to(
                device=device, dtype=dtype
            )
            extrinsics = sliced["extrinsics"][star_index][0].to(
                device=device, dtype=dtype
            )
            if center_id in fixed_pose_ids and entering_frame_id in indexes:
                center = torch.as_tensor(
                    initial_centers[center_id], device=device, dtype=dtype
                )
                scale_numerator = torch.zeros((), device=device, dtype=dtype)
                scale_denominator = torch.zeros((), device=device, dtype=dtype)
                entering_member_index = None
                for member_index in range(1, len(indexes)):
                    member_id = local_to_global[int(indexes[member_index])]
                    score = scores[member_index]
                    if float(score.item()) <= self.valid_pose_threshold:
                        continue
                    if member_id == entering_frame_id:
                        entering_member_index = member_index
                        continue
                    if member_id not in fixed_pose_ids:
                        continue
                    member_rotation = torch.as_tensor(
                        initial_rotations[member_id],
                        device=device,
                        dtype=dtype,
                    )
                    predicted_displacement = -(
                        member_rotation.transpose(0, 1)
                        @ extrinsics[member_index, :3, 3]
                    )
                    known_displacement = torch.as_tensor(
                        initial_centers[member_id],
                        device=device,
                        dtype=dtype,
                    ) - center
                    scale_numerator += score * torch.dot(
                        predicted_displacement, known_displacement
                    )
                    scale_denominator += score * torch.dot(
                        known_displacement, known_displacement
                    )
                if (
                    entering_member_index is not None
                    and float(scale_denominator.item())
                    > torch.finfo(dtype).eps
                ):
                    scale = scale_numerator / scale_denominator
                    if float(scale.item()) > torch.finfo(dtype).eps:
                        entering_translation = extrinsics[
                            entering_member_index, :3, 3
                        ]
                        entering_displacement = -(
                            entering_rotation.transpose(0, 1)
                            @ entering_translation
                        ) / scale
                        metric_center_candidates.append(
                            center + entering_displacement
                        )
                        metric_center_weights.append(
                            scores[entering_member_index]
                        )
            elif center_id == entering_frame_id:
                known_centers_for_star: list[torch.Tensor] = []
                predicted_displacements: list[torch.Tensor] = []
                known_weights: list[torch.Tensor] = []
                for member_index in range(1, len(indexes)):
                    member_id = local_to_global[int(indexes[member_index])]
                    score = scores[member_index]
                    if (
                        member_id not in fixed_pose_ids
                        or float(score.item()) <= self.valid_pose_threshold
                    ):
                        continue
                    member_rotation = torch.as_tensor(
                        initial_rotations[member_id],
                        device=device,
                        dtype=dtype,
                    )
                    known_centers_for_star.append(
                        torch.as_tensor(
                            initial_centers[member_id],
                            device=device,
                            dtype=dtype,
                        )
                    )
                    predicted_displacements.append(
                        -(
                            member_rotation.transpose(0, 1)
                            @ extrinsics[member_index, :3, 3]
                        )
                    )
                    known_weights.append(score)
                if len(known_centers_for_star) >= 2:
                    star_centers = torch.stack(known_centers_for_star)
                    star_displacements = torch.stack(predicted_displacements)
                    star_weights = torch.stack(known_weights)
                    star_weights /= star_weights.sum()
                    center_mean = (
                        star_weights[:, None] * star_centers
                    ).sum(dim=0)
                    displacement_mean = (
                        star_weights[:, None] * star_displacements
                    ).sum(dim=0)
                    centered_centers = star_centers - center_mean
                    centered_displacements = (
                        star_displacements - displacement_mean
                    )
                    scale_denominator = (
                        star_weights[:, None]
                        * centered_centers
                        * centered_centers
                    ).sum()
                    if (
                        float(scale_denominator.item())
                        > torch.finfo(dtype).eps
                    ):
                        scale = (
                            star_weights[:, None]
                            * centered_centers
                            * centered_displacements
                        ).sum() / scale_denominator
                        if float(scale.item()) > torch.finfo(dtype).eps:
                            metric_center_candidates.append(
                                center_mean - displacement_mean / scale
                            )
                            metric_center_weights.append(
                                torch.stack(known_weights).sum()
                            )

        translations = torch.stack([value[3] for value in edge_records]).to(
            device=device, dtype=dtype
        )
        world_directions = torch.empty_like(translations)
        member_entering_mask = ~center_is_entering
        if bool(member_entering_mask.any()):
            world_directions[member_entering_mask] = -(
                entering_rotation.transpose(0, 1)
                @ translations[member_entering_mask, :, None]
            ).squeeze(2)
        if bool(center_is_entering.any()):
            world_directions[center_is_entering] = -(
                known_rotations[center_is_entering].transpose(1, 2)
                @ translations[center_is_entering, :, None]
            ).squeeze(2)
        direction_norms = torch.linalg.vector_norm(world_directions, dim=1)
        valid_directions = direction_norms > torch.finfo(dtype).eps
        if int(valid_directions.sum().item()) < self.minimum_incremental_edges:
            return None
        weights = weights[valid_directions]
        weights /= weights.sum()
        directions = world_directions[valid_directions]
        directions /= direction_norms[valid_directions, None]
        known_centers = torch.stack(
            [
                torch.as_tensor(
                    initial_centers[edge_records[index][0]],
                    device=device,
                    dtype=dtype,
                )
                for index, valid in enumerate(valid_directions.tolist())
                if valid
            ]
        )
        identity = torch.eye(3, device=device, dtype=dtype)
        projectors = identity[None] - directions[:, :, None] * directions[:, None, :]
        normal = (weights[:, None, None] * projectors).sum(dim=0)
        gradient = (
            weights[:, None]
            * (projectors @ known_centers[:, :, None]).squeeze(2)
        ).sum(dim=0)
        eigenvalues = torch.linalg.eigvalsh(normal)
        maximum_eigenvalue = eigenvalues[-1].clamp_min(torch.finfo(dtype).eps)
        condition = float((maximum_eigenvalue / eigenvalues[0].clamp_min(0)).item())
        body_ids = sorted(fixed_pose_ids)
        predicted_center = torch.as_tensor(
            initial_centers[body_ids[-1]], device=device, dtype=dtype
        )
        if len(body_ids) >= 2:
            predicted_center = predicted_center + (
                predicted_center
                - torch.as_tensor(
                    initial_centers[body_ids[-2]], device=device, dtype=dtype
                )
            )
        regularization_applied = (
            not np.isfinite(condition)
            or condition > self.incremental_center_condition_limit
        )
        if regularization_applied:
            regularization = (
                maximum_eigenvalue * self.incremental_center_regularization
            )
            normal = normal + regularization * identity
            gradient = gradient + regularization * predicted_center
        entering_center = torch.linalg.solve(normal, gradient)
        metric_center_spread = 0.0
        if metric_center_candidates:
            candidates = torch.stack(metric_center_candidates)
            candidate_weights = torch.stack(metric_center_weights)
            candidate_weights /= candidate_weights.sum()
            robust_center = (
                candidate_weights[:, None] * candidates
            ).sum(dim=0)
            if len(metric_center_candidates) > 2:
                residuals = torch.linalg.vector_norm(
                    candidates - robust_center, dim=1
                )
                median_residual = residuals.median().clamp_min(
                    torch.finfo(dtype).eps
                )
                robust_weights = candidate_weights / torch.maximum(
                    residuals, median_residual
                )
                robust_weights /= robust_weights.sum()
                robust_center = (
                    robust_weights[:, None] * candidates
                ).sum(dim=0)
            metric_center_spread = float(
                torch.linalg.vector_norm(
                    candidates - robust_center, dim=1
                ).max().item()
            )
            entering_center = robust_center
        center_refinement_started = time.perf_counter()
        local_centers = {
            global_to_local[frame_id]: (
                entering_center.cpu().numpy()
                if frame_id == entering_frame_id
                else np.asarray(initial_centers[frame_id]).copy()
            )
            for frame_id in frame_ids
        }
        _, initial_scales = initialize_mst_structures(
            sliced, local_rotations
        )
        refined_local_centers = similarity_averaging(
            sliced,
            local_rotations,
            global_centers=local_centers,
            global_scales=initial_scales,
            max_num_iterations=50,
            fixed_center_ids={
                global_to_local[frame_id] for frame_id in fixed_pose_ids
            },
        )
        entering_center = torch.as_tensor(
            refined_local_centers[global_to_local[entering_frame_id]],
            device=device,
            dtype=dtype,
        )
        center_refinement_wall = (
            time.perf_counter() - center_refinement_started
        )
        line_residuals = torch.linalg.vector_norm(
            (projectors @ (entering_center - known_centers)[:, :, None]).squeeze(2),
            dim=1,
        )

        rotations = {
            frame_id: np.asarray(initial_rotations[frame_id]).copy()
            for frame_id in fixed_pose_ids
        }
        centers = {
            frame_id: np.asarray(initial_centers[frame_id]).copy()
            for frame_id in fixed_pose_ids
        }
        rotations[entering_frame_id] = entering_rotation.cpu().numpy()
        centers[entering_frame_id] = entering_center.cpu().numpy()
        report = {
            "contractId": "jarailsense.gluemap-fixed-anchor-window/v1",
            "status": "passed",
            "publishable": False,
            "diagnosticMode": "incremental-entering-pose",
            "backend": (
                "torch-cuda+ceres-native"
                if device.type == "cuda"
                else "torch-cpu+ceres-native"
            ),
            "nativeThreadCount": resolve_native_thread_count(),
            "frameCount": len(frame_ids),
            "firstFrameId": frame_ids[0],
            "lastFrameId": frame_ids[-1],
            "fixedPoseCount": len(fixed_pose_ids),
            "enteringFrameId": entering_frame_id,
            "validEdgeCount": len(valid_edges),
            "incrementalEdgeCount": len(edge_records),
            "maximumIncrementalRotationResidualDegrees": (
                maximum_rotation_residual_degrees
            ),
            "incrementalRotationRefinementWallSeconds": (
                rotation_refinement_wall
            ),
            "incrementalCenterConditionEstimate": condition,
            "incrementalCenterRegularizationApplied": regularization_applied,
            "incrementalCenterRefinementBackend": "ceres-native",
            "incrementalCenterRefinementWallSeconds": (
                center_refinement_wall
            ),
            "incrementalMetricScaleCandidateCount": len(
                metric_center_candidates
            ),
            "maximumIncrementalMetricCenterSpread": metric_center_spread,
            "maximumIncrementalCenterLineResidual": float(
                line_residuals.max().item()
            ),
            "intrinsicsGroupCount": len(self._cached_intrinsics or []),
            "fixedAnchorMaximumRotationMatrixDelta": 0.0,
            "fixedAnchorMaximumCenterDelta": 0.0,
        }
        return FixedAnchorWindowSolution(
            frame_ids=tuple(frame_ids),
            rotations=rotations,
            centers=centers,
            intrinsics=self._copy_intrinsics(self._cached_intrinsics or []),
            report=report,
        )

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

        entering_ids = set(frame_ids) - fixed_global
        periodic_full_solve_due = (
            self._cached_intrinsics is not None
            and self.incremental_full_solve_interval is not None
            and self._incremental_windows_since_full_solve
            >= self.incremental_full_solve_interval - 1
        )
        if (
            self.incremental_entering_pose
            and not periodic_full_solve_due
            and self._cached_intrinsics is not None
            and initial_rotations is not None
            and initial_centers is not None
            and len(entering_ids) == 1
            and fixed_global == set(frame_ids) - entering_ids
        ):
            incremental_started = time.perf_counter()
            incremental = self._solve_incremental_entering_pose(
                predictions,
                sliced,
                global_to_local,
                frame_ids,
                initial_rotations,
                initial_centers,
                fixed_global,
                next(iter(entering_ids)),
            )
            if incremental is not None:
                self._incremental_windows_since_full_solve += 1
                if self.incremental_full_solve_interval is not None:
                    incremental.report["incrementalFullSolveInterval"] = (
                        self.incremental_full_solve_interval
                    )
                incremental.report["incrementalWindowsSinceFullSolve"] = (
                    self._incremental_windows_since_full_solve
                )
                incremental.report["solveWallSeconds"] = (
                    time.perf_counter() - incremental_started
                )
                return incremental

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
        if self.sequential_edges is None:
            gluer.sequential_edges = {
                (first_local, second_local)
                for first_local, first_global in local_to_global.items()
                for second_local, second_global in local_to_global.items()
                if first_local < second_local
                and second_global - first_global
                <= self.sequential_neighbor_distance
            }
        else:
            gluer.sequential_edges = {
                (global_to_local[first], global_to_local[second])
                for first, second in self.sequential_edges
                if first in global_to_local and second in global_to_local
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
            "incrementalFullSolveInterval": (
                self.incremental_full_solve_interval
            ),
            "fullSolveReason": (
                "initial"
                if self._cached_intrinsics is None
                else (
                    "periodic-refresh"
                    if periodic_full_solve_due
                    else "incremental-confidence-fallback"
                )
            ),
        }
        self._cached_intrinsics = self._copy_intrinsics(intrinsics)
        self._incremental_windows_since_full_solve = 0
        return FixedAnchorWindowSolution(
            frame_ids=tuple(frame_ids),
            rotations=rotations,
            centers=centers,
            intrinsics=intrinsics,
            report=report,
        )
