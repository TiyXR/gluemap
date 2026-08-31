"""Performance-first pose-only PnP backends and startup auto-selection."""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, Iterable

import cv2
import numpy as np
import psutil
import torch

from gluemap.estimators.cuda_pose_only_pnp import solve_pose_only_pnp_cuda


class PoseOnlyPnpError(ValueError):
    pass


@dataclass(frozen=True)
class PoseOnlyPnpResult:
    rotation_world_to_camera: np.ndarray
    translation_world_to_camera: np.ndarray
    inlier_mask: np.ndarray
    inlier_count: int
    inlier_ratio: float
    positive_depth_fraction: float
    reprojection_p95_pixels: float
    condition_estimate: float
    hypothesis_count: int
    refinement_iterations: int
    backend: str

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
            "gpuUsed": self.backend.startswith("torch-cuda"),
        }


def _numpy_inputs(
    points_world: Any, points_image: Any, intrinsics: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    world = np.ascontiguousarray(np.asarray(points_world, dtype=np.float64))
    image = np.ascontiguousarray(np.asarray(points_image, dtype=np.float64))
    camera = np.ascontiguousarray(np.asarray(intrinsics, dtype=np.float64))
    if world.ndim != 2 or world.shape[1] != 3:
        raise PoseOnlyPnpError("world points must have shape N,3")
    if image.shape != (world.shape[0], 2):
        raise PoseOnlyPnpError("image points must have shape N,2")
    if camera.shape != (3, 3):
        raise PoseOnlyPnpError("intrinsics must have shape 3,3")
    if world.shape[0] < 6:
        raise PoseOnlyPnpError("insufficient PnP correspondences")
    return world, image, camera


def solve_pose_only_pnp_opencv(
    points_world: Any,
    points_image: Any,
    intrinsics: Any,
    *,
    hypothesis_count: int = 128,
    reprojection_threshold_pixels: float = 3.0,
    confidence: float = 0.999,
    refinement_iterations: int = 8,
    random_seed: int = 0,
) -> PoseOnlyPnpResult:
    world, image, camera = _numpy_inputs(
        points_world, points_image, intrinsics
    )
    cv2.setRNGSeed(int(random_seed) & 0x7FFFFFFF)
    passed, rotation_vector, translation, inlier_indexes = cv2.solvePnPRansac(
        world,
        image,
        camera,
        None,
        iterationsCount=hypothesis_count,
        reprojectionError=reprojection_threshold_pixels,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not passed or inlier_indexes is None or len(inlier_indexes) < 6:
        raise PoseOnlyPnpError("PnP RANSAC found fewer than six inliers")
    selected = inlier_indexes[:, 0]
    if refinement_iterations > 0:
        criteria = (
            cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS,
            refinement_iterations,
            1e-7,
        )
        rotation_vector, translation = cv2.solvePnPRefineVVS(
            world[selected],
            image[selected],
            camera,
            None,
            rotation_vector,
            translation,
            criteria=criteria,
        )
    projected, jacobian = cv2.projectPoints(
        world, rotation_vector, translation, camera, None
    )
    errors = np.linalg.norm(projected[:, 0, :] - image, axis=1)
    rotation, _ = cv2.Rodrigues(rotation_vector)
    depths = ((rotation @ world.T) + translation.reshape(3, 1))[2]
    inliers = (errors <= reprojection_threshold_pixels) & (depths > 0)
    if int(inliers.sum()) < 6:
        raise PoseOnlyPnpError("refined PnP pose has fewer than six inliers")
    pose_jacobian = jacobian.reshape(world.shape[0], 2, -1)[
        inliers, :, :6
    ].reshape(-1, 6)
    normal = pose_jacobian.T @ pose_jacobian
    condition = float(np.linalg.cond(normal))
    return PoseOnlyPnpResult(
        rotation_world_to_camera=rotation,
        translation_world_to_camera=translation[:, 0],
        inlier_mask=inliers,
        inlier_count=int(inliers.sum()),
        inlier_ratio=float(inliers.mean()),
        positive_depth_fraction=float((depths > 0).mean()),
        reprojection_p95_pixels=float(np.quantile(errors[inliers], 0.95)),
        condition_estimate=condition,
        hypothesis_count=hypothesis_count,
        refinement_iterations=refinement_iterations,
        backend="opencv-cpu-epnp-ransac-vvs/v1",
    )


def _solve_cuda_as_numpy(
    points_world: Any, points_image: Any, intrinsics: Any, **kwargs: Any
) -> PoseOnlyPnpResult:
    value = solve_pose_only_pnp_cuda(
        torch.as_tensor(points_world),
        torch.as_tensor(points_image),
        torch.as_tensor(intrinsics),
        **kwargs,
    )
    return PoseOnlyPnpResult(
        rotation_world_to_camera=value.rotation_world_to_camera.cpu().numpy(),
        translation_world_to_camera=value.translation_world_to_camera.cpu().numpy(),
        inlier_mask=value.inlier_mask.cpu().numpy(),
        inlier_count=value.inlier_count,
        inlier_ratio=value.inlier_ratio,
        positive_depth_fraction=value.positive_depth_fraction,
        reprojection_p95_pixels=value.reprojection_p95_pixels,
        condition_estimate=value.condition_estimate,
        hypothesis_count=value.hypothesis_count,
        refinement_iterations=value.refinement_iterations,
        backend=value.backend,
    )


def select_pose_only_pnp_backend(
    points_world: Any,
    points_image: Any,
    intrinsics: Any,
    *,
    policy: str = "auto-benchmark",
    benchmark_calls: int = 3,
    solver_kwargs: dict[str, Any] | None = None,
) -> tuple[Callable[..., PoseOnlyPnpResult], dict[str, Any]]:
    """Benchmark eligible backends on startup and return the fastest one."""
    if policy not in {"auto-benchmark", "cpu", "cuda-required"}:
        raise PoseOnlyPnpError("pose-only PnP backend policy is unsupported")
    kwargs = dict(solver_kwargs or {})
    candidates: list[tuple[str, Callable[..., PoseOnlyPnpResult]]] = []
    if policy in {"auto-benchmark", "cpu"}:
        candidates.append(("opencv-cpu", solve_pose_only_pnp_opencv))
    if policy in {"auto-benchmark", "cuda-required"}:
        if not torch.cuda.is_available():
            if policy == "cuda-required":
                raise PoseOnlyPnpError("CUDA-required PnP backend is unavailable")
        else:
            candidates.append(("torch-cuda", _solve_cuda_as_numpy))
    timings = []
    for name, solver in candidates:
        solver(points_world, points_image, intrinsics, **kwargs)
        started = time.perf_counter()
        for _ in range(benchmark_calls):
            solver(points_world, points_image, intrinsics, **kwargs)
        if name == "torch-cuda":
            torch.cuda.synchronize()
        wall = time.perf_counter() - started
        timings.append((wall / benchmark_calls, name, solver))
    if not timings:
        raise PoseOnlyPnpError("no eligible pose-only PnP backend")
    timings.sort(key=lambda value: value[0])
    best_seconds, best_name, best_solver = timings[0]
    report = {
        "contractId": "jarailsense.gluemap-pose-only-pnp-selection/v1",
        "status": "passed",
        "policy": policy,
        "selectedBackend": best_name,
        "secondsPerFrame": best_seconds,
        "candidates": [
            {"backend": name, "secondsPerFrame": seconds}
            for seconds, name, _ in timings
        ],
    }
    return best_solver, report


def dynamic_pose_worker_count(
    *, estimated_bytes_per_worker: int = 64 * 1024 * 1024
) -> int:
    logical = os.cpu_count() or 1
    cpu_limit = max(1, math.floor(logical * 0.95))
    available = psutil.virtual_memory().available
    memory_limit = max(1, math.floor(available * 0.90 / estimated_bytes_per_worker))
    return min(cpu_limit, memory_limit)


def solve_pose_batch_parallel(
    correspondences: Iterable[tuple[Any, Any, Any, dict[str, Any]]],
    solver: Callable[..., PoseOnlyPnpResult],
    *,
    maximum_in_flight: int,
) -> list[PoseOnlyPnpResult]:
    values = list(correspondences)
    workers = min(dynamic_pose_worker_count(), maximum_in_flight, len(values))
    if workers < 1:
        return []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(solver, world, image, camera, **kwargs) for world, image, camera, kwargs in values]
        return [future.result() for future in futures]
