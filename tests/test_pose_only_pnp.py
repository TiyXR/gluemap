from __future__ import annotations

import numpy as np
import torch

from gluemap.estimators.pose_only_pnp import (
    dynamic_pose_worker_count,
    select_pose_only_pnp_backend,
    solve_pose_batch_parallel,
    solve_pose_only_pnp_opencv,
)


def fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13)
    world = rng.random((256, 3))
    world[:, :2] = (world[:, :2] - 0.5) * 6.0
    world[:, 2] = world[:, 2] * 8.0 + 12.0
    camera = np.array(
        [[1200.0, 0.0, 960.0], [0.0, 1180.0, 540.0], [0.0, 0.0, 1.0]]
    )
    translation = np.array([0.4, -0.2, 1.0])
    points_camera = world + translation
    image = np.stack(
        (
            camera[0, 0] * points_camera[:, 0] / points_camera[:, 2] + camera[0, 2],
            camera[1, 1] * points_camera[:, 1] / points_camera[:, 2] + camera[1, 2],
        ),
        axis=-1,
    )
    image += rng.normal(0.0, 0.15, image.shape)
    image[:32] = rng.random((32, 2)) * np.array([1920.0, 1080.0])
    return world, image, camera


def test_opencv_pose_only_pnp_and_parallel_batch() -> None:
    world, image, camera = fixture()
    value = solve_pose_only_pnp_opencv(
        world, image, camera, reprojection_threshold_pixels=2.0
    )
    assert value.inlier_count >= 220
    assert value.reprojection_p95_pixels < 0.5
    batch = solve_pose_batch_parallel(
        [(world, image, camera, {"reprojection_threshold_pixels": 2.0})] * 4,
        solve_pose_only_pnp_opencv,
        maximum_in_flight=4,
    )
    assert len(batch) == 4
    assert dynamic_pose_worker_count() >= 1


def test_auto_backend_benchmarks_real_cpu_and_cuda_candidates() -> None:
    world, image, camera = fixture()
    _, report = select_pose_only_pnp_backend(
        world,
        image,
        camera,
        benchmark_calls=1,
        solver_kwargs={"hypothesis_count": 64, "reprojection_threshold_pixels": 2.0},
    )
    assert report["status"] == "passed"
    assert {value["backend"] for value in report["candidates"]} >= {"opencv-cpu"}
    if torch.cuda.is_available():
        assert {value["backend"] for value in report["candidates"]} == {
            "opencv-cpu",
            "torch-cuda",
        }
