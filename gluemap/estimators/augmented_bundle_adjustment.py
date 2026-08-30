import logging
import time
from collections.abc import Callable

import numpy as np
import pyceres
import pycolmap

from gluemap.estimators.fixed_lag_prior import FejPriorState
import pygluemap

from gluemap.utils.runtime_capacity import resolve_native_thread_count

logger = logging.getLogger(__name__)


def _update_poses_from_reconstruction(
    source_recon: pycolmap.Reconstruction,
    target_recon: pycolmap.Reconstruction,
) -> None:
    """
    Copy BA-optimized poses and camera intrinsics from source to target
    reconstruction. Matches images by name.
    """
    source_by_name = {
        img.name: (img_id, img) for img_id, img in source_recon.images.items()
    }
    for target_id, target_img in target_recon.images.items():
        if target_img.name in source_by_name:
            src_id, src_img = source_by_name[target_img.name]
            target_recon.frames[
                target_id
            ].rig_from_world = src_img.cam_from_world()
    # Copy camera intrinsics
    for cam_id, cam in source_recon.cameras.items():
        if cam_id in target_recon.cameras:
            target_recon.cameras[cam_id].params = cam.params


def _pycolmap_loss_type(name: str) -> pycolmap.LossFunctionType:
    """
    Map a loss type name to a pycolmap.LossFunctionType enum value.

    Args:
        name: One of ``"trivial"``, ``"huber"``, ``"cauchy"``.

    Returns:
        Matching ``pycolmap.LossFunctionType`` enum value.
    """
    mapping = {
        "trivial": pycolmap.LossFunctionType.TRIVIAL,
        "huber": pycolmap.LossFunctionType.HUBER,
        "cauchy": pycolmap.LossFunctionType.CAUCHY,
    }
    if name not in mapping:
        raise ValueError(
            f"Unknown loss type '{name}', "
            f"expected one of {list(mapping.keys())}"
        )
    return mapping[name]


def _pyceres_loss_function(name: str) -> pyceres.LossFunction | None:
    """
    Map a loss type name to a pyceres.LossFunction (or None for trivial).

    Args:
        name: One of ``"trivial"``, ``"huber"``, ``"arctan"``, ``"cauchy"``.

    Returns:
        Configured ``pyceres.LossFunction``, or ``None`` for the trivial
        (squared) loss.
    """
    configs = {
        "trivial": None,
        "huber": {"name": "huber", "params": [1.0], "magnitude": 1.0},
        "arctan": {"name": "arctan", "params": [5.0], "magnitude": 1.0},
        "cauchy": {"name": "cauchy", "params": [1.0], "magnitude": 1.0},
    }
    if name not in configs:
        raise ValueError(
            f"Unknown loss type '{name}', "
            f"expected one of {list(configs.keys())}"
        )
    cfg = configs[name]
    return pyceres.LossFunction(cfg) if cfg is not None else None


# Sentinel: when callers pass no explicit loss_function, fall back to Arctan.
# ``None`` itself is a valid Ceres value (trivial / squared loss), so we need
# a distinct sentinel to distinguish "caller wants trivial" from "caller wants
# the default".
_DEFAULT_LOSS = object()


def _validate_resolved_ba_backend(summary: object, gpu_requested: bool) -> None:
    if not gpu_requested:
        return
    ceres_summary = getattr(summary, "ceres_summary", summary)
    dense_library = str(ceres_summary.dense_linear_algebra_library_type)
    sparse_library = str(ceres_summary.sparse_linear_algebra_library_type)
    actual_cuda = dense_library.endswith("CUDA") or sparse_library.endswith(
        "CUDA_SPARSE"
    )
    if not actual_cuda:
        raise RuntimeError("Ceres CUDA BA request resolved to a CPU solver")


def _configure_ceres_cpu_concurrency(
    options: pycolmap.CeresBundleAdjustmentOptions,
) -> int:
    """Keep PyCOLMAP from silently serializing real fixed-lag windows.

    COLMAP defaults to one thread below 50,000 residuals.  A 16-camera
    railway window is usually just below that threshold but is repeated
    thousands of times, so the startup-resolved 95% CPU budget remains the
    authoritative worker count for every window.
    """
    thread_count = resolve_native_thread_count()
    options.solver_options.num_threads = thread_count
    options.min_num_residuals_for_cpu_multi_threading = 0
    return thread_count


def _solve_configured_bundle_adjuster(
    bundle_adjuster: object,
    ba_options: pycolmap.BundleAdjustmentOptions,
    ba_config: pycolmap.BundleAdjustmentConfig,
) -> object:
    """Solve the already-built PyCOLMAP problem with the selected Ceres.

    PyCOLMAP's embedded Ceres may not include CUDA/cuDSS even when the
    separately version-pinned ``pygluemap`` Ceres does.  The exposed Problem
    shares the pyceres ABI, so GPU policy must route that same problem through
    ``pygluemap.solve_cuda`` instead of asking PyCOLMAP to silently fall back.
    """
    if not ba_options.ceres.use_gpu:
        return bundle_adjuster.solve()
    if not hasattr(bundle_adjuster, "problem"):
        raise RuntimeError("CUDA bundle adjustment requires exposed Ceres problem")
    problem = bundle_adjuster.problem
    solver_options = ba_options.ceres.create_solver_options(ba_config, problem)
    summary = pyceres.SolverSummary()
    pygluemap.solve_cuda(solver_options, problem, summary)
    return summary


def _add_virtual_track_residuals(
    problem: pyceres.Problem,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    reference_reconstruction: pycolmap.Reconstruction,
    negative_depth_observations: dict[int, set[int]],
    loss_function: pyceres.LossFunction | None | object = _DEFAULT_LOSS,
) -> list[object]:
    """
    Add reprojection residuals for virtual tracks to an existing ceres problem.

    Pose and intrinsic parameter blocks are resolved through
    ``reference_reconstruction`` (the real reconstruction handed to
    ``pycolmap.create_default_ceres_bundle_adjuster``) so that their numpy
    buffers are the same ones pycolmap is already optimizing -- the virtual
    residuals thus contribute to the same parameter blocks rather than
    detached copies.

    Virtual points3D are read from ``virtual_reconstruction``; their xyz
    arrays become new parameter blocks in ``problem`` via the residual block.

    Args:
        problem: Ceres problem owned by the active bundle adjuster; new
            residual blocks are appended to it in place.
        virtual_reconstruction: Reconstruction whose points3D are virtual.
            ``None`` or empty is a no-op.
        reference_reconstruction: Real reconstruction whose pose and
            intrinsics buffers are already parameter blocks of ``problem``.
        negative_depth_observations: ``{image_id: {point2D_idx, ...}}``
            marking observations that should use the negative-depth cost.
        loss_function: ``pyceres.LossFunction`` to apply to virtual
            residuals, or ``None`` for the trivial (squared) loss. If left
            at the sentinel ``_DEFAULT_LOSS``, defaults to Arctan for
            backward compatibility.

    Returns:
        Strong Python references to the virtual cost functions. They must
        remain alive until the Ceres solve completes on Windows.
    """
    if (
        virtual_reconstruction is None
        or len(virtual_reconstruction.points3D) == 0
    ):
        return []

    # Default to Arctan loss for backward compatibility.
    if loss_function is _DEFAULT_LOSS:
        loss_function = _pyceres_loss_function("arctan")

    # Match virtual images to reference images by name so the function is
    # independent of the image-ID convention used by each reconstruction.
    name_to_ref_id = {
        img.name: img_id
        for img_id, img in reference_reconstruction.images.items()
    }

    num_constraints = 0
    num_negative = 0
    num_skipped = 0
    num_none = 0
    cost_handles: list[object] = []

    for point3D in virtual_reconstruction.points3D.values():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            num_none += 1
            continue

        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx

            if image_id not in virtual_reconstruction.images:
                num_skipped += 1
                continue

            image = virtual_reconstruction.images[image_id]
            ref_id = name_to_ref_id.get(image.name)
            if ref_id is None:
                num_skipped += 1
                continue

            if pt_idx >= len(image.points2D):
                num_skipped += 1
                continue
            point2D = image.points2D[pt_idx].xy

            camera_id = reference_reconstruction.images[ref_id].camera_id

            # Pose & intrinsics come from the reference reconstruction so the
            # underlying numpy buffers are shared with pycolmap's residuals.
            cam_pose = reference_reconstruction.frames[
                ref_id
            ].rig_from_world.params
            camera_params = reference_reconstruction.cameras[camera_id].params
            active_model_id = reference_reconstruction.cameras[camera_id].model

            is_negative = (
                image_id in negative_depth_observations
                and pt_idx in negative_depth_observations[image_id]
            )
            if is_negative:
                cost = pygluemap.ReprojErrorCostWithNegativeDepth(
                    active_model_id, point2D
                )
                num_negative += 1
            else:
                cost = pygluemap.ReprojErrorCost(active_model_id, point2D)

            problem.add_residual_block(
                cost,
                loss_function,
                [world_point, cam_pose, camera_params],
            )
            cost_handles.append(cost)
            num_constraints += 1

    logger.info(
        f"Added {num_constraints} virtual reprojection constraints "
        f"({num_negative} with negative depth, "
        f"{num_skipped} skipped, {num_none} with no xyz)"
    )
    return cost_handles


def _add_fej_pose_prior(
    problem: pyceres.Problem,
    reconstruction: pycolmap.Reconstruction,
    prior: FejPriorState | None,
    image_id_by_prior_camera_id: dict[int, int] | None,
) -> tuple[list[object], list[object]]:
    if prior is None:
        return [], []
    mapping = image_id_by_prior_camera_id or {}
    if set(mapping) != set(prior.camera_ids):
        raise ValueError("FEJ prior camera/image identity differs")
    image_ids = [mapping[camera_id] for camera_id in prior.camera_ids]
    if len(set(image_ids)) != len(image_ids) or any(
        image_id not in reconstruction.images for image_id in image_ids
    ):
        raise ValueError("FEJ prior image identity is invalid")
    parameters = [
        reconstruction.frames[image_id].rig_from_world.params
        for image_id in image_ids
    ]
    if any(not problem.has_parameter_block(value) for value in parameters):
        raise ValueError("FEJ prior pose is absent from the Ceres problem")
    cost = pygluemap.CreateFejPosePriorCost(
        prior.factor.detach().cpu().numpy(),
        prior.factor_residual.detach().cpu().numpy(),
        prior.linearization_points.detach().cpu().numpy(),
    )
    residual_block = problem.add_residual_block(cost, None, parameters)
    return [cost], [residual_block]


def bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    negative_depth_observations: dict[int, set[int]],
    max_num_iterations: int = 200,
    loss_type_normal: str = "huber",
    loss_type_virtual: str = "arctan",
    linear_solver_type: str = "auto",
    fixed_pose_ids: set[int] | None = None,
    fix_intrinsics: bool = False,
    device_policy: str = "cuda-preferred",
    ceres_cuda_available: bool | None = None,
    fej_prior: FejPriorState | None = None,
    fej_prior_image_ids: dict[int, int] | None = None,
    post_solve_problem_callback: (
        Callable[[pyceres.Problem, pycolmap.Reconstruction], None] | None
    ) = None,
) -> tuple[
    pycolmap.Reconstruction,
    pycolmap.Reconstruction | None,
    pyceres.SolverSummary,
]:
    """
    Bundle adjustment over real + virtual reconstructions.

    The real reconstruction is optimized via pycolmap's built-in ceres
    bundle adjuster (handles manifolds, gauge fixing, solver selection).
    Virtual residuals are appended manually to the same ceres problem via
    ``_add_virtual_track_residuals`` so that they share the pose/intrinsic
    parameter blocks with the real residuals.

    Args:
        reconstruction: pycolmap.Reconstruction holding the real tracks
            plus authoritative poses and intrinsics. Optimized in-place.
        virtual_reconstruction: pycolmap.Reconstruction whose points3D
            are virtual; may be None or empty for a pure real BA. Its
            points3D.xyz values are optimized in-place as part of the
            joint solve.
        negative_depth_observations: Dict[image_id, Set[point2D_idx]]
            marking observations that should use the negative-depth cost.
        max_num_iterations: Max Ceres iterations.
        loss_type_normal: Loss function for real tracks. One of
            ``"trivial"``, ``"huber"``, ``"cauchy"``.
        loss_type_virtual: Loss function for virtual tracks. One of
            ``"trivial"``, ``"huber"``, ``"arctan"``, ``"cauchy"``.
        fixed_pose_ids: Existing canonical window poses to keep constant.
        fix_intrinsics: Keep all current camera intrinsics constant after the
            initial-anchor calibration has been frozen.
        device_policy: ``cuda-required``, ``cuda-preferred`` or ``cpu``.
        ceres_cuda_available: Passed Ceres solver-probe result. PyTorch,
            PyCOLMAP SIFT or pygluemap CUDA capability is not sufficient
            evidence for CUDA BA.
        fej_prior: Optional persistent pose-only square-root prior.
        fej_prior_image_ids: Exact prior-camera to reconstruction-image map.
        post_solve_problem_callback: Optional in-lifetime Ceres problem
            consumer used to capture the solved local linearization once.

    Returns:
        (reconstruction, virtual_reconstruction, summary) with parameters
        updated in-place and the Ceres solver summary.
    """
    num_virtual = (
        len(virtual_reconstruction.points3D)
        if virtual_reconstruction is not None
        else 0
    )
    logger.info(
        f"Bundle adjustment: {len(reconstruction.points3D)} real tracks, "
        f"{num_virtual} virtual tracks"
    )

    if num_virtual > 0 and not hasattr(pycolmap.BundleAdjuster, "problem"):
        logger.warning(
            "PyCOLMAP does not expose BundleAdjuster.problem; "
            "dropping %d virtual tracks and running pure-real BA",
            num_virtual,
        )
        virtual_reconstruction = None
        num_virtual = 0

    # --- Build pycolmap BA over the real reconstruction --------------------
    ba_options = pycolmap.BundleAdjustmentOptions()
    # Restore stock Ceres convergence tolerances.
    ba_options.ceres.solver_options = pyceres.SolverOptions()
    ba_options.ceres.solver_options.max_num_iterations = max_num_iterations
    requested_thread_count = _configure_ceres_cpu_concurrency(ba_options.ceres)
    ba_options.ceres.auto_select_solver_type = True
    solver_types = {
        "dense-schur": pyceres.LinearSolverType.DENSE_SCHUR,
        "sparse-schur": pyceres.LinearSolverType.SPARSE_SCHUR,
        "iterative-schur": pyceres.LinearSolverType.ITERATIVE_SCHUR,
    }
    if linear_solver_type in solver_types:
        ba_options.ceres.auto_select_solver_type = False
        ba_options.ceres.solver_options.linear_solver_type = solver_types[
            linear_solver_type
        ]
    elif linear_solver_type != "auto":
        raise ValueError(f"Unsupported BA solver: {linear_solver_type}")
    if device_policy not in {"cuda-required", "cuda-preferred", "cpu"}:
        raise ValueError("Unsupported BA device policy")
    cuda_available = ceres_cuda_available is True
    if device_policy == "cuda-required" and not cuda_available:
        raise RuntimeError("CUDA/cuDSS bundle adjustment is unavailable")
    ba_options.ceres.use_gpu = device_policy != "cpu" and cuda_available
    ba_options.ceres.loss_function_type = _pycolmap_loss_type(loss_type_normal)

    ba_config = pycolmap.BundleAdjustmentConfig()
    for image_id in reconstruction.images:
        ba_config.add_image(image_id)
    for point3D_id in reconstruction.points3D:
        ba_config.add_variable_point(point3D_id)
    fixed_poses = set(fixed_pose_ids or set())
    unknown_fixed = fixed_poses - set(reconstruction.images)
    if unknown_fixed:
        raise ValueError("Fixed BA pose is absent from the reconstruction")
    if fixed_poses:
        for image_id in sorted(fixed_poses):
            ba_config.set_constant_rig_from_world_pose(image_id)
    else:
        ba_config.fix_gauge(pycolmap.BundleAdjustmentGauge.TWO_CAMS_FROM_WORLD)
    if fix_intrinsics:
        for camera_id in reconstruction.cameras:
            ba_config.set_constant_cam_intrinsics(camera_id)

    bundle_adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        ba_options, ba_config, reconstruction
    )
    logger.info(
        "Ceres BA concurrency: requested=%d, residual threshold=%d",
        requested_thread_count,
        ba_options.ceres.min_num_residuals_for_cpu_multi_threading,
    )

    if num_virtual == 0 and fej_prior is None and not hasattr(
        bundle_adjuster, "problem"
    ):
        logger.info(
            "PyCOLMAP does not expose BundleAdjuster.problem; "
            "using the public pure-real bundle adjustment solver"
        )
        summary = bundle_adjuster.solve()
        logger.info(str(summary))
        _validate_resolved_ba_backend(summary, ba_options.ceres.use_gpu)
        return reconstruction, virtual_reconstruction, summary

    problem = bundle_adjuster.problem

    logger.info(
        f"After pycolmap BA construction: "
        f"{problem.num_residual_blocks()} residual blocks, "
        f"{problem.num_parameter_blocks()} parameter blocks, "
        f"{problem.num_residuals()} residuals"
    )

    # --- Append virtual residuals to the same problem ----------------------
    virtual_cost_handles = _add_virtual_track_residuals(
        problem,
        virtual_reconstruction=virtual_reconstruction,
        reference_reconstruction=reconstruction,
        negative_depth_observations=negative_depth_observations,
        loss_function=_pyceres_loss_function(loss_type_virtual),
    )
    fej_cost_handles, fej_residual_blocks = _add_fej_pose_prior(
        problem,
        reconstruction,
        fej_prior,
        fej_prior_image_ids,
    )

    logger.info(
        f"After virtual residual add: "
        f"{problem.num_residual_blocks()} residual blocks, "
        f"{problem.num_parameter_blocks()} parameter blocks, "
        f"{problem.num_residuals()} residuals"
    )
    logger.info(
        "Holding %d virtual and %d FEJ prior costs through solve",
        len(virtual_cost_handles),
        len(fej_cost_handles),
    )

    # --- Solve -------------------------------------------------------------
    summary = _solve_configured_bundle_adjuster(
        bundle_adjuster, ba_options, ba_config
    )
    logger.info(str(summary))

    _validate_resolved_ba_backend(summary, ba_options.ceres.use_gpu)
    if post_solve_problem_callback is not None:
        # The next fixed-lag prior is formed from visual residuals and merges
        # ``previous_prior`` exactly once in the GPU Schur stage.  Remove the
        # Python FEJ block before parallel CRS evaluation: retaining it here
        # both double-counts history and drives Ceres worker threads through
        # the Python GIL.
        for residual_block in fej_residual_blocks:
            problem.remove_residual_block(residual_block)
        callback_started = time.perf_counter()
        post_solve_problem_callback(problem, reconstruction)
        logger.info(
            "Post-solve Ceres problem callback: %.6fs",
            time.perf_counter() - callback_started,
        )

    # --- Sync poses/intrinsics into the virtual reconstruction -------------
    # Only the real reconstruction's numpy buffers flowed into the ceres
    # problem (see ``_add_virtual_track_residuals``); the virtual
    # reconstruction still holds the pre-solve values. Copy optimized
    # poses and per-camera intrinsics over so downstream consumers
    # reading from virtual_reconstruction observe consistent state.
    if virtual_reconstruction is not None:
        # Lazy import to avoid a circular estimators -> controllers import
        # at module load time.

        _update_poses_from_reconstruction(
            reconstruction, virtual_reconstruction
        )

    return reconstruction, virtual_reconstruction, summary
