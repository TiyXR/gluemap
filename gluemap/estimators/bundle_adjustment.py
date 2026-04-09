import numpy as np

import pygluemap
import pyceres

import pycolmap

# Re-export from colmap_utils for backwards compatibility
from gluemap.utils.colmap_utils import (
    intrinsics_to_colmap_params,
    colmap_params_to_intrinsics,
)

import logging
logger = logging.getLogger(__name__)



def add_reprojection_error(
    prob,
    costs,
    losses,
    reconstruction,
    negative_depth_observations,
    virtual_point_start,
    fisheye_intrinsics_params=None,
):
    """
    Add reprojection error for all tracks in reconstruction.points3D.

    Reads poses, intrinsics, and keypoints directly from the reconstruction.
    Caches numpy array references for pointer consistency with Ceres.

    Returns:
        first_camera_id: The ID of the first camera added to the problem (for gauge fixing)
    """
    points3D = reconstruction.points3D
    if points3D is None:
        return None

    if len(reconstruction.cameras) == 0:
        logger.warning("No cameras in reconstruction")
        return None

    fisheye_model_id = pycolmap.CameraModelId.SIMPLE_FISHEYE

    num_constraints = 0
    num_skipped = 0
    num_virtual = 0
    first_camera_id = None

    loss_huber = pyceres.LossFunction(
        {"name": "huber", "params": [1.0], "magnitude": 1.0}
    )
    loss_arctan = pyceres.LossFunction(
        {"name": "arctan", "params": [5.0], "magnitude": 1.0}
    )
    # Step 5: Add reprojection error for each track observation
    num_none = 0
    isnegative_count = 0
    for point3D_id, point3D in points3D.items():
        # Get world point from points3D.xyz (set by initialize_world_points)
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            num_none += 1
            continue
        elements = list(point3D.track.elements)

        for elem in elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx

            if image_id not in reconstruction.images:
                num_skipped += 1
                continue

            image = reconstruction.images[image_id]
            point2D = image.points2D[pt_idx].xy
            camera_id = image.camera_id

            # rig_from_world.params is a mutable numpy view into the C++ Rigid3d
            cam_pose = reconstruction.frames[image_id].rig_from_world.params
            # camera.params is a mutable numpy view into the C++ Camera
            camera_params = reconstruction.cameras[camera_id].params

            # Determine if this is a virtual point
            vp_start = virtual_point_start.get(
                image_id, len(image.points2D)
            )
            is_virtual_point = pt_idx >= vp_start

            # Virtual points use duplicate SIMPLE_FISHEYE camera (fixed params)
            if is_virtual_point and fisheye_intrinsics_params is not None:
                camera_params = fisheye_intrinsics_params[camera_id]
                active_model_id = fisheye_model_id
            else:
                active_model_id = reconstruction.cameras[camera_id].model

            # Determine if this observation has negative depth
            is_negative_depth = (
                image_id in negative_depth_observations
                and pt_idx in negative_depth_observations[image_id]
            )

            # Choose cost function based on negative depth flag
            if is_negative_depth:
                cost = pygluemap.ReprojErrorCostWithNegativeDepth(
                    active_model_id, point2D
                )
                isnegative_count += 1
            else:
                cost = pygluemap.ReprojErrorCost(active_model_id, point2D)

            # Choose loss function: Arctan for virtual points, Huber for real points
            if is_virtual_point:
                loss = loss_arctan
                num_virtual += 1
            else:
                loss = loss_huber

            prob.add_residual_block(
                cost,
                loss,
                [world_point, cam_pose, camera_params],
            )

            # Track first camera added to problem
            if first_camera_id is None:
                first_camera_id = image_id

            costs.append(cost)
            num_constraints += 1

    logger.info(
        f"Added {num_constraints} reprojection error constraints ({num_virtual} virtual)"
    )
    logger.info(f"Skipped {num_skipped} observations")
    logger.info(f"Number of None camera parameters skipped: {num_none}")
    logger.info(f"Number of negative observation: {isnegative_count}")

    return first_camera_id


def bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    negative_depth_observations,
    virtual_point_start,
    max_num_iterations: int = 200,
    fix_rotations_first_pass: bool = False,
    fisheye_intrinsics_params=None,
) -> pycolmap.Reconstruction:
    """
    Bundle adjustment with reprojection error on pycolmap.Reconstruction.

    Optimizes reconstruction.frames[].rig_from_world.params and
    reconstruction.cameras[].params directly in-place via Ceres.

    Args:
        reconstruction: pycolmap.Reconstruction with cameras, images, points3D
        negative_depth_observations: Dict[image_id, Set[point2D_idx]] for observations
            that have negative depth (use ReprojErrorCostWithNegativeDepth)
        virtual_point_start: Dict[image_id, int] indicating where virtual points
            start in each image's points2D list. Points before this index are
            real (use Huber loss), points at/after are virtual (use Arctan loss).
        max_num_iterations: Max Ceres iterations
        fix_rotations_first_pass: Fix rotations in first pass to let translations settle

    Returns:
        reconstruction: Modified in-place and returned
    """
    logger.info("Performing bundle adjustment with reprojection error...")
    logger.info(f"Bundle adjustment with {len(reconstruction.points3D)} tracks")

    # Build problem and add residuals (reads directly from reconstruction)
    prob = pyceres.Problem()
    costs = []
    losses = []

    first_camera_id = add_reprojection_error(
        prob,
        costs,
        losses,
        reconstruction,
        negative_depth_observations=negative_depth_observations,
        virtual_point_start=virtual_point_start,
        fisheye_intrinsics_params=fisheye_intrinsics_params,
    )

    # Set manifolds
    # Use product manifold for 7D pose: quaternion (4D, 3D tangent) + translation (3D, 3D tangent)
    for image_id in reconstruction.images:
        pose_params = reconstruction.frames[image_id].rig_from_world.params
        if prob.has_parameter_block(pose_params):
            pose_manifold = pygluemap.CreatePoseManifold()
            prob.set_manifold(pose_params, pose_manifold)

    for camera_id, camera in reconstruction.cameras.items():
        params = camera.params
        if prob.has_parameter_block(params):
            pp_idxs = list(camera.principal_point_idxs())
            prob.set_manifold(params, pyceres.SubsetManifold(len(params), pp_idxs))

    # Fix fisheye camera params (not optimized)
    if fisheye_intrinsics_params is not None:
        for params in fisheye_intrinsics_params:
            if params is not None and prob.has_parameter_block(params):
                prob.set_parameter_block_constant(params)

    # Fix gauge freedom
    if first_camera_id is not None:
        # Fix both rotation and translation of first camera
        first_pose = reconstruction.frames[first_camera_id].rig_from_world.params
        if prob.has_parameter_block(first_pose):
            prob.set_parameter_block_constant(first_pose)
            logger.info(
                f"Fixed rotation and translation gauge: camera {first_camera_id}"
            )

        # Fix scale: pin the largest translation component of a second camera
        second_camera_id = None
        for image_id in reconstruction.images:
            if image_id != first_camera_id:
                pose_params = reconstruction.frames[image_id].rig_from_world.params
                if prob.has_parameter_block(pose_params):
                    second_camera_id = image_id
                    break
        if second_camera_id is not None:
            # Get translation part of the pose (indices 4-6)
            second_pose = reconstruction.frames[second_camera_id].rig_from_world.params
            t2 = second_pose[4:]
            fixed_idx = int(np.argmax(np.abs(t2)))
            # Create product manifold: full quaternion manifold + subset translation manifold
            scale_gauge_manifold = pygluemap.CreatePoseManifoldWithFixedTransComponent(
                fixed_idx
            )
            prob.set_manifold(second_pose, scale_gauge_manifold)
            logger.info(
                f"Fixed scale gauge: camera {second_camera_id}, translation component {fixed_idx}"
            )
        else:
            logger.warning("No second camera available to fix scale gauge")
    else:
        logger.warning("No cameras added to problem, cannot fix gauge")

    # Solve (solver config matching COLMAP's bundle_adjustment_ceres.cc)
    options = pyceres.SolverOptions()
    options.max_num_iterations = max_num_iterations
    options.max_num_consecutive_invalid_steps = 10
    options.minimizer_progress_to_stdout = prob.num_residuals() > 5_000_000

    # Use SPARSE_NORMAL_CHOLESKY to avoid Schur ordering issues with custom manifolds
    num_images = len(reconstruction.images)
    options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY

    # TODO: SPARSE_SCHUR / DENSE_SCHUR somehow gives error.
    # Need to debug this.
    # if num_images <= 200:
    #     options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    #     # options.linear_solver_type = pyceres.LinearSolverType.DENSE_SCHUR
    # elif num_images <= 1000:
    #     options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR
    # else:
    #     options.linear_solver_type = pyceres.LinearSolverType.SPARSE_SCHUR

    # Adaptive threading based on problem size (matching COLMAP)
    if prob.num_residuals() < 50000:
        options.num_threads = 1
    else:
        options.num_threads = 32

    logger.info(
        f"Solver: {options.linear_solver_type.name} ({num_images} images, "
        f"{prob.num_residuals()} residuals)"
    )

    logger.info("Solving the optimization problem...")

    if fix_rotations_first_pass:
        # First pass: Fix quaternion part of all poses (except gauge-fixed camera)
        # Create manifold that fixes rotation (quaternion) but allows translation
        for image_id in reconstruction.images:
            if image_id != first_camera_id:
                pose_params = reconstruction.frames[image_id].rig_from_world.params
                if prob.has_parameter_block(pose_params):
                    translation_only_manifold = (
                        pygluemap.CreateTranslationOnlyManifold()
                    )
                    prob.set_manifold(pose_params, translation_only_manifold)

        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        # pygluemap.solve_cuda(options, prob, summary)
        logger.info(summary.BriefReport())

        # Release rotations for second pass - restore full pose manifold
        for image_id in reconstruction.images:
            if image_id != first_camera_id:
                pose_params = reconstruction.frames[image_id].rig_from_world.params
                if prob.has_parameter_block(pose_params):
                    # Restore appropriate manifold
                    if image_id == second_camera_id and second_camera_id is not None:
                        # Restore scale gauge manifold for second camera
                        t2 = pose_params[4:]
                        fixed_idx = int(np.argmax(np.abs(t2)))
                        scale_gauge_manifold = (
                            pygluemap.CreatePoseManifoldWithFixedTransComponent(
                                fixed_idx
                            )
                        )
                        prob.set_manifold(pose_params, scale_gauge_manifold)
                    else:
                        # Restore standard pose manifold
                        restored_manifold = pygluemap.CreatePoseManifold()
                        prob.set_manifold(pose_params, restored_manifold)

    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    logger.info(summary.BriefReport())


    return reconstruction
