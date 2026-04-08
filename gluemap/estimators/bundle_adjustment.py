from collections import defaultdict
import numpy as np
import torch

import pygluemap
import pyceres

from gluemap.estimators.establish_tracks import (
    establish_tracks_from_tracks_dict,
    TrackEstablishmentOptions,
)
from gluemap.math.geometry import (
    rotation_matrix_to_quaternion,
    quaternion_to_rotation_matrix,
)

import pycolmap

# Re-export from colmap_utils for backwards compatibility
from gluemap.utils.colmap_utils import (
    intrinsics_to_colmap_params,
    colmap_params_to_intrinsics,
)

import logging
logger = logging.getLogger(__name__)


def get_camera_model_id(camera_model):
    """Get COLMAP camera model ID from string name."""
    camera_model_map = {
        "SIMPLE_PINHOLE": pycolmap.CameraModelId.SIMPLE_PINHOLE,
        "PINHOLE": pycolmap.CameraModelId.PINHOLE,
        "SIMPLE_RADIAL": pycolmap.CameraModelId.SIMPLE_RADIAL,
        "RADIAL": pycolmap.CameraModelId.RADIAL,
        "SIMPLE_FISHEYE": pycolmap.CameraModelId.SIMPLE_FISHEYE,
    }
    return camera_model_map.get(camera_model, pycolmap.CameraModelId.SIMPLE_PINHOLE)


def get_camera_model_name(camera_model_id):
    """Get camera model string name from COLMAP camera model ID."""
    camera_model_map = {
        pycolmap.CameraModelId.SIMPLE_PINHOLE: "SIMPLE_PINHOLE",
        pycolmap.CameraModelId.PINHOLE: "PINHOLE",
        pycolmap.CameraModelId.SIMPLE_RADIAL: "SIMPLE_RADIAL",
        pycolmap.CameraModelId.RADIAL: "RADIAL",
        pycolmap.CameraModelId.SIMPLE_FISHEYE: "SIMPLE_FISHEYE",
    }
    return camera_model_map.get(camera_model_id, "SIMPLE_PINHOLE")


def initialize_parameters(
    global_rotations,
    global_centers,
):

    # Build shared camera parameter dictionary (7D concatenated quat+trans)
    cam_poses = {}

    for image_id in global_rotations:
        if image_id not in global_centers:
            continue
        R = global_rotations[image_id]
        center = global_centers[image_id]

        # Convert rotation matrix to quaternion (xyzw format)
        quat_xyzw = rotation_matrix_to_quaternion(R, w_first=False)

        # cam_from_world_translation = -R @ center
        translation = (-R @ center).astype(np.float64)

        # Concatenate into 7D pose: [qx, qy, qz, qw, tx, ty, tz]
        cam_poses[image_id] = np.concatenate([quat_xyzw, translation])

    return cam_poses


def extract_optimized_parameters(
    cam_poses,
    intrinsics_params,
    camera_model="SIMPLE_PINHOLE",
):
    """
    Convert optimized Ceres parameters back to global format.

    Args:
        cam_poses: Dict[image_id, np.ndarray (7,)] - concatenated quat+trans [qx,qy,qz,qw,tx,ty,tz]
        intrinsics_params: List[np.ndarray] - camera params by type
        camera_model: Camera model string

    Returns:
        global_rotations: Dict[image_id, np.ndarray (3,3)]
        global_centers: Dict[image_id, np.ndarray (3,)]
        global_intrinsics: List of 3x3 intrinsics matrices (as torch tensors)
    """
    global_rotations = {}
    global_centers = {}

    for image_id, pose in cam_poses.items():
        # Split 7D pose into quaternion and translation
        quat_xyzw = pose[:4]
        translation = pose[4:]

        # Convert quaternion (xyzw) to rotation matrix
        R = quaternion_to_rotation_matrix(quat_xyzw, w_first=False)
        global_rotations[image_id] = R

        # Convert translation to camera center: center = -R^T @ translation
        center = -R.T @ translation
        global_centers[image_id] = center

    # Convert intrinsics params back to 3x3 matrices (as torch tensors)
    global_intrinsics = []
    for params in intrinsics_params:
        if params is not None:
            intrinsics_matrix = colmap_params_to_intrinsics(params, camera_model)
            # Convert to torch tensor and add batch dimension to match original format
            intrinsics_tensor = torch.from_numpy(intrinsics_matrix).unsqueeze(0)
            global_intrinsics.append(intrinsics_tensor)
        else:
            global_intrinsics.append(None)

    return global_rotations, global_centers, global_intrinsics


def extract_ba_data_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
):
    """
    Extract BA data from a pycolmap.Reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction with cameras, images, points3D

    Returns:
        global_rotations: Dict[image_id, np.ndarray(3,3)]
        global_centers: Dict[image_id, np.ndarray(3,)]
        intrinsics_mapping: Dict[image_id, camera_id]
        intrinsics_params: List[np.ndarray] indexed by camera_id
        keypoints_per_image: Dict[image_id, np.ndarray(N,2)]
        camera_model: str
    """
    global_rotations = {}
    global_centers = {}
    intrinsics_mapping = {}
    keypoints_per_image = {}

    # Get camera model from first camera
    camera_model = None
    intrinsics_params = []

    for camera_id, camera in reconstruction.cameras.items():
        if camera_model is None:
            camera_model = get_camera_model_name(camera.model)
        # Ensure list is long enough
        while len(intrinsics_params) <= camera_id:
            intrinsics_params.append(None)
        intrinsics_params[camera_id] = np.array(camera.params, dtype=np.float64)

    for image_id, image in reconstruction.images.items():
        # Extract rotation and center from cam_from_world
        R = np.array(image.cam_from_world().rotation.matrix())
        t = np.array(image.cam_from_world().translation)
        center = -R.T @ t

        global_rotations[image_id] = R
        global_centers[image_id] = center
        intrinsics_mapping[image_id] = image.camera_id

        # Extract 2D keypoints
        if len(image.points2D) > 0:
            keypoints = np.array([p.xy for p in image.points2D], dtype=np.float64)
        else:
            keypoints = np.zeros((0, 2), dtype=np.float64)
        keypoints_per_image[image_id] = keypoints

    return (
        global_rotations,
        global_centers,
        intrinsics_mapping,
        intrinsics_params,
        keypoints_per_image,
        camera_model,
    )


def update_reconstruction_from_ba_results(
    reconstruction: pycolmap.Reconstruction,
    cam_poses,
    intrinsics_params,
):
    """
    Update reconstruction in-place with optimized BA parameters.

    Args:
        reconstruction: pycolmap.Reconstruction to update
        cam_poses: Dict[image_id, np.ndarray(7,)] - concatenated quat+trans
        intrinsics_params: List[np.ndarray] - camera params indexed by camera_id
    """
    # Update image poses
    for image_id, pose in cam_poses.items():
        if image_id not in reconstruction.images:
            continue

        # Split 7D pose into quaternion and translation
        quat_xyzw = pose[:4]
        t = pose[4:]

        # Convert quaternion to rotation matrix
        R = quaternion_to_rotation_matrix(quat_xyzw, w_first=False)

        # Create new Rigid3d from rotation matrix and translation
        # In pycolmap 3.13+, pose is managed via frames (frame_id == image_id in trivial setup)
        reconstruction.frames[image_id].rig_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(R), t
        )

    # Update camera intrinsics
    for camera_id, params in enumerate(intrinsics_params):
        if params is not None and camera_id in reconstruction.cameras:
            reconstruction.cameras[camera_id].params = params

    # Note: points3D.xyz are already updated in-place during optimization


def add_reprojection_error(
    prob,
    costs,
    losses,
    points3D,
    keypoints_per_image,
    negative_depth_observations,
    virtual_point_start,
    cam_poses,
    intrinsics_params,
    intrinsics_mapping,
    camera_model="SIMPLE_PINHOLE",
    fisheye_intrinsics_params=None,
):
    """
    Add reprojection error for all tracks in points3D.

    For each track observation, adds a reprojection error residual using
    either ReprojErrorCost or ReprojErrorCostWithNegativeDepth based on
    the negative depth flag.

    Loss function selection:
    - Real points (pt_idx < virtual_point_start[image_id]): Huber loss
    - Virtual points (pt_idx >= virtual_point_start[image_id]): Arctan loss

    Args:
        prob: pyceres.Problem instance
        costs: List to append cost functions to
        losses: List to append loss functions to
        points3D: Established 3D point tracks (with xyz already initialized)
        keypoints_per_image: Dict[image_id, np.ndarray(N,2)] - 2D keypoints per image
        negative_depth_observations: Dict[image_id, Set[point2D_idx]] - observations
            with negative depth (use ReprojErrorCostWithNegativeDepth)
        virtual_point_start: Dict[image_id, int] - index where virtual points start
            in each image's points2D list
        cam_poses: Dict[image_id, np.ndarray(7,)] - concatenated quat+trans [qx,qy,qz,qw,tx,ty,tz]
        intrinsics_params: List[np.ndarray] - camera intrinsics params indexed by camera type
        intrinsics_mapping: Dict[image_id, camera_type_idx]
        camera_model: Camera model string (e.g., "SIMPLE_PINHOLE")

    Returns:
        first_camera_id: The ID of the first camera added to the problem (for gauge fixing)
    """
    if points3D is None:
        return None

    if cam_poses is None:
        logger.warning("cam_poses not provided")
        return None

    if intrinsics_params is None or intrinsics_mapping is None:
        logger.warning("intrinsics_params/intrinsics_mapping not provided")
        return None

    # Get camera model ID for COLMAP cost function
    camera_model_id = get_camera_model_id(camera_model)
    fisheye_model_id = pycolmap.CameraModelId.SIMPLE_FISHEYE

    num_constraints = 0
    num_skipped = 0
    num_virtual = 0
    first_camera_id = None  # Track first camera added to the problem

    # Define loss functions for real and virtual points
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

            point2D = keypoints_per_image[image_id][pt_idx].astype(np.float64)

            camera_type_idx = intrinsics_mapping[image_id]

            # Get shared camera pose (7D concatenated quat+trans)
            if image_id not in cam_poses:
                num_skipped += 1
                continue
            cam_pose = cam_poses[image_id]

            # Determine if this is a virtual point
            # Virtual points start at virtual_point_start[image_id] in the points2D list
            vp_start = virtual_point_start.get(
                image_id, len(keypoints_per_image.get(image_id, []))
            )
            is_virtual_point = pt_idx >= vp_start

            # Virtual points use duplicate SIMPLE_FISHEYE camera (fixed params)
            if is_virtual_point and fisheye_intrinsics_params is not None:
                camera_params = fisheye_intrinsics_params[camera_type_idx]
                active_model_id = fisheye_model_id
            else:
                camera_params = intrinsics_params[camera_type_idx]
                active_model_id = camera_model_id

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

    Loss function selection:
    - Real points (pt_idx < virtual_point_start[image_id]): Huber loss
    - Virtual points (pt_idx >= virtual_point_start[image_id]): Arctan loss

    Cost function selection:
    - Normal depth: ReprojErrorCost
    - Negative depth (pt_idx in negative_depth_observations[image_id]): ReprojErrorCostWithNegativeDepth

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

    # Step 1: Extract data from reconstruction
    (
        global_rotations,
        global_centers,
        intrinsics_mapping,
        intrinsics_params,
        keypoints_per_image,
        camera_model,
    ) = extract_ba_data_from_reconstruction(reconstruction)

    # Step 2: Initialize optimization parameters
    cam_poses = initialize_parameters(
        global_rotations,
        global_centers,
    )

    logger.info(f"Bundle adjustment with {len(reconstruction.points3D)} tracks")

    # Step 4: Build problem
    prob = pyceres.Problem()
    costs = []
    losses = []

    # Step 5: Add reprojection constraints
    first_camera_id = add_reprojection_error(
        prob,
        costs,
        losses,
        points3D=reconstruction.points3D,
        keypoints_per_image=keypoints_per_image,
        negative_depth_observations=negative_depth_observations,
        virtual_point_start=virtual_point_start,
        cam_poses=cam_poses,
        intrinsics_params=intrinsics_params,
        intrinsics_mapping=intrinsics_mapping,
        camera_model=camera_model,
        fisheye_intrinsics_params=fisheye_intrinsics_params,
    )

    # Step 6: Set manifolds
    # Use product manifold for 7D pose: quaternion (4D, 3D tangent) + translation (3D, 3D tangent)
    for image_id in cam_poses:
        if prob.has_parameter_block(cam_poses[image_id]):
            pose_manifold = pygluemap.CreatePoseManifold()
            prob.set_manifold(cam_poses[image_id], pose_manifold)

    camera_model_id = get_camera_model_id(camera_model)
    pp_idxs = list(pycolmap.Camera(model=camera_model_id).principal_point_idxs())

    for params in intrinsics_params:
        if params is not None and prob.has_parameter_block(params):
            prob.set_manifold(params, pyceres.SubsetManifold(len(params), pp_idxs))

    # Fix fisheye camera params (not optimized)
    if fisheye_intrinsics_params is not None:
        for params in fisheye_intrinsics_params:
            if params is not None and prob.has_parameter_block(params):
                prob.set_parameter_block_constant(params)

    # Step 7: Fix gauge freedom
    if first_camera_id is not None:
        # Fix both rotation and translation of first camera
        if first_camera_id in cam_poses:
            if prob.has_parameter_block(cam_poses[first_camera_id]):
                prob.set_parameter_block_constant(cam_poses[first_camera_id])
                logger.info(
                    f"Fixed rotation and translation gauge: camera {first_camera_id}"
                )

        # Fix scale: pin the largest translation component of a second camera
        second_camera_id = None
        for image_id in cam_poses:
            if image_id != first_camera_id and prob.has_parameter_block(
                cam_poses[image_id]
            ):
                second_camera_id = image_id
                break
        if second_camera_id is not None:
            # Get translation part of the pose (indices 4-6)
            t2 = cam_poses[second_camera_id][4:]
            fixed_idx = int(np.argmax(np.abs(t2)))
            # Create product manifold: full quaternion manifold + subset translation manifold
            scale_gauge_manifold = pygluemap.CreatePoseManifoldWithFixedTransComponent(
                fixed_idx
            )
            prob.set_manifold(cam_poses[second_camera_id], scale_gauge_manifold)
            logger.info(
                f"Fixed scale gauge: camera {second_camera_id}, translation component {fixed_idx}"
            )
        else:
            logger.warning("No second camera available to fix scale gauge")
    else:
        logger.warning("No cameras added to problem, cannot fix gauge")

    # Step 9: Solve (solver config matching COLMAP's bundle_adjustment_ceres.cc)
    options = pyceres.SolverOptions()
    options.max_num_iterations = max_num_iterations
    options.max_num_consecutive_invalid_steps = 10
    options.minimizer_progress_to_stdout = prob.num_residuals() > 5_000_000

    # Use SPARSE_NORMAL_CHOLESKY to avoid Schur ordering issues with custom manifolds
    num_images = len(cam_poses)
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
        for image_id in cam_poses:
            if image_id != first_camera_id:
                if prob.has_parameter_block(cam_poses[image_id]):
                    translation_only_manifold = (
                        pygluemap.CreateTranslationOnlyManifold()
                    )
                    prob.set_manifold(cam_poses[image_id], translation_only_manifold)

        summary = pyceres.SolverSummary()
        pyceres.solve(options, prob, summary)
        # pygluemap.solve_cuda(options, prob, summary)

        logger.info(summary.BriefReport())

        # Release rotations for second pass - restore full pose manifold
        for image_id in cam_poses:
            if image_id != first_camera_id:
                if prob.has_parameter_block(cam_poses[image_id]):
                    # Restore appropriate manifold
                    if image_id == second_camera_id and second_camera_id is not None:
                        # Restore scale gauge manifold for second camera
                        t2 = cam_poses[second_camera_id][4:]
                        fixed_idx = int(np.argmax(np.abs(t2)))
                        scale_gauge_manifold = (
                            pygluemap.CreatePoseManifoldWithFixedTransComponent(
                                fixed_idx
                            )
                        )
                        prob.set_manifold(cam_poses[image_id], scale_gauge_manifold)
                    else:
                        # Restore standard pose manifold
                        restored_manifold = pygluemap.CreatePoseManifold()
                        prob.set_manifold(cam_poses[image_id], restored_manifold)

    # reconstruction.write("results/wrong_init")
    summary = pyceres.SolverSummary()
    pyceres.solve(options, prob, summary)
    logger.info(summary.BriefReport())

    # Step 10: Update reconstruction in-place
    update_reconstruction_from_ba_results(
        reconstruction,
        cam_poses,
        intrinsics_params,
    )

    return reconstruction
