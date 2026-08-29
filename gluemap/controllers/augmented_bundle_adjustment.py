"""
Iterative Bundle Adjustment Controller.

This module provides iterative bundle adjustment with reprojection error
filtering. It separates track establishment from BA optimization and supports
multiple filtering iterations until convergence.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
import pycolmap

from gluemap.estimators.augmented_bundle_adjustment import bundle_adjustment
from gluemap.math.reprojection_error import (
    ReprojectionErrorType,
    compute_point_error,
    filter_reconstruction_by_reprojection_error,
)
from gluemap.utils.colmap import camera_from_intrinsics_matrix

logger = logging.getLogger(__name__)


def build_negative_depth_observations(
    pts2d_idx_inv: dict[int, list],
    images_points2d_virtual_isnegative: dict[int, list] | None,
) -> dict[int, set]:
    """
    Convert old negative depth format to Dict[image_id, Set[point2D_idx]].

    Args:
        pts2d_idx_inv: Dict[image_id, List] - inverse mapping for real points
        images_points2d_virtual_isnegative: Dict[image_id, List[int]] - 0/1
            flags for each virtual point indicating negative depth

    Returns:
        Dict[image_id, Set[int]] - set of point2D indices with negative depth
    """
    if images_points2d_virtual_isnegative is None:
        return {}

    result = {}
    for image_id in images_points2d_virtual_isnegative:
        negative_set = set()
        num_real_pts = len(pts2d_idx_inv.get(image_id, []))

        for virtual_idx, is_negative in enumerate(
            images_points2d_virtual_isnegative[image_id]
        ):
            if is_negative:
                # Virtual points are appended after real points
                point2D_idx = num_real_pts + virtual_idx
                negative_set.add(point2D_idx)

        if negative_set:
            result[image_id] = negative_set

    return result


def build_reconstruction_for_ba(
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    global_intrinsics: list,
    intrinsics_mapping: dict[int, int],
    points3D: dict[int, pycolmap.Point3D],
    keypoints_per_image: dict[int, np.ndarray],
    image_sizes: list[tuple[int, int]] | None = None,
    images_list: list[str] | None = None,
    camera_model: str = "SIMPLE_PINHOLE",
) -> pycolmap.Reconstruction:
    """
    Build pycolmap.Reconstruction from separate data structures.

    All inputs use 0-indexed image / camera IDs (matching keypoints_per_image,
    global_rotations, etc.); the returned reconstruction uses 1-indexed image
    and camera IDs to match the COLMAP convention used by the database written
    in prepare_glomap_prior.

    Args:
        global_rotations: Dict[image_id, np.ndarray(3,3)] (0-indexed keys)
        global_centers: Dict[image_id, np.ndarray(3,)] (0-indexed keys)
        global_intrinsics: List of intrinsics tensors (0-indexed)
        intrinsics_mapping: Dict[image_id, camera_type_idx] (0-indexed both
            sides)
        points3D: Dict[point3D_id, pycolmap.Point3D] with 0-indexed image_ids in
            track elements
        keypoints_per_image: Dict[image_id, np.ndarray(N,2)] (0-indexed keys)
        image_sizes: List[(height, width)] indexed by 0-indexed image_id -
            optional
        images_list: List[str] of image filenames indexed by 0-indexed image_id
            - optional
        camera_model: Camera model string

    Returns:
        pycolmap.Reconstruction with 1-indexed image_id / camera_id.
    """
    reconstruction = pycolmap.Reconstruction()

    # Add cameras (camera_id is 1-indexed in the output reconstruction)
    for camera_id, intrinsics in enumerate(global_intrinsics):
        if intrinsics is None:
            continue

        # Find image size for this camera (use first image with this camera_id)
        width, height = None, None
        if image_sizes is not None:
            for img_id, cam_id in intrinsics_mapping.items():
                if cam_id == camera_id and img_id < len(image_sizes):
                    height, width = image_sizes[img_id]
                    break

        # Note that there is an extra dimension in intrinsics, so we take
        # intrinsics[0]
        camera = camera_from_intrinsics_matrix(
            intrinsics[0], camera_model, width, height, camera_id + 1
        )
        reconstruction.add_camera_with_trivial_rig(camera)

    # Add images (image_id is 1-indexed in the output reconstruction)
    for image_id in global_rotations:
        if image_id not in global_centers:
            continue
        if image_id not in intrinsics_mapping:
            continue

        R = global_rotations[image_id]
        center = global_centers[image_id]

        image = pycolmap.Image()
        image.image_id = image_id + 1
        image.camera_id = intrinsics_mapping[image_id] + 1

        # Add 2D points
        if image_id in keypoints_per_image:
            for xy in keypoints_per_image[image_id]:
                image.points2D.append(pycolmap.Point2D(xy))

        # Set image name from images_list if available, otherwise use image_id
        if images_list is not None and image_id < len(images_list):
            image.name = images_list[image_id]
        else:
            image.name = str(image_id)

        # Set pose: cam_from_world
        # t = -R @ center
        t = -R @ center
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(R), t)
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    # Add 3D points. Track elements arrive with 0-indexed image_ids; rebuild
    # each track with image_id+1 so observations point to the 1-indexed images.
    for point3D in points3D.values():
        xyz = (
            point3D.xyz.reshape(3, 1) if point3D.xyz.ndim == 1 else point3D.xyz
        )
        new_track = pycolmap.Track()
        for elem in point3D.track.elements:
            new_track.add_element(elem.image_id + 1, elem.point2D_idx)
        reconstruction.add_point3D(xyz, new_track)

    return reconstruction


def extract_results_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
) -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    list,
    dict[int, pycolmap.Point3D],
]:
    """
    Extract BA results from reconstruction back to separate data structures.

    Args:
        reconstruction: pycolmap.Reconstruction with optimized parameters

    Returns:
        Tuple of:
            - global_rotations: Dict[image_id, np.ndarray(3,3)]
            - global_centers: Dict[image_id, np.ndarray(3,)]
            - global_intrinsics: List of intrinsics tensors
            - points3D: Dict[point3D_id, pycolmap.Point3D]
    """
    import torch

    global_rotations = {}
    global_centers = {}

    for image_id, image in reconstruction.images.items():
        R = np.array(image.cam_from_world().rotation.matrix())
        t = np.array(image.cam_from_world().translation)
        center = -R.T @ t

        global_rotations[image_id] = R
        global_centers[image_id] = center

    # Build intrinsics list
    max_camera_id = (
        max(reconstruction.cameras.keys()) if reconstruction.cameras else -1
    )
    global_intrinsics = [None] * (max_camera_id + 1)

    for camera_id, camera in reconstruction.cameras.items():
        intrinsics_matrix = camera.calibration_matrix()
        intrinsics_tensor = torch.from_numpy(intrinsics_matrix).unsqueeze(0)
        global_intrinsics[camera_id] = intrinsics_tensor

    # Points3D are already in the reconstruction
    points3D = dict(reconstruction.points3D)

    return global_rotations, global_centers, global_intrinsics, points3D


@dataclass
class IterativeBAOptions:
    """Options for iterative bundle adjustment."""

    # Maximum BA iterations per round
    max_ba_iterations: int = 200

    # Number of iterative filtering rounds
    max_filter_iterations: int = 1

    # Normalized reprojection error threshold for outlier detection
    # (error / focal_length)
    normalized_reproj_threshold: float = 0.01

    # Minimum track length after filtering
    min_track_length: int = 2

    # Convergence threshold (fraction of outliers removed)
    convergence_threshold: float = 0.01

    # Whether to fix rotations in first BA pass
    fix_rotations_first_pass: bool = False

    # Whether to filter virtual points same as real tracks
    filter_virtual_points: bool = True

    # Ceres linear solver selected by the product profile.
    linear_solver_type: str = "auto"


def prune_track_outliers(
    points3D: dict[int, pycolmap.Point3D],
    inlier_mask: dict[int, list[bool]],
) -> None:
    """
    Remove outlier observations from tracks.

    Args:
        points3D: Dictionary of Point3D objects (modified in place)
        inlier_mask: Dict[point3D_id, List[bool]] indicating inliers
    """
    points_to_remove = []
    total_outliers_removed = 0

    for point3D_id in list(points3D.keys()):
        # If point is not present in inlier_mask, filter it out
        if point3D_id not in inlier_mask:
            points_to_remove.append(point3D_id)
            continue

        flags = inlier_mask[point3D_id]
        point3D = points3D[point3D_id]
        elements = list(point3D.track.elements)

        # Create new track with only inliers
        new_track = pycolmap.Track()
        num_inliers = 0
        for elem, is_inlier in zip(elements, flags, strict=False):
            if is_inlier:
                new_track.add_element(elem.image_id, elem.point2D_idx)
                num_inliers += 1
            else:
                total_outliers_removed += 1

        if num_inliers >= 2:
            point3D.track = new_track
        else:
            points_to_remove.append(point3D_id)

    # Remove tracks with too few inliers
    for point3D_id in points_to_remove:
        del points3D[point3D_id]

    logger.info(f"Pruned {total_outliers_removed} outlier observations")
    logger.info(f"Removed {len(points_to_remove)} tracks with < 2 inliers")


def initialize_world_points(
    predictions_dict: dict,
    global_rotations: dict[int, np.ndarray],
    global_centers: dict[int, np.ndarray],
    points3D: dict[int, pycolmap.Point3D],
    pts2d_idx_inv: dict[int, list] | None,
    pts2d_idx_virtual_inv: dict[int, list] | None,
    keypoints_per_image: dict[int, np.ndarray],
    cameras: list[pycolmap.Camera | None],
    intrinsics_mapping: dict[int, int],
    angular_error_threshold_deg: float = 1.0,
    negative_depth_observations: dict[int, set] | None = None,
) -> dict[int, pycolmap.Point3D]:
    """
    Initialize 3D world points from virtual point predictions.

    For each track, finds the first virtual-point observation taken from the
    star center camera (pos==0), transforms it into world coordinates, then
    counts inliers across the track's other observations and prunes outliers.

    Args:
        predictions_dict: Dictionary containing points3d_virtual and indexes.
        global_rotations: Dictionary of global rotation matrices
        global_centers: Dictionary of global camera centers
        points3D: Dictionary of established tracks (pycolmap.Point3D) -
            modified in place
        pts2d_idx_inv: Inverse mapping for real points; used only to compute
            the virtual-point offset within keypoints_per_image.
        pts2d_idx_virtual_inv: Inverse mapping for virtual points
        keypoints_per_image: Dict[image_id, np.ndarray(N, 2)] - 2D observations
        cameras: List[pycolmap.Camera] - cameras indexed by camera type
        intrinsics_mapping: Dict[image_id, camera_type_idx]
        angular_error_threshold_deg: Angular error threshold in degrees for
            inlier counting

    Returns:
        points3D: Modified dictionary with xyz set and outliers pruned
    """
    inlier_mask = {}
    num_tracks_before = len(points3D)
    num_initialized_virtual = 0

    total_virtual_inlier_count = 0
    total_virtual_outlier_count = 0
    inlier_count_per_pair = defaultdict(lambda: defaultdict(int))
    if (
        pts2d_idx_virtual_inv is not None
        and "points3d_virtual" in predictions_dict
    ):
        for point3D_id, point3D in list(points3D.items()):
            elements = list(point3D.track.elements)

            if len(elements) < 2:
                continue

            # Find first valid virtual point observation with pos==0
            # (center camera)
            world_point = None
            for elem in elements:
                image_id, pt_idx = elem.image_id, elem.point2D_idx

                if image_id not in pts2d_idx_virtual_inv:
                    continue

                # Compute virtual index by subtracting real points offset
                # In keypoints_per_image, virtual points are at indices
                # len(real_pts) onwards
                num_real_pts = len(pts2d_idx_inv.get(image_id, []))
                virtual_idx = pt_idx - num_real_pts

                if virtual_idx < 0 or virtual_idx >= len(
                    pts2d_idx_virtual_inv[image_id]
                ):
                    break

                idx_star, pos, j = pts2d_idx_virtual_inv[image_id][virtual_idx]

                # Only use center camera observations (pos==0)
                if pos != 0:
                    continue

                # Get virtual 3D point (in star center frame)
                virtual_point = (
                    predictions_dict["points3d_virtual"][idx_star][0, j]
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

                # Skip points with z values too close to 0 (degenerate cases)
                if abs(virtual_point[2]) < 1e-3:
                    continue

                # Transform to world coordinates using center camera of star
                center_image_id = predictions_dict["indexes"][idx_star][0]
                if (
                    center_image_id not in global_rotations
                    or center_image_id not in global_centers
                ):
                    continue

                R = global_rotations[center_image_id]
                center = global_centers[center_image_id]
                world_point = R.T @ virtual_point + center
                break

            if world_point is not None:
                # Compute inlier mask for virtual point observations
                inlier_flags = []
                for elem in elements:
                    image_id, pt_idx = elem.image_id, elem.point2D_idx
                    is_neg = (
                        negative_depth_observations is not None
                        and image_id in negative_depth_observations
                        and pt_idx in negative_depth_observations[image_id]
                    )
                    error = compute_point_error(
                        world_point,
                        global_rotations[image_id],
                        global_centers[image_id],
                        keypoints_per_image[image_id][pt_idx],
                        cameras[intrinsics_mapping[image_id]],
                        error_type=ReprojectionErrorType.ANGULAR,
                        is_negative_depth=is_neg,
                    )
                    is_inlier = error < angular_error_threshold_deg
                    inlier_flags.append(is_inlier)
                    if is_inlier:
                        inlier_count_per_pair[center_image_id][image_id] += 1
                        inlier_count_per_pair[image_id][center_image_id] += 1
                        total_virtual_inlier_count += 1
                    else:
                        total_virtual_outlier_count += 1

                inlier_count = sum(inlier_flags)
                if inlier_count >= 2:
                    points3D[point3D_id].xyz = world_point.astype(np.float64)
                    inlier_mask[point3D_id] = inlier_flags
                    num_initialized_virtual += 1
                else:
                    # Mark all as outliers if not enough inliers
                    inlier_mask[point3D_id] = [False] * len(elements)

    logger.info(
        f"Virtual points: {total_virtual_inlier_count} inliers, "
        f"{total_virtual_outlier_count} outliers"
    )
    if num_initialized_virtual > 0:
        logger.info(
            "Average inliers per virtual track: "
            f"{total_virtual_inlier_count / num_initialized_virtual:.2f}"
        )

    # Prune outliers from tracks (modify points3D in place)
    prune_track_outliers(points3D, inlier_mask)

    logger.info(f"Tracks: {num_tracks_before} -> {len(points3D)} after pruning")
    logger.info(f"Initialized {num_initialized_virtual} virtual world points")

    return points3D


def iterative_bundle_adjustment(
    reconstruction: pycolmap.Reconstruction,
    virtual_reconstruction: pycolmap.Reconstruction | None,
    negative_depth_observations: dict[int, set],
    options: IterativeBAOptions = None,
) -> tuple[
    pycolmap.Reconstruction,
    pycolmap.Reconstruction | None,
    list[object],
]:
    """
    Run iterative bundle adjustment with reprojection error filtering.

    Main entry point for the iterative BA controller. Runs multiple rounds of
    BA + filtering until convergence over a paired (real, virtual)
    reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction with the real tracks and the
            authoritative poses/intrinsics. Optimized in-place.
        virtual_reconstruction: pycolmap.Reconstruction holding virtual tracks
            whose xyz values are jointly optimized in-place, or ``None`` for a
            pure real BA.
        negative_depth_observations: Dict[image_id, Set[point2D_idx]] for
            negative depth observations.
        options: BA options (uses defaults if None).

    Returns:
        (reconstruction, virtual_reconstruction) -- both modified in-place.
    """
    if options is None:
        options = IterativeBAOptions()

    if options.fix_rotations_first_pass:
        logger.warning(
            "fix_rotations_first_pass is no longer supported with "
            "pycolmap-based BA; ignoring."
        )

    num_virtual_tracks = (
        len(virtual_reconstruction.points3D)
        if virtual_reconstruction is not None
        else 0
    )
    logger.info(
        f"Starting iterative BA with {len(reconstruction.points3D)} real "
        f"tracks, {num_virtual_tracks} virtual tracks"
    )
    logger.info(f"  Max filter iterations: {options.max_filter_iterations}")
    logger.info(
        f"  Normalized reproj threshold: {options.normalized_reproj_threshold}"
    )
    logger.info(f"  Min track length: {options.min_track_length}")

    def _count_observations():
        real_obs = sum(
            len(list(p.track.elements))
            for p in reconstruction.points3D.values()
        )
        virt_obs = (
            sum(
                len(list(p.track.elements))
                for p in virtual_reconstruction.points3D.values()
            )
            if virtual_reconstruction is not None
            else 0
        )
        return real_obs, virt_obs

    real_obs, virt_obs = _count_observations()
    logger.info(f"Initial observations: real={real_obs}, virtual={virt_obs}")
    summaries: list[object] = []

    # Iterative filtering loop (outer loop runs BA, inner loop tightens
    # threshold)
    iteration = 0
    while iteration < options.max_filter_iterations:
        # Scaling factor decreases per iteration: 3, 2, 1, 1, ...
        scaling = max(3 - iteration, 1)
        current_threshold = scaling * options.normalized_reproj_threshold

        logger.info(
            f"=== Iteration {iteration + 1}/{options.max_filter_iterations} ==="
        )
        current_virtual_tracks = (
            len(virtual_reconstruction.points3D)
            if virtual_reconstruction is not None
            else 0
        )
        logger.info(
            f"Tracks: real={len(reconstruction.points3D)}, "
            f"virtual={current_virtual_tracks}"
        )
        logger.info(f"Threshold scaling: {scaling}x -> {current_threshold:.4f}")

        # Run BA via bundle_adjustment
        reconstruction, virtual_reconstruction, summary = bundle_adjustment(
            reconstruction,
            virtual_reconstruction,
            negative_depth_observations,
            max_num_iterations=options.max_ba_iterations,
            linear_solver_type=options.linear_solver_type,
        )
        summaries.append(summary)

        # Inner loop: filter and tighten threshold when too few tracks filtered
        # (matches C++ IterativeBundleAdjustment pattern)
        logger.info("Filtering tracks by reprojection ...")
        total_filtered = 0
        should_stop = True  # Will be set to False if enough tracks are filtered

        while iteration < options.max_filter_iterations:
            scaling = max(3 - iteration, 1)
            current_threshold = scaling * options.normalized_reproj_threshold

            obs_removed_total = 0
            tracks_removed_total = 0
            for recon, neg_depth, label in (
                (reconstruction, None, "real: "),
                (
                    virtual_reconstruction,
                    negative_depth_observations,
                    "virtual: ",
                ),
            ):
                if recon is None:
                    continue
                obs_removed, tracks_removed = (
                    filter_reconstruction_by_reprojection_error(
                        recon,
                        ReprojectionErrorType.NORMALIZED,
                        current_threshold,
                        options.min_track_length,
                        negative_depth_observations=neg_depth,
                        log_level=logging.DEBUG,
                        log_prefix=label,
                    )
                )
                obs_removed_total += obs_removed
                tracks_removed_total += tracks_removed

            total_filtered += obs_removed_total

            logger.info(
                f"Filtered: {obs_removed_total} observations, "
                f"{tracks_removed_total} tracks removed"
            )

            # Check if enough tracks were filtered (> convergence_threshold of
            # combined points)
            num_points = len(reconstruction.points3D) + (
                len(virtual_reconstruction.points3D)
                if virtual_reconstruction is not None
                else 0
            )
            if (
                num_points > 0
                and total_filtered > options.convergence_threshold * num_points
            ):
                # Enough filtered, break inner loop to run BA again
                should_stop = False
                iteration += 1
                break
            else:
                # Too few filtered, tighten threshold and filter again
                # without BA
                iteration += 1
                if iteration < options.max_filter_iterations:
                    logger.debug(
                        f"Low removal ({total_filtered} total), tightening "
                        f"threshold (iteration -> {iteration + 1})"
                    )

        if should_stop:
            # Max iterations reached during tightening, stop outer loop
            logger.info("Max iterations reached, stopping")
            break

    real_obs, virt_obs = _count_observations()
    final_virtual_tracks = (
        len(virtual_reconstruction.points3D)
        if virtual_reconstruction is not None
        else 0
    )
    logger.info(
        f"Final: {len(reconstruction.points3D)} real tracks ({real_obs} obs), "
        f"{final_virtual_tracks} virtual tracks ({virt_obs} obs)"
    )

    return reconstruction, virtual_reconstruction, summaries
