from enum import Enum

import numpy as np
import pycolmap


class ReprojectionErrorType(Enum):
    PIXEL = "pixel"
    NORMALIZED = "normalized"
    ANGULAR = "angular"


def compute_point_error(
    world_point: np.ndarray,
    R: np.ndarray,
    center: np.ndarray,
    observed: np.ndarray,
    camera: pycolmap.Camera,
    error_type: ReprojectionErrorType = ReprojectionErrorType.PIXEL,
    is_negative_depth: bool = False,
) -> float:
    """
    Compute error for a single 3D point observation.

    Args:
        world_point: 3D point in world coordinates (3,).
        R: Rotation matrix world-to-camera (3, 3).
        center: Camera center in world coordinates (3,).
        observed: Observed 2D point (2,).
        camera: pycolmap.Camera with intrinsics.
        error_type: PIXEL (pixels), NORMALIZED (pixels / focal), or ANGULAR (degrees).
        is_negative_depth: If True, negate X_cam before projection.

    Returns:
        Error value (float), or float("inf") for degenerate cases.
    """
    X_cam = R @ (world_point - center)

    if is_negative_depth:
        X_cam = -X_cam

    if error_type == ReprojectionErrorType.ANGULAR:
        if not is_negative_depth and X_cam[2] <= 0:
            return float("inf")
        norm_3d = np.linalg.norm(X_cam)
        if norm_3d < 1e-10:
            return float("inf")
        ray_3d = X_cam / norm_3d
        ray_2d = camera.cam_from_img(observed)
        ray_obs = np.array([ray_2d[0], ray_2d[1], 1.0])
        ray_obs = ray_obs / np.linalg.norm(ray_obs)
        cos_angle = np.clip(np.dot(ray_3d, ray_obs), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))
    else:
        if X_cam[2] <= 0:
            return float("inf")
        projected = camera.img_from_cam(X_cam)
        pixel_error = np.sqrt(
            (projected[0] - observed[0]) ** 2
            + (projected[1] - observed[1]) ** 2
        )
        if error_type == ReprojectionErrorType.NORMALIZED:
            return pixel_error / camera.focal_length
        return pixel_error


def _compute_errors_batch(
    X_cam: np.ndarray,
    observed: np.ndarray,
    camera: pycolmap.Camera,
    neg_mask: np.ndarray,
    error_type: ReprojectionErrorType,
) -> np.ndarray:
    """
    Compute errors for a batch of observations sharing the same camera.

    Args:
        X_cam: Camera-space points (N, 3).
        observed: Observed 2D points (N, 2).
        camera: pycolmap.Camera.
        neg_mask: Boolean mask (N,) — True for negative-depth observations.
        error_type: PIXEL, NORMALIZED, or ANGULAR.

    Returns:
        Errors array (N,).
    """
    N = X_cam.shape[0]
    errors = np.full(N, float("inf"))

    # Negate camera-space coords for negative-depth observations
    X_proc = X_cam.copy()
    if np.any(neg_mask):
        X_proc[neg_mask] = -X_proc[neg_mask]

    if error_type == ReprojectionErrorType.ANGULAR:
        # For non-negative-depth, require positive Z
        valid = X_proc[:, 2] > 0
        norms = np.linalg.norm(X_proc, axis=1)
        valid &= norms > 1e-10
        if not np.any(valid):
            return errors

        rays_3d = X_proc[valid] / norms[valid, np.newaxis]
        rays_2d = camera.cam_from_img(observed[valid])  # (M, 2)
        rays_obs = np.column_stack([rays_2d, np.ones(rays_2d.shape[0])])
        rays_obs = rays_obs / np.linalg.norm(rays_obs, axis=1, keepdims=True)
        cos_angles = np.clip(np.sum(rays_3d * rays_obs, axis=1), -1.0, 1.0)
        errors[valid] = np.degrees(np.arccos(cos_angles))
    else:
        valid = X_proc[:, 2] > 0
        if not np.any(valid):
            return errors

        projected = camera.img_from_cam(X_proc[valid])  # (M, 2)
        pixel_errors = np.linalg.norm(projected - observed[valid], axis=1)
        if error_type == ReprojectionErrorType.NORMALIZED:
            pixel_errors = pixel_errors / camera.focal_length
        errors[valid] = pixel_errors

    return errors


def compute_all_errors_from_reconstruction(
    reconstruction: pycolmap.Reconstruction,
    error_type: ReprojectionErrorType = ReprojectionErrorType.PIXEL,
    negative_depth_observations: dict[int, set] | None = None,
    virtual_point_start: dict[int, int] | None = None,
    fisheye_cameras: dict[int, pycolmap.Camera] | None = None,
) -> dict[int, list[tuple[int, int, float]]]:
    """
    Compute errors for all 3D point track observations in a reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction with cameras, images, and points3D.
        error_type: PIXEL, NORMALIZED, or ANGULAR.
        negative_depth_observations: Dict[image_id, Set[point2D_idx]] for
            observations where the 3D point has negative depth.
        virtual_point_start: Dict[image_id, int] — where virtual points start per image.
            If provided along with fisheye_cameras, virtual observations use the
            fisheye camera for error computation.
        fisheye_cameras: Dict[camera_id, pycolmap.Camera] — SIMPLE_FISHEYE cameras
            indexed by camera_id, used for virtual point observations.

    Returns:
        Dict mapping point3D_id -> List[(image_id, point2D_idx, error)].
    """
    # --- First pass: gather observations per image ---
    # obs_by_image[image_id] = list of (point3D_id, pt_idx, world_xyz)
    obs_by_image: dict[int, list[tuple[int, int, np.ndarray]]] = {}
    # Track invalid observations (missing image/camera/out-of-bounds)
    invalid_obs: dict[int, list[tuple[int, int]]] = {}

    for point3D_id, point3D in reconstruction.points3D.items():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            continue

        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx

            if image_id not in reconstruction.images:
                invalid_obs.setdefault(point3D_id, []).append(
                    (image_id, pt_idx)
                )
                continue

            image = reconstruction.images[image_id]
            if pt_idx >= len(image.points2D):
                invalid_obs.setdefault(point3D_id, []).append(
                    (image_id, pt_idx)
                )
                continue

            if image.camera_id not in reconstruction.cameras:
                invalid_obs.setdefault(point3D_id, []).append(
                    (image_id, pt_idx)
                )
                continue

            obs_by_image.setdefault(image_id, []).append(
                (point3D_id, pt_idx, world_point)
            )

    # --- Second pass: vectorized per-image error computation ---
    # error_results[(point3D_id, image_id, pt_idx)] = error
    error_results: dict[tuple[int, int, int], float] = {}

    for image_id, obs_list in obs_by_image.items():
        image = reconstruction.images[image_id]
        camera_id = image.camera_id
        camera = reconstruction.cameras[camera_id]
        n_pts2d = len(image.points2D)
        pose = image.cam_from_world()

        # Determine virtual point start for this image
        vp_start = (
            virtual_point_start.get(image_id, n_pts2d)
            if virtual_point_start is not None
            else n_pts2d
        )
        has_fisheye = (
            fisheye_cameras is not None and camera_id in fisheye_cameras
        )

        # Separate pinhole vs fisheye observations
        pinhole_obs = []
        fisheye_obs = []
        for point3D_id, pt_idx, world_xyz in obs_list:
            if pt_idx >= vp_start and has_fisheye:
                fisheye_obs.append((point3D_id, pt_idx, world_xyz))
            else:
                pinhole_obs.append((point3D_id, pt_idx, world_xyz))

        # Process each camera group
        for cam, group in [(camera, pinhole_obs)]:
            if not group:
                continue
            _process_obs_group(
                group,
                image,
                pose,
                cam,
                image_id,
                negative_depth_observations,
                error_type,
                error_results,
            )

        if has_fisheye and fisheye_obs:
            _process_obs_group(
                fisheye_obs,
                image,
                pose,
                fisheye_cameras[camera_id],
                image_id,
                negative_depth_observations,
                error_type,
                error_results,
            )

    # --- Third pass: assemble errors_per_track ---
    errors_per_track: dict[int, list[tuple[int, int, float]]] = {}

    for point3D_id, point3D in reconstruction.points3D.items():
        world_point = point3D.xyz
        if world_point is None or np.all(world_point == 0):
            continue

        track_errors = []

        # Add invalid observations
        if point3D_id in invalid_obs:
            for image_id, pt_idx in invalid_obs[point3D_id]:
                track_errors.append((image_id, pt_idx, float("inf")))

        # Add computed errors
        for elem in point3D.track.elements:
            image_id, pt_idx = elem.image_id, elem.point2D_idx
            key = (point3D_id, image_id, pt_idx)
            if key in error_results:
                track_errors.append((image_id, pt_idx, error_results[key]))

        errors_per_track[point3D_id] = track_errors

    return errors_per_track


def _process_obs_group(
    group: list[tuple[int, int, np.ndarray]],
    image: pycolmap.Image,
    pose,
    camera: pycolmap.Camera,
    image_id: int,
    negative_depth_observations: dict[int, set] | None,
    error_type: ReprojectionErrorType,
    error_results: dict[tuple[int, int, int], float],
) -> None:
    """Process a group of observations for one image+camera, writing results into error_results."""
    point3D_ids = [g[0] for g in group]
    pt_idxs = [g[1] for g in group]
    world_points = np.array([g[2] for g in group])  # (N, 3)
    observed = np.array([image.points2D[idx].xy for idx in pt_idxs])  # (N, 2)

    # Batched world-to-camera transform
    X_cam = pose * world_points  # (N, 3)

    # Build negative-depth mask
    if (
        negative_depth_observations is not None
        and image_id in negative_depth_observations
    ):
        neg_set = negative_depth_observations[image_id]
        neg_mask = np.array([idx in neg_set for idx in pt_idxs], dtype=bool)
    else:
        neg_mask = np.zeros(len(group), dtype=bool)

    errors = _compute_errors_batch(
        X_cam, observed, camera, neg_mask, error_type
    )

    for i, (p3d_id, pt_idx) in enumerate(zip(point3D_ids, pt_idxs)):
        error_results[(p3d_id, image_id, pt_idx)] = float(errors[i])
