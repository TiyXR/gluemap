"""Evaluation functions for comparing COLMAP reconstructions."""

import os
from typing import Dict, Any, List, Tuple, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pycolmap

try:
    from evo.core.trajectory import PosePath3D
    from evo.core.metrics import APE, RPE, PoseRelation
    from evo.core.units import Unit
    from evo.core import geometry
    EVO_AVAILABLE = True
except ImportError:
    EVO_AVAILABLE = False

M_PI = np.pi


def load_colmap_reconstruction(path: str) -> pycolmap.Reconstruction:
    """Load a COLMAP reconstruction from disk.

    Args:
        path: Path to the COLMAP reconstruction directory
            (containing cameras.bin, images.bin, points3D.bin or .txt equivalents)

    Returns:
        pycolmap.Reconstruction object

    Raises:
        FileNotFoundError: If the reconstruction path does not exist
        RuntimeError: If the reconstruction cannot be loaded
    """
    if not os.path.isdir(path):
        raise FileNotFoundError(f"Reconstruction path does not exist: {path}")

    reconstruction = pycolmap.Reconstruction()
    reconstruction.read(path)

    return reconstruction


def extract_poses(
    reconstruction: pycolmap.Reconstruction,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Extract camera poses from a COLMAP reconstruction.

    Args:
        reconstruction: pycolmap.Reconstruction object

    Returns:
        Tuple of (R_dict, C_dict) where:
            R_dict: Dictionary mapping image names to 3x3 rotation matrices
            C_dict: Dictionary mapping image names to 3x1 camera centers
    """
    R_dict = {}
    C_dict = {}
    for image_id, image in reconstruction.images.items():
        # Get rotation matrix from quaternion
        R = image.cam_from_world().rotation.matrix()
        # Camera center: C = -R^T @ t
        t = image.cam_from_world().translation
        C = -R.T @ t
        R_dict[image.name] = np.array(R)
        C_dict[image.name] = np.array(C)
    return R_dict, C_dict


def calc_pairwise_relative_pose_batch(
    R_dict: Dict[str, np.ndarray],
    c_dict: Dict[str, np.ndarray],
    keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute all pairwise relative poses in batch.

    Args:
        R_dict: Dictionary mapping image names to 3x3 rotation matrices
        c_dict: Dictionary mapping image names to 3x1 camera centers
        keys: Optional list of image names to use (default: sorted keys)

    Returns:
        Tuple of (R_rel, t_rel) where:
            R_rel: (3, N*N*3) array of relative rotations
            t_rel: (3, N*N) array of normalized relative translations
    """
    if keys is None:
        keys = sorted(list(R_dict.keys()))

    N = len(keys)
    R_batch = np.zeros([N * 3, 3])
    c_batch = np.zeros([N, 3])

    for i, key in enumerate(keys):
        if key not in R_dict or R_dict[key] is None:
            R_batch[i * 3 : i * 3 + 3, :] = np.eye(3)
            c_batch[i] = np.zeros([3])
        else:
            R_batch[i * 3 : i * 3 + 3, :] = R_dict[key]
            c_batch[i] = c_dict[key]

    # Pre-allocate memory
    R_rel = np.zeros([3, N * N * 3])
    t_rel = np.zeros([3, N * N])

    for j in range(N):
        R_rel[:, N * 3 * j : N * 3 * (j + 1)] = (
            R_batch[j * 3 : j * 3 + 3, :] @ R_batch.T
        )
        t_rel[:, N * j : N * (j + 1)] = (
            R_batch[j * 3 : j * 3 + 3, :] @ (c_batch - c_batch[j]).T
        )

    # Perform column-wise normalization
    t_rel[:, np.linalg.norm(t_rel, axis=0) < 1e-10] = 0
    t_rel = t_rel / (1e-10 + np.linalg.norm(t_rel, axis=0))

    return R_rel, t_rel


def calc_pairwise_relative_error_batch(
    R_gt: Dict[str, np.ndarray],
    c_gt: Dict[str, np.ndarray],
    R_est: Dict[str, np.ndarray],
    c_est: Dict[str, np.ndarray],
    keys: Optional[List[str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute pairwise relative pose errors in batch.

    Args:
        R_gt: Ground truth rotation matrices
        c_gt: Ground truth camera centers
        R_est: Estimated rotation matrices
        c_est: Estimated camera centers
        keys: Optional list of image names to use

    Returns:
        Tuple of (R_error, t_error) where:
            R_error: (N*N,) array of rotation errors in degrees
            t_error: (N*N,) array of translation errors in degrees
    """
    R_rel_calc, t_rel_calc = calc_pairwise_relative_pose_batch(R_est, c_est, keys)
    R_rel_gt, t_rel_gt = calc_pairwise_relative_pose_batch(R_gt, c_gt, keys)

    R_diff = R_rel_calc - R_rel_gt
    t_diff = t_rel_calc - t_rel_gt
    
    # Compute rotation error using Frobenius norm
    R_error = np.zeros([R_diff.shape[1] // 3])
    for i in range(3):
        for j in range(3):
            R_error += R_diff[i, j::3] * R_diff[i, j::3]
    R_error = 1 - R_error / 4
    R_error = np.arccos(np.clip(R_error, -1, 1)) * 180 / M_PI

    # Compute translation error
    t_error = np.sum(t_diff * t_diff, axis=0)
    t_error = 1 - t_error / 2
    t_error = np.arccos(np.clip(t_error, -1, 1)) * 180 / M_PI

    return R_error, t_error


def compute_pairwise_pose_errors(
    pred_R: Dict[str, np.ndarray],
    pred_C: Dict[str, np.ndarray],
    gt_R: Dict[str, np.ndarray],
    gt_C: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute pose errors for all image pairs using batch computation.

    Args:
        pred_R: Predicted rotation matrices
        pred_C: Predicted camera centers
        gt_R: Ground truth rotation matrices
        gt_C: Ground truth camera centers

    Returns:
        Tuple of (rotation_errors, translation_errors, pose_errors) where each
        is an array of errors in degrees for all N*N pairs
    """
    # Find common images
    common_images = sorted(set(pred_R.keys()) & set(gt_R.keys()))

    if len(common_images) < 2:
        return np.array([]), np.array([]), np.array([])

    # Compute all pairwise errors in batch
    rotation_errors, translation_errors = calc_pairwise_relative_error_batch(
        gt_R, gt_C, pred_R, pred_C, keys=common_images
    )

    pose_errors = np.maximum(rotation_errors, translation_errors)

    return rotation_errors, translation_errors, pose_errors


def compute_auc_at_thresholds(
    errors: np.ndarray, thresholds: List[float]
) -> Dict[str, float]:
    """Compute AUC at each threshold using actual error values.

    AUC@t = area under accuracy curve from 0 to t, normalized by t.
    Uses actual error values as integration points for maximum accuracy.

    Args:
        errors: Array of errors in degrees
        thresholds: List of thresholds in degrees (e.g., [5, 10, 20])

    Returns:
        Dictionary mapping threshold names to AUC values (e.g., {"auc@5": 0.85, ...})
    """
    if len(errors) == 0:
        return {f"auc@{t}": 0.0 for t in thresholds}

    errors = np.sort(errors)
    results = {}

    for t in thresholds:
        # Filter errors <= t, prepend 0, append t
        relevant = errors[errors <= t]
        points = np.concatenate([[0], relevant, [t]])
        # Accuracy at each point = fraction of all errors <= that point
        accuracies = np.searchsorted(errors, points, side="right") / len(errors)
        # Trapezoidal integration, normalized by threshold
        auc = np.trapz(accuracies, points) / t
        results[f"auc@{t}"] = float(auc)

    return results


def compute_evo_metrics(
    pred_R: Dict[str, np.ndarray],
    pred_C: Dict[str, np.ndarray],
    gt_R: Dict[str, np.ndarray],
    gt_C: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Compute evo trajectory metrics (ATE, ARE, RPE) after Sim(3) alignment.

    Args:
        pred_R: Predicted rotation matrices (cam_from_world)
        pred_C: Predicted camera centers (world coords)
        gt_R: Ground truth rotation matrices (cam_from_world)
        gt_C: Ground truth camera centers (world coords)

    Returns:
        Dictionary with ate_*, are_*, rpe_t_*, rpe_r_* metrics
    """
    common = sorted(set(pred_R) & set(gt_R))
    if len(common) < 2:
        return {}

    # Build world_from_cam SE(3) poses: [[R^T, C], [0, 1]]
    gt_poses, est_poses = [], []
    for name in common:
        T_gt = np.eye(4)
        T_gt[:3, :3] = gt_R[name].T
        T_gt[:3, 3] = gt_C[name]
        gt_poses.append(T_gt)

        T_est = np.eye(4)
        T_est[:3, :3] = pred_R[name].T
        T_est[:3, 3] = pred_C[name]
        est_poses.append(T_est)

    traj_ref = PosePath3D(poses_se3=gt_poses)
    traj_est = PosePath3D(poses_se3=est_poses)

    # Sim(3) alignment (SfM is up to scale)
    try:
        r_align, t_align, scale = geometry.umeyama_alignment(
            traj_est.positions_xyz.T, traj_ref.positions_xyz.T, with_scale=True
        )
    except geometry.GeometryException:
        print("Warning: Umeyama alignment failed (degenerate configuration), skipping evo metrics")
        return {}

    # Apply alignment to estimated poses
    aligned_poses = []
    for pose in est_poses:
        aligned = np.eye(4)
        aligned[:3, :3] = r_align @ pose[:3, :3]
        aligned[:3, 3] = scale * (r_align @ pose[:3, 3]) + t_align
        aligned_poses.append(aligned)

    traj_est_aligned = PosePath3D(poses_se3=aligned_poses)

    # Compute metrics
    results = {}
    metric_configs = [
        ("ate", APE, PoseRelation.translation_part),
        ("are", APE, PoseRelation.rotation_angle_deg),
        ("rpe_t", RPE, PoseRelation.translation_part),
        ("rpe_r", RPE, PoseRelation.rotation_angle_deg),
    ]
    for prefix, metric_cls, relation in metric_configs:
        if metric_cls == APE:
            metric = metric_cls(relation)
        else:
            metric = metric_cls(relation, delta=1, delta_unit=Unit.frames, all_pairs=False)
        metric.process_data((traj_ref, traj_est_aligned))
        stats = metric.get_all_statistics()
        results[f"{prefix}_rmse"] = stats["rmse"]
        results[f"{prefix}_mean"] = stats["mean"]
        results[f"{prefix}_median"] = stats["median"]

    return results


def _remap_to_basename(
    R_dict: Dict[str, np.ndarray], C_dict: Dict[str, np.ndarray]
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Remap pose dicts to use basename-only keys."""
    R_out, C_out = {}, {}
    for name in R_dict:
        base = os.path.basename(name)
        R_out[base] = R_dict[name]
        C_out[base] = C_dict[name]
    return R_out, C_out


def compare_reconstructions(
    pred_reconstruction: pycolmap.Reconstruction,
    gt_reconstruction: pycolmap.Reconstruction,
    pose_thresholds: Optional[List[float]] = None,
    save_path: Optional[str] = None,
    error_metric: str = "pose",
    match_by_basename: bool = False,
    use_evo: bool = False,
) -> Dict[str, Any]:
    """Compare predicted reconstruction against ground truth.

    Computes pairwise pose AUC metrics using efficient batch computation.
    Optionally computes evo trajectory metrics (ATE, ARE, RPE).

    Args:
        pred_reconstruction: Predicted COLMAP reconstruction
        gt_reconstruction: Ground truth COLMAP reconstruction
        pose_thresholds: List of thresholds for AUC computation (in degrees)
        save_path: Optional path to save pairwise error heatmap
        error_metric: Which error to use for AUC computation.
            "pose" (default): max(rotation, translation)
            "rotation": rotation error only
            "translation": translation error only
        match_by_basename: If True, match images by basename only
        use_evo: If True, compute evo trajectory metrics (ATE, ARE, RPE)

    Returns:
        Dictionary containing comparison metrics:
            - num_images_pred: Number of images in predicted reconstruction
            - num_images_gt: Number of images in ground truth reconstruction
            - num_images_common: Number of common images
            - num_pairs: Number of image pairs evaluated
            - auc@X: AUC at threshold X degrees (for each threshold)
            - ate_*, are_*, rpe_t_*, rpe_r_*: evo metrics (if use_evo=True)
    """
    if pose_thresholds is None:
        pose_thresholds = [1, 3, 5]

    # Extract poses as separate R and C dictionaries
    pred_R, pred_C = extract_poses(pred_reconstruction)
    gt_R, gt_C = extract_poses(gt_reconstruction)

    # Optionally match by basename only (e.g. when GT uses full paths but pred uses filenames)
    if match_by_basename:
        pred_R, pred_C = _remap_to_basename(pred_R, pred_C)
        gt_R, gt_C = _remap_to_basename(gt_R, gt_C)

    # Find common images
    common_images = set(pred_R.keys()) & set(gt_R.keys())

    # Compute metrics
    metrics = {
        "num_images_pred": len(pred_reconstruction.images),
        "num_images_gt": len(gt_reconstruction.images),
        "num_images_common": len(common_images),
    }

    if use_evo:
        # Compute evo trajectory metrics (ATE, ARE, RPE)
        if not EVO_AVAILABLE:
            raise ImportError(
                "evo library is required for trajectory evaluation. "
                "Install with: pip install evo"
            )
        evo_metrics = compute_evo_metrics(pred_R, pred_C, gt_R, gt_C)
        metrics.update(evo_metrics)
    else:
        # Compute standard pairwise AUC metrics
        rotation_errors, translation_errors, pose_errors = compute_pairwise_pose_errors(
            pred_R, pred_C, gt_R, gt_C
        )

        metrics["num_pairs"] = len(pose_errors)

        # Plot pairwise error heatmap
        if save_path is not None and len(pose_errors) > 0:
            N = len(sorted(set(pred_R.keys()) & set(gt_R.keys())))
            error_matrix = np.clip(pose_errors.reshape(N, N), 0, 20)
            fig, ax = plt.subplots(figsize=(8, 7))
            im = ax.imshow(error_matrix, vmin=0, vmax=20, cmap="hot")
            ax.set_xlabel("Image index")
            ax.set_ylabel("Image index")
            ax.set_title("Pairwise Pose Error (degrees)")
            fig.colorbar(im, ax=ax, label="Error (deg)")
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved pairwise error heatmap to {save_path}")

        # Select error array based on error_metric
        if error_metric == "rotation":
            errors_for_auc = rotation_errors
        elif error_metric == "translation":
            errors_for_auc = translation_errors
        else:
            errors_for_auc = pose_errors

        # Compute AUC at each threshold
        pose_aucs = compute_auc_at_thresholds(errors_for_auc, pose_thresholds)
        metrics.update(pose_aucs)

    return metrics
