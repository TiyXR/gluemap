import numpy as np
import torch


def get_tracks_dict_indexes(tracks_dict):
    if isinstance(tracks_dict["indexes"], dict):
        indexes = list(tracks_dict["indexes"].keys())
    elif isinstance(tracks_dict["indexes"], list):
        indexes = list(range(len(tracks_dict["indexes"])))

    return indexes


def restore_identity(extrinsics):
    B, N, _, _ = extrinsics.shape

    extrinsics_0 = extrinsics[:, 0:1].clone()
    extrinsics[:, :, :3, :3] = extrinsics[:, :, :3, :3] @ extrinsics_0[
        :, :, :3, :3
    ].transpose(-1, -2).expand(-1, N, -1, -1)
    extrinsics[:, :, :3, 3:] = extrinsics[:, :, :3, 3:] - extrinsics[
        :, :, :3, :3
    ] @ extrinsics_0[:, 0:1, :3, 3:].expand(-1, N, -1, -1)

    return extrinsics


def collect_extrinsics_intrinsics(
    global_rotations, global_centers, global_intrinsics, intrinsics_mapping, indexes
):
    rotations = torch.from_numpy(
        np.stack([global_rotations[idx] for idx in indexes])
    ).unsqueeze(0)
    translations = torch.from_numpy(
        np.stack(
            [
                -global_rotations[idx] @ global_centers[idx].reshape(3, 1)
                for idx in indexes
            ]
        )
    ).unsqueeze(0)

    extrinsics = torch.cat([rotations, translations], dim=-1)  # (B, N, 3, 4)
    intrinsics = (
        torch.stack(
            [global_intrinsics[intrinsics_mapping[idx]] for idx in indexes], dim=1
        )
        .cpu()
        .to(torch.float64)
    )

    return extrinsics, intrinsics


def collect_extrinsics_intrinsics_centered(
    global_rotations, global_centers, global_intrinsics, intrinsics_mapping, indexes
):
    extrinsics, intrinsics = collect_extrinsics_intrinsics(
        global_rotations, global_centers, global_intrinsics, intrinsics_mapping, indexes
    )

    extrinsics = restore_identity(extrinsics)
    return extrinsics, intrinsics


def calculate_index_mappings(query_index, S, device=None):
    new_order = torch.arange(S)
    new_order[0] = query_index
    new_order[query_index] = 0
    if device is not None:
        new_order = new_order.to(device)
    return new_order


def switch_tensor_order(tensors, order, dim=1):
    return [
        torch.index_select(tensor, dim, order) if tensor is not None else None
        for tensor in tensors
    ]


def _rotation_error_deg(R_pred, R_gt):
    """Geodesic rotation error in degrees between two 3x3 rotation matrices."""
    trace_val = np.trace(R_pred.T @ R_gt)
    cos_angle = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _translation_direction_error_deg(t_pred, t_gt):
    """Angular error in degrees between two translation direction vectors."""
    n1 = np.linalg.norm(t_pred)
    n2 = np.linalg.norm(t_gt)
    if n1 < 1e-10 or n2 < 1e-10:
        return 180.0
    cos_angle = np.clip(np.dot(t_pred, t_gt) / (n1 * n2), -1.0, 1.0)
    return np.degrees(np.arccos(cos_angle))


def _lookup_gt(img_id, gt_rotations, gt_centers, images_list):
    """Look up GT rotation and center for an image ID, handling str/int keys."""
    if images_list is not None:
        key = images_list[img_id]
    else:
        key = img_id
    if key not in gt_rotations:
        return None, None
    return np.array(gt_rotations[key]), np.array(gt_centers[key])


def compare_poses_with_gt(sim3_dict, predictions_dict, gt_rotations, gt_centers,
                          images_list=None, return_raw=False):
    """Compare SIM(3) edges and per-star extrinsics against ground truth relative poses.

    Convention: sim3_dict[(A, B)] = (s, R, t) with A=source, B=target,
    i.e. p_B = s * R @ p_A + t. The rotation R maps from star A's frame
    to star B's frame, comparable to GT: R_gt_B @ R_gt_A^T.

    Args:
        sim3_dict: Dict[(int,int), (float, ndarray(3,3), ndarray(3,))]
        predictions_dict: must contain "indexes" and "extrinsics"
        gt_rotations: Dict[key, ndarray(3,3)] — GT camera-from-world rotations
        gt_centers: Dict[key, ndarray(3,)] — GT camera centers
        images_list: optional List[str] mapping image_id → image name for GT lookup

    Returns:
        Dict with summary statistics for sim3 and extrinsic edge errors (in degrees).
    """
    sim3_rot_errors = []
    sim3_trans_errors = []

    # Part A: SIM(3) edges (center-to-center)
    for (A, B), (s_AB, R_AB, t_AB) in sim3_dict.items():
        if A >= B:
            continue
        img_A = predictions_dict["indexes"][A][0]
        img_B = predictions_dict["indexes"][B][0]

        R_gt_A, C_gt_A = _lookup_gt(img_A, gt_rotations, gt_centers, images_list)
        R_gt_B, C_gt_B = _lookup_gt(img_B, gt_rotations, gt_centers, images_list)
        if R_gt_A is None or R_gt_B is None:
            continue

        # GT relative rotation: A→B
        R_gt_rel = R_gt_B @ R_gt_A.T
        sim3_rot_errors.append(_rotation_error_deg(R_AB, R_gt_rel))

        # GT relative translation direction
        t_gt = -R_gt_B @ (C_gt_B - C_gt_A)
        sim3_trans_errors.append(_translation_direction_error_deg(
            t_AB.flatten(), t_gt.flatten()
        ))
    # breakpoint()

    # Part B: Star extrinsic edges (center-to-neighbor)
    ext_rot_errors = []
    ext_trans_errors = []
    indexes = get_tracks_dict_indexes(predictions_dict)

    for idx in indexes:
        img_ids = predictions_dict["indexes"][idx]
        extrinsics = predictions_dict["extrinsics"][idx]  # (1, N, 3, 4)
        img_center = img_ids[0]
        R_gt_center, C_gt_center = _lookup_gt(
            img_center, gt_rotations, gt_centers, images_list
        )
        if R_gt_center is None:
            continue

        N = extrinsics.shape[1]
        for i in range(1, N):
            img_neighbor = img_ids[i]
            R_gt_nb, C_gt_nb = _lookup_gt(
                img_neighbor, gt_rotations, gt_centers, images_list
            )
            if R_gt_nb is None:
                continue

            # Predicted relative pose (camera-from-center → maps center→neighbor)
            if isinstance(extrinsics, torch.Tensor):
                R_pred = extrinsics[0, i, :3, :3].cpu().numpy()
                t_pred = extrinsics[0, i, :3, 3].cpu().numpy()
            else:
                R_pred = extrinsics[0, i, :3, :3]
                t_pred = extrinsics[0, i, :3, 3]

            # GT relative rotation: center→neighbor
            R_gt_rel = R_gt_nb @ R_gt_center.T
            ext_rot_errors.append(_rotation_error_deg(R_pred, R_gt_rel))

            # GT relative translation direction
            t_gt = -R_gt_nb @ (C_gt_nb - C_gt_center)
            ext_trans_errors.append(_translation_direction_error_deg(
                t_pred.flatten(), t_gt.flatten()
            ))

    # Build results
    results = {}

    sim3_rot_errors = np.array(sim3_rot_errors) if sim3_rot_errors else np.array([])
    sim3_trans_errors = np.array(sim3_trans_errors) if sim3_trans_errors else np.array([])
    ext_rot_errors = np.array(ext_rot_errors) if ext_rot_errors else np.array([])
    ext_trans_errors = np.array(ext_trans_errors) if ext_trans_errors else np.array([])

    for prefix, rot_err, trans_err in [
        ("sim3", sim3_rot_errors, sim3_trans_errors),
        ("extrinsics", ext_rot_errors, ext_trans_errors),
    ]:
        if len(rot_err) > 0:
            results[f"{prefix}_rot_mean"] = float(np.mean(rot_err))
            results[f"{prefix}_rot_median"] = float(np.median(rot_err))
            results[f"{prefix}_rot_max"] = float(np.max(rot_err))
            results[f"{prefix}_trans_mean"] = float(np.mean(trans_err))
            results[f"{prefix}_trans_median"] = float(np.median(trans_err))
            results[f"{prefix}_trans_max"] = float(np.max(trans_err))
        else:
            results[f"{prefix}_rot_mean"] = float("nan")
            results[f"{prefix}_rot_median"] = float("nan")
            results[f"{prefix}_rot_max"] = float("nan")
            results[f"{prefix}_trans_mean"] = float("nan")
            results[f"{prefix}_trans_median"] = float("nan")
            results[f"{prefix}_trans_max"] = float("nan")
        results[f"{prefix}_num_edges"] = len(rot_err)

    if return_raw:
        return results, sim3_rot_errors, sim3_trans_errors, ext_rot_errors, ext_trans_errors
    return results
