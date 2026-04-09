import logging
import os
from typing import List, Optional

import numpy as np
import pycolmap
import torch

logger = logging.getLogger(__name__)


def intrinsics_to_colmap_camera(intrinsics_matrix, camera_model="SIMPLE_PINHOLE"):
    """
    Convert 3x3 intrinsics matrix to a pycolmap.Camera.

    Args:
        intrinsics_matrix: 3x3 numpy array or torch tensor
        camera_model: Camera model string

    Returns:
        pycolmap.Camera
    """
    params = intrinsics_to_colmap_params(intrinsics_matrix, camera_model)
    return pycolmap.Camera(
        camera_id=0, model=camera_model, width=1, height=1, params=params
    )


def intrinsics_to_colmap_params(intrinsics_matrix, camera_model="SIMPLE_PINHOLE"):
    """
    Convert 3x3 intrinsics matrix to COLMAP camera parameter array.

    Args:
        intrinsics_matrix: 3x3 numpy array or torch tensor
        camera_model: Camera model string

    Returns:
        numpy array of camera parameters
    """
    if torch.is_tensor(intrinsics_matrix):
        intrinsics_matrix = intrinsics_matrix.cpu().numpy()

    fx = intrinsics_matrix[0, 0]
    fy = intrinsics_matrix[1, 1]
    cx = intrinsics_matrix[0, 2]
    cy = intrinsics_matrix[1, 2]

    if camera_model in ("SIMPLE_RADIAL", "SIMPLE_PINHOLE"):
        # params: f, cx, cy (+ k for SIMPLE_RADIAL)
        f = (fx + fy) / 2
        params = [f, cx, cy]
    elif camera_model in ("PINHOLE", "RADIAL"):
        # params: fx, fy, cx, cy (+ k for RADIAL)
        params = [fx, fy, cx, cy]
    else:
        raise NotImplementedError(f"Camera model {camera_model} not supported")

    if "RADIAL" in camera_model:
        params.append(0.0)

    return np.array(params, dtype=np.float64)



def extract_gt_intrinsics(
    gt_path: str,
    images_list: List[str],
    intrinsics_mapping: List[int],
    match_by_basename: bool = False,
) -> List[Optional[torch.Tensor]]:
    """Extract GT intrinsics from a COLMAP reconstruction directory.

    Returns List[Optional[torch.Tensor]] indexed by camera_id,
    each tensor shape (1, 3, 3), matching global_intrinsics format.
    """
    gt_recon = pycolmap.Reconstruction()
    gt_recon.read(gt_path)

    name_to_gt_cam = {}
    for img_id, img in gt_recon.images.items():
        key = os.path.basename(img.name) if match_by_basename else img.name
        name_to_gt_cam[key] = gt_recon.cameras[img.camera_id]

    num_cameras = max(intrinsics_mapping) + 1
    gt_intrinsics = [None] * num_cameras

    for i, name in enumerate(images_list):
        cam_id = intrinsics_mapping[i]
        key = os.path.basename(name) if match_by_basename else name
        if key in name_to_gt_cam and gt_intrinsics[cam_id] is None:
            gt_cam = name_to_gt_cam[key]
            K = gt_cam.calibration_matrix()
            gt_intrinsics[cam_id] = torch.tensor(K, dtype=torch.float32).unsqueeze(0)

    num_set = sum(1 for x in gt_intrinsics if x is not None)
    logger.info(f"GT intrinsics loaded: {num_set}/{num_cameras} cameras matched")
    return gt_intrinsics
