import logging
import os
from typing import List, Optional

import numpy as np
import pycolmap
import torch

logger = logging.getLogger(__name__)


def camera_from_intrinsics_matrix(
    intrinsics_matrix, camera_model="SIMPLE_PINHOLE", width=None, height=None, camera_id=0
):
    """
    Create a pycolmap.Camera from a 3x3 intrinsics matrix.

    Args:
        intrinsics_matrix: 3x3 numpy array or torch tensor
        camera_model: Camera model string
        width: Image width (defaults to 2 * cx)
        height: Image height (defaults to 2 * cy)
        camera_id: Camera ID

    Returns:
        pycolmap.Camera
    """
    if torch.is_tensor(intrinsics_matrix):
        intrinsics_matrix = intrinsics_matrix.cpu().numpy()

    fx, fy = float(intrinsics_matrix[0, 0]), float(intrinsics_matrix[1, 1])
    cx, cy = float(intrinsics_matrix[0, 2]), float(intrinsics_matrix[1, 2])

    if width is None:
        width = int(2 * cx)
    if height is None:
        height = int(2 * cy)

    camera = pycolmap.Camera.create_from_model_id(
        camera_id, pycolmap.CameraModelId(camera_model), 1.0, width, height
    )
    if len(camera.focal_length_idxs()) == 2:
        camera.focal_length_x = fx
        camera.focal_length_y = fy
    else:
        camera.focal_length = (fx + fy) / 2
    camera.principal_point_x = cx
    camera.principal_point_y = cy
    return camera



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
