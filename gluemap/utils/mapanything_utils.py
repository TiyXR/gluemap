from thirdparty.mapanything.utils.image import preprocess_inputs
from thirdparty.mapanything.utils.geometry import closed_form_pose_inverse

import torch


def compose_input_views(images, global_rotations=None, global_centers=None, global_intrinsics=None):
    import numpy as np

    if images.ndim == 4:
        images = images.unsqueeze(0)
    input_views = []
    for i in range(images.shape[1]):
        view = {"img": images[0, i].permute(1, 2, 0)}  # (H, W, 3)
        if global_rotations is not None:
            camera_pose = np.eye(4, dtype=np.float32)
            camera_pose[:3, :3] = global_rotations[0, i].T
            camera_pose[:3, 3] = global_centers[0, i]
            view["camera_poses"] = camera_pose

        if global_intrinsics is not None:
            view["intrinsics"] = global_intrinsics[0, i]

        input_views.append(view)

    input_views = preprocess_inputs(input_views)
    return input_views


@torch.no_grad()
def mapanything_inference(model, images, device="cuda", dtype=torch.bfloat16):
    images = images.to(device)
    processed_views = compose_input_views(images)

    predictions = model.infer(
        processed_views,
        memory_efficient_inference=False,
        use_amp=True,
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=True,
        apply_confidence_mask=False,
        confidence_percentile=10,
        ignore_calibration_inputs=False,
        ignore_depth_inputs=False,
        ignore_pose_inputs=False,
        ignore_depth_scale_inputs=False,
        ignore_pose_scale_inputs=False,
    )

    predictions = retrieve_mapanything_result(predictions)
    return predictions


def retrieve_mapanything_result(predictions):
    extrinsics = []
    intrinsics = []
    depths = []
    confs = []

    for view_idx, pred in enumerate(predictions):
        intrinsics_torch = pred["intrinsics"][0]  # (3, 3)
        camera_pose_torch = pred["camera_poses"][0]  # (4, 4)
        extrinsics.append(camera_pose_torch)
        intrinsics.append(intrinsics_torch)
        depths.append(pred["depth_z"][0, :, :])  # (H, W, 1)
        confs.append(pred["conf"][0, :, :])  # (H, W)

    depth_z = torch.stack(depths, dim=0).unsqueeze(0)  # (1, N, H, W)
    conf_z = torch.stack(confs, dim=0).unsqueeze(0)  # (1, N, H, W)

    extrinsics = closed_form_pose_inverse(torch.stack(extrinsics, dim=0)).unsqueeze(0)
    intrinsics = torch.stack(intrinsics, dim=0).unsqueeze(0)

    predictions = {
        "depth": depth_z,
        "depth_conf": conf_z,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
    }
    return predictions
