import logging
import os
import time
import torch

from gluemap.ff_inference.local_inference import create_local_inference
from gluemap.utils.model_loader import load_models

logger = logging.getLogger(__name__)
from gluemap.utils.colmap import write_to_colmap_format
from gluemap.math.scaling import rescale_intrinsics


def restore_intrinsics(intrinsics, images_change, inverse=False):
    scales_curr = [images_change[j] for j in range(len(images_change))]
    intrinscs_curr = [intrinsics[:, j] for j in range(len(scales_curr))]
    intrinscs_curr = torch.stack(rescale_intrinsics(intrinscs_curr, scales_curr, inverse=inverse), dim=1)
    return intrinscs_curr


def _extrinsics_to_global(extrinsics):
    """Convert extrinsics tensor to global_rotations / global_centers dicts.

    Args:
        extrinsics: (1, N, 4, 4) or (1, N, 3, 4) cam-from-world transforms

    Returns:
        global_rotations: dict {img_id: np.array (3,3)}
        global_centers: dict {img_id: np.array (3,)}
    """
    E = extrinsics[0].float().cpu().numpy()  # (N, 4, 4) or (N, 3, 4)
    N = E.shape[0]

    global_rotations = {}
    global_centers = {}
    for i in range(N):
        R_cw = E[i, :3, :3]
        t_cw = E[i, :3, 3]
        global_rotations[i] = R_cw
        global_centers[i] = -R_cw.T @ t_cw

    return global_rotations, global_centers


def _build_global_intrinsics(intrinsics, intrinsics_mapping, num_cameras):
    """Average per-image intrinsics into per-camera intrinsics.

    Args:
        intrinsics: (1, N, 3, 3) tensor in original image coordinates
        intrinsics_mapping: dict {img_id: cam_id}
        num_cameras: number of unique cameras

    Returns:
        list of tensors, indexed by camera_id, each shape (1, 3, 3)
    """
    K = intrinsics[0].float().cpu()  # (N, 3, 3)
    N = K.shape[0]

    # Accumulate intrinsics per camera
    accum = [[] for _ in range(num_cameras)]
    for i in range(N):
        cam_id = intrinsics_mapping[i]
        accum[cam_id].append(K[i])

    global_intrinsics = [None] * num_cameras
    for cam_id in range(num_cameras):
        if accum[cam_id]:
            avg = torch.stack(accum[cam_id]).mean(dim=0)
            global_intrinsics[cam_id] = avg.unsqueeze(0)  # (1, 3, 3)

    return global_intrinsics


def run_direct_inference_pipeline(
    args, dataset_pair, world_size, rank, device, dtype,
    pairs=None, device_id="0", models=None,
):
    """
    Direct backbone inference: load all images, run backbone, write COLMAP.

    Skips SALAD retrieval, two-view inference, star construction, global mapping,
    and refinement. Feeds all images to the backbone in a single forward pass.

    Output is written to {backbone}/ folder inside args.curr_path.

    Args:
        args: Argument namespace (must include chosen_model)
        dataset_pair: Dataset object (used for metadata: images_list, images_shape_ori, etc.)
        world_size: Number of distributed processes
        rank: Current process rank
        device: Torch device
        dtype: Torch dtype
        pairs: Unused (kept for API compatibility)
        device_id: Unused (kept for API compatibility)
        models: Optional pre-loaded models dict

    Returns:
        Tuple of (pred_dir, timing_dict) or (None, timing_dict)
    """
    timing = {}
    t_pipeline_start = time.perf_counter()

    backbone = getattr(args, "chosen_model", "pi3")
    output_dir = backbone
    cache_file = os.path.join(args.curr_path, f"direct_result_{backbone}.pth")

    os.makedirs(args.curr_path, exist_ok=True)
    logger.info(f"[Direct Inference] Running with backbone: {backbone}")

    # Step 1: Load model
    t0 = time.perf_counter()
    if models is not None and backbone in models:
        model = models[backbone]
    else:
        loaded, device = load_models(args, keys={backbone})
        model = loaded[backbone]
    timing["model_loading"] = time.perf_counter() - t0

    # Check cache
    if getattr(args, "force_load", False) and os.path.exists(cache_file):
        logger.info(f"  Loading cached result from {cache_file}")
        cached = torch.load(cache_file, map_location="cpu", weights_only=False)
        global_rotations = cached["global_rotations"]
        global_centers = cached["global_centers"]
        global_intrinsics = cached["global_intrinsics"]
        timing["forward_pass"] = 0.0
    else:
        # Step 2: Prepare images (always load at 518px / patch_size=14 for backbone)
        t0 = time.perf_counter()
        N = len(dataset_pair.images_list)

        from gluemap.utils.load_fn import load_and_preprocess_images
        image_paths = [
            os.path.join(dataset_pair.images_path[i], dataset_pair.images_list[i])
            for i in range(N)
        ]
        force_square = getattr(dataset_pair, "force_square", True)
        images, images_ori, images_change = load_and_preprocess_images(
            image_paths, image_size=518, patch_size=14, force_square=force_square,
        )
        images = images.unsqueeze(0).float()  # (1, N, 3, H, W)
        timing["image_loading"] = time.perf_counter() - t0

        # Step 3: Run backbone forward pass via LocalInference
        t0 = time.perf_counter()
        local_inf = create_local_inference(model, backbone, device, dtype)
        with torch.no_grad():
            predictions = local_inf.predict({"images": images})

        torch.cuda.synchronize()
        timing["forward_pass"] = time.perf_counter() - t0

        # Step 4: Extract poses
        t0 = time.perf_counter()
        extrinsics = predictions["extrinsics"]
        intrinsics = predictions["intrinsics"]

        # Convert extrinsics to global format
        global_rotations, global_centers = _extrinsics_to_global(extrinsics)

        # Rescale intrinsics from preprocessed to original image coordinates
        intrinsics_rescaled = restore_intrinsics(intrinsics, images_change)

        # Build per-camera intrinsics
        num_cameras = max(dataset_pair.intrinsics_mapping.values()) + 1
        global_intrinsics = _build_global_intrinsics(
            intrinsics_rescaled, dataset_pair.intrinsics_mapping, num_cameras
        )
        timing["pose_extraction"] = time.perf_counter() - t0

        # Free GPU memory
        del images, predictions
        torch.cuda.empty_cache()

        # Save cache
        if rank == 0:
            torch.save({
                "global_rotations": global_rotations,
                "global_centers": global_centers,
                "global_intrinsics": global_intrinsics,
            }, cache_file)

    # Step 5: Write COLMAP output (rank 0 only)
    if rank == 0:
        t0 = time.perf_counter()
        colmap_dir = os.path.join(args.curr_path, output_dir)
        logger.info(f"  Writing COLMAP output to: {colmap_dir}")

        write_to_colmap_format(
            colmap_dir,
            dataset_pair.images_shape_ori,
            {},  # predictions_dict (empty, pose_only=True)
            global_rotations,
            global_centers,
            {},  # global_tracks (empty, pose_only=True)
            global_intrinsics,
            dataset_pair.intrinsics_mapping,
            images_list=dataset_pair.images_list,
            pose_only=True,
            camera_type=dataset_pair.camera_model,
        )
        timing["write_colmap"] = time.perf_counter() - t0
        timing["total_pipeline"] = time.perf_counter() - t_pipeline_start

        logger.info(f"[Direct Inference Profiling] Backbone: {backbone}")
        logger.info(f"  Model loading:    {timing.get('model_loading', 0):.2f}s")
        logger.info(f"  Image loading:    {timing.get('image_loading', 0):.2f}s")
        logger.info(f"  Forward pass:     {timing.get('forward_pass', 0):.2f}s")
        logger.info(f"  Pose extraction:  {timing.get('pose_extraction', 0):.2f}s")
        logger.info(f"  Write COLMAP:     {timing.get('write_colmap', 0):.2f}s")
        logger.info(f"  Total pipeline:   {timing['total_pipeline']:.2f}s")

        # Save timing
        timing_path = os.path.join(args.curr_path, f"pipeline_timing_direct_{backbone}.pth")
        torch.save(timing, timing_path)

        return output_dir, timing

    timing["total_pipeline"] = time.perf_counter() - t_pipeline_start
    return None, timing
