"""Utility functions for loading CO3D data from h5py files and pickle ground truth."""

import os
import pickle

import h5py
import numpy as np
import pycolmap
from PIL import Image


def convert_pt3d_RT_to_opencv(Rot, Trans):
    """Convert PyTorch3D extrinsic matrices to OpenCV convention.

    Args:
        Rot: 3x3 rotation matrix in PyTorch3D format
        Trans: 3D translation vector in PyTorch3D format

    Returns:
        3x4 extrinsic matrix [R|t] in OpenCV format
    """
    rot = np.array(Rot, dtype=np.float64)
    trans = np.array(Trans, dtype=np.float64)
    trans[:2] *= -1
    rot[:, :2] *= -1
    rot = rot.T
    return np.hstack((rot, trans[:, None]))


def extract_images_from_h5py(h5py_path, image_names):
    """Extract images from an h5py file to a temporary directory.

    Args:
        h5py_path: Path to the h5py file (e.g., .../apple.110_13051_23361.h5)
        image_names: List of image names to extract (e.g., ['frame000034.jpg', ...])

    Returns:
        Path to the directory containing extracted images
    """
    # Derive output dir from h5py filename
    stem = os.path.splitext(os.path.basename(h5py_path))[0]  # e.g., apple.110_13051_23361
    output_dir = os.path.join("/tmp", "co3d_extracted", f"{stem}_{len(image_names)}img")
    os.makedirs(output_dir, exist_ok=True)

    # Check if already extracted
    existing = set(os.listdir(output_dir))
    if all(name in existing for name in image_names):
        print(f"CO3D images already extracted to {output_dir}")
        return output_dir

    # Extract from h5py
    with h5py.File(h5py_path, "r") as f:
        img_group = f["images"]
        for name in image_names:
            out_path = os.path.join(output_dir, name)
            if os.path.exists(out_path):
                continue
            img_array = img_group[name][()]  # uint8 numpy array (H, W, 3)
            Image.fromarray(img_array).save(out_path)

    print(f"Extracted {len(image_names)} CO3D images to {output_dir}")
    return output_dir


def load_co3d_gt_as_reconstruction(h5py_path, image_names):
    """Load CO3D ground truth from pickle and build a pycolmap Reconstruction.

    Args:
        h5py_path: Path to the h5py file (used to locate the pickle and identify the sequence)
        image_names: List of image names matching the pickle frame order

    Returns:
        pycolmap.Reconstruction with GT poses
    """
    # Parse category and sequence name from h5py filename
    h5_basename = os.path.basename(h5py_path)  # apple.110_13051_23361.h5
    stem = os.path.splitext(h5_basename)[0]  # apple.110_13051_23361
    parts = stem.split(".", 1)
    category = parts[0]
    seq_name = parts[1]

    # Find and load pickle file
    data_dir = os.path.dirname(h5py_path)
    n_imgs = len(image_names)
    pkl_candidates = [
        os.path.join(data_dir, f"co3d_gt_{category}_{n_imgs}img.pkl"),
        os.path.join(data_dir, f"co3d_gt_{category}.pkl"),
    ]
    pkl_path = None
    for candidate in pkl_candidates:
        if os.path.exists(candidate):
            pkl_path = candidate
            break
    if pkl_path is None:
        raise FileNotFoundError(
            f"No CO3D GT pickle found for category={category}, tried: {pkl_candidates}"
        )

    with open(pkl_path, "rb") as f:
        all_data = pickle.load(f)

    if seq_name not in all_data:
        raise KeyError(f"Sequence '{seq_name}' not found in {pkl_path}")

    gt_frames = all_data[seq_name]

    # Build pycolmap Reconstruction
    reconstruction = pycolmap.Reconstruction()

    # Add a dummy camera (evaluation only uses relative poses, not intrinsics)
    camera = pycolmap.Camera(
        camera_id=0,
        model="SIMPLE_PINHOLE",
        width=640,
        height=480,
        params=[500.0, 320.0, 240.0],
    )
    reconstruction.add_camera_with_trivial_rig(camera)

    # Add images with GT poses
    for i, (name, frame_data) in enumerate(zip(image_names, gt_frames)):
        R = np.array(frame_data["R"], dtype=np.float64)
        T = np.array(frame_data["T"], dtype=np.float64)
        extrinsic = convert_pt3d_RT_to_opencv(R, T)

        rot_matrix = extrinsic[:3, :3]
        tvec = extrinsic[:3, 3]

        quat = pycolmap.Rotation3d(rot_matrix).quat
        cam_from_world = pycolmap.Rigid3d(pycolmap.Rotation3d(quat), tvec)

        image = pycolmap.Image(
            image_id=i,
            name=name,
            camera_id=0,
        )
        reconstruction.add_image_with_trivial_frame(image, cam_from_world)

    return reconstruction
