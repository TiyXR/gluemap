import logging
import torch
import pycolmap

import numpy as np
import os

logger = logging.getLogger(__name__)

from tqdm import tqdm
from pathlib import Path

from scipy.spatial import cKDTree

from gluemap.math.union_find import UnionFind
from gluemap.utils.colmap_utils import camera_from_intrinsics_matrix


def merge_colmap_databases(
    db_path_primary: str,
    db_path_secondary: str,
    output_path: str = None,
    primary_features_first: bool = True,
) -> str:
    """
    Merge two COLMAP databases by creating a fresh database from scratch.

    Images are matched by name. Features and matches from both databases
    are combined. Cameras from primary are used for common images.

    Args:
        db_path_primary: Path to the primary database (cameras preserved for existing images)
        db_path_secondary: Path to the secondary database (features/matches merged)
        output_path: Path for output database. If None, modifies primary in-place.
        primary_features_first: If True, primary keypoints come first in the merged
            array (default). If False, secondary keypoints come first.

    Returns:
        Path to the merged database
    """
    target_path = output_path if output_path is not None else db_path_primary

    # Remove existing output database if it exists
    if os.path.exists(target_path):
        os.remove(target_path)

    # Open source databases (read-only)
    db_primary = pycolmap.Database.open(db_path_primary)

    db_secondary = pycolmap.Database.open(db_path_secondary)

    # Create fresh output database
    db_output = pycolmap.Database.open(target_path)

    # Read all data from primary
    primary_cameras = {cam.camera_id: cam for cam in db_primary.read_all_cameras()}
    primary_images = {img.name: img for img in db_primary.read_all_images()}

    # Read all data from secondary
    secondary_cameras = {cam.camera_id: cam for cam in db_secondary.read_all_cameras()}
    secondary_images = {img.name: img for img in db_secondary.read_all_images()}

    # Determine common and unique images
    all_names = set(primary_images.keys()) | set(secondary_images.keys())
    common_names = set(primary_images.keys()) & set(secondary_images.keys())

    # Step 1: Write cameras from primary (use primary cameras for common images)
    logger.info("Writing cameras...")
    for cam_id, cam in primary_cameras.items():
        new_cam = pycolmap.Camera(
            camera_id=cam_id,
            model=cam.model_name,
            params=cam.params,
            width=cam.width,
            height=cam.height,
        )
        db_output.write_camera(new_cam)

    # Add any additional cameras from secondary (for images only in secondary)
    max_camera_id = max(primary_cameras.keys()) if primary_cameras else 0
    secondary_cam_to_output_cam = {}
    for name in secondary_images.keys():
        if name not in primary_images:
            sec_img = secondary_images[name]
            if sec_img.camera_id not in secondary_cam_to_output_cam:
                max_camera_id += 1
                secondary_cam_to_output_cam[sec_img.camera_id] = max_camera_id
                sec_cam = secondary_cameras[sec_img.camera_id]
                new_cam = pycolmap.Camera(
                    camera_id=max_camera_id,
                    model=sec_cam.model_name,
                    params=sec_cam.params,
                    width=sec_cam.width,
                    height=sec_cam.height,
                )
                db_output.write_camera(new_cam)

    # Step 2: Write images and track keypoint offsets
    name_to_output_image_id = {}
    primary_offsets = {}   # image_name -> index offset for primary keypoints
    secondary_offsets = {} # image_name -> index offset for secondary keypoints

    logger.info("Writing images and merging keypoints...")
    for name in all_names:
        if name in primary_images:
            pri_img = primary_images[name]
            new_img = pycolmap.Image()
            new_img.name = name
            new_img.image_id = pri_img.image_id
            new_img.camera_id = pri_img.camera_id
            db_output.write_image(new_img, use_image_id=True)
            name_to_output_image_id[name] = pri_img.image_id
        else:
            sec_img = secondary_images[name]
            new_img = pycolmap.Image()
            new_img.name = name
            new_img.image_id = sec_img.image_id
            new_img.camera_id = secondary_cam_to_output_cam[sec_img.camera_id]
            db_output.write_image(new_img, use_image_id=True)
            name_to_output_image_id[name] = sec_img.image_id

    # Step 3: Merge and write keypoints
    logger.info("Merging and writing keypoints...")
    for name in all_names:
        output_image_id = name_to_output_image_id[name]

        pri_kp = None
        sec_kp = None

        if name in primary_images:
            pri_kp = db_primary.read_keypoints(primary_images[name].image_id)
        if name in secondary_images:
            sec_kp = db_secondary.read_keypoints(secondary_images[name].image_id)

        # Compute per-source offsets and merge keypoints
        pri_kp_xy = pri_kp[:, :2] if pri_kp is not None and len(pri_kp) > 0 else None
        sec_kp_xy = sec_kp[:, :2] if sec_kp is not None and len(sec_kp) > 0 else None

        if pri_kp_xy is not None and sec_kp_xy is not None:
            if primary_features_first:
                merged_kp = np.vstack([pri_kp_xy, sec_kp_xy])
                primary_offsets[name] = 0
                secondary_offsets[name] = len(pri_kp_xy)
            else:
                merged_kp = np.vstack([sec_kp_xy, pri_kp_xy])
                secondary_offsets[name] = 0
                primary_offsets[name] = len(sec_kp_xy)
        elif pri_kp_xy is not None:
            merged_kp = pri_kp_xy
            primary_offsets[name] = 0
            secondary_offsets[name] = 0
        elif sec_kp_xy is not None:
            merged_kp = sec_kp_xy
            primary_offsets[name] = 0
            secondary_offsets[name] = 0
        else:
            merged_kp = None
            primary_offsets[name] = 0
            secondary_offsets[name] = 0

        if merged_kp is not None and len(merged_kp) > 0:
            db_output.write_keypoints(output_image_id, merged_kp)

    # Step 4: Collect all matches from both databases
    all_matches = {}  # (out_id1, out_id2) -> merged matches array

    # Read primary matches
    primary_id_to_name = {img.image_id: name for name, img in primary_images.items()}
    pair_ids_pri, matches_list_pri = db_primary.read_all_matches()
    for pair_id, matches in zip(pair_ids_pri, matches_list_pri):
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        if matches is None or len(matches) == 0:
            continue
        name1 = primary_id_to_name.get(id1)
        name2 = primary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        out_id1 = name_to_output_image_id[name1]
        out_id2 = name_to_output_image_id[name2]
        offset1 = primary_offsets.get(name1, 0)
        offset2 = primary_offsets.get(name2, 0)
        key = (min(out_id1, out_id2), max(out_id1, out_id2))
        remapped = matches.copy()
        if offset1 > 0 or offset2 > 0:
            if out_id1 <= out_id2:
                remapped[:, 0] += offset1
                remapped[:, 1] += offset2
            else:
                remapped[:, 0] += offset2
                remapped[:, 1] += offset1
        if key not in all_matches:
            all_matches[key] = []
        all_matches[key].append(remapped)
    
    logger.info("Merging matches from secondary database...")
    
    # Read secondary matches (with offset)
    secondary_id_to_name = {img.image_id: name for name, img in secondary_images.items()}
    pair_ids_sec, matches_list_sec = db_secondary.read_all_matches()
    for pair_id, matches in zip(pair_ids_sec, matches_list_sec):
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        if matches is None or len(matches) == 0:
            continue
        name1 = secondary_id_to_name.get(id1)
        name2 = secondary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        out_id1 = name_to_output_image_id[name1]
        out_id2 = name_to_output_image_id[name2]
        offset1 = secondary_offsets.get(name1, 0)
        offset2 = secondary_offsets.get(name2, 0)
        key = (min(out_id1, out_id2), max(out_id1, out_id2))
        remapped = matches.copy()
        if offset1 > 0 or offset2 > 0:
            if out_id1 <= out_id2:
                remapped[:, 0] += offset1
                remapped[:, 1] += offset2
            else:
                remapped[:, 0] += offset2
                remapped[:, 1] += offset1
        if key not in all_matches:
            all_matches[key] = []
        all_matches[key].append(remapped)

    # Write merged matches
    for (id1, id2), match_list in all_matches.items():
        if len(match_list) > 0:
            merged = np.vstack(match_list)
            db_output.write_matches(id1, id2, merged)

    # Step 5: Collect all two-view geometries
    all_geometries = {}  # (out_id1, out_id2) -> (inliers_list, config)

    # Read primary geometries
    for pair_id in pair_ids_pri:
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        name1 = primary_id_to_name.get(id1)
        name2 = primary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        try:
            geom = db_primary.read_two_view_geometry(id1, id2)
            if geom is not None and geom.inlier_matches is not None and len(geom.inlier_matches) > 0:
                out_id1 = name_to_output_image_id[name1]
                out_id2 = name_to_output_image_id[name2]
                offset1 = primary_offsets.get(name1, 0)
                offset2 = primary_offsets.get(name2, 0)
                key = (min(out_id1, out_id2), max(out_id1, out_id2))
                remapped = geom.inlier_matches.copy()
                if offset1 > 0 or offset2 > 0:
                    if out_id1 <= out_id2:
                        remapped[:, 0] += offset1
                        remapped[:, 1] += offset2
                    else:
                        remapped[:, 0] += offset2
                        remapped[:, 1] += offset1
                if key not in all_geometries:
                    all_geometries[key] = ([], geom.config)
                all_geometries[key][0].append(remapped)
        except:
            pass

    # Read secondary geometries (with offset)
    for pair_id in pair_ids_sec:
        id1, id2 = pycolmap.pair_id_to_image_pair(pair_id)
        name1 = secondary_id_to_name.get(id1)
        name2 = secondary_id_to_name.get(id2)
        if name1 is None or name2 is None:
            continue
        try:
            geom = db_secondary.read_two_view_geometry(id1, id2)
            if geom is not None and geom.inlier_matches is not None and len(geom.inlier_matches) > 0:
                out_id1 = name_to_output_image_id[name1]
                out_id2 = name_to_output_image_id[name2]
                offset1 = secondary_offsets.get(name1, 0)
                offset2 = secondary_offsets.get(name2, 0)
                key = (min(out_id1, out_id2), max(out_id1, out_id2))
                remapped = geom.inlier_matches.copy()
                if offset1 > 0 or offset2 > 0:
                    if out_id1 <= out_id2:
                        remapped[:, 0] += offset1
                        remapped[:, 1] += offset2
                    else:
                        remapped[:, 0] += offset2
                        remapped[:, 1] += offset1
                if key not in all_geometries:
                    all_geometries[key] = ([], geom.config)
                all_geometries[key][0].append(remapped)
        except:
            pass

    # Write merged two-view geometries
    for (id1, id2), (inliers_list, config) in all_geometries.items():
        if len(inliers_list) > 0:
            merged_inliers = np.vstack(inliers_list)
            new_geom = pycolmap.TwoViewGeometry()
            new_geom.inlier_matches = merged_inliers
            new_geom.config = config
            db_output.write_two_view_geometry(id1, id2, new_geom)

    # Close all databases
    db_primary.close()
    db_secondary.close()
    db_output.close()

    logger.info(f"Created merged database with {len(all_names)} images")
    return target_path


def remap_cameras_to_intrinsics(database, images_list, intrinsics_mapping):
    """
    Remap camera IDs to be consistent with intrinsics_mapping values.

    After pycolmap.import_images/extract_features creates cameras with arbitrary IDs,
    this function:
    1. Reads existing cameras and their parameters
    2. Updates/writes cameras with IDs = intrinsics_mapping_value + 1
    3. Updates all images to reference the new camera IDs

    Args:
        database: pycolmap.Database instance
        images_list: List of image names
        intrinsics_mapping: Dict mapping image index to intrinsics group index

    Returns:
        idx_to_image_id: Dict mapping image index to database image_id
    """
    # Read cameras and build mapping from old camera_id to camera params
    cameras = database.read_all_cameras()
    old_camera_params = {cam.camera_id: cam for cam in cameras}
    existing_camera_ids = set(old_camera_params.keys())

    # Track which old camera to use for each intrinsics group (first occurrence)
    intrinsics_to_old_camera = {}
    idx_to_image_id = {}
    for i in range(len(images_list)):
        image = database.read_image_with_name(images_list[i])
        idx_to_image_id[i] = image.image_id
        intrinsics_val = intrinsics_mapping[i]
        if intrinsics_val not in intrinsics_to_old_camera:
            intrinsics_to_old_camera[intrinsics_val] = image.camera_id

    # Create/update cameras with correct IDs (intrinsics_val + 1)
    new_cameras = {}
    for intrinsics_val, old_cam_id in intrinsics_to_old_camera.items():
        new_cam_id = intrinsics_val + 1  # 1-indexed for COLMAP
        old_cam = old_camera_params[old_cam_id]

        new_cam = pycolmap.Camera(
            camera_id=new_cam_id,
            model=old_cam.model_name,
            params=old_cam.params,
            width=old_cam.width,
            height=old_cam.height,
        )

        if new_cam_id in existing_camera_ids:
            # Update existing camera with new params
            database.update_camera(new_cam)
        else:
            # Write new camera
            database.write_camera(new_cam)
        new_cameras[new_cam_id] = new_cam

    # Create rigs for new camera IDs (rig_id == camera_id in trivial setup)
    existing_rigs = {rig.rig_id for rig in database.read_all_rigs()}
    for new_cam_id, cam in new_cameras.items():
        if new_cam_id not in existing_rigs:
            rig = pycolmap.Rig()
            rig.rig_id = new_cam_id
            rig.add_ref_sensor(cam.sensor_id)
            database.write_rig(rig, use_rig_id=True)

    # Update images and frames to reference new camera/rig IDs
    for i in range(len(images_list)):
        image = database.read_image_with_name(images_list[i])
        image.camera_id = intrinsics_mapping[i] + 1
        frame = database.read_frame(image.frame_id)
        frame.rig_id = intrinsics_mapping[i] + 1
        frame.clear_data_ids()
        frame.add_data_id(image.data_id)
        database.update_frame(frame)
        database.update_image(image)

    return idx_to_image_id


def establish_keypoints_and_correspondences(
    tracks_dict,
    N,
    add_tracks,
    add_virtual_points,
    device="cuda",
    use_1_indexed=True,
):
    """
    Establish keypoints and correspondences from tracks.

    This function collects 2D points from tracks, builds point index mappings,
    and establishes correspondences between image pairs.

    Args:
        tracks_dict: Dictionary containing tracks, scores, indexes, etc.
        N: Number of images
        add_tracks: Whether to add real tracks
        add_virtual_points: Whether to add virtual points
        device: CUDA device for tensor operations
        use_1_indexed: If True, correspondences use 1-indexed image IDs (COLMAP style).
                       If False, use 0-indexed image IDs.

    Returns:
        keypoints_per_image: Dict mapping image_id to keypoints numpy array
        correspondences: Dict mapping (image_id1, image_id2) to list of (pt_idx1, pt_idx2)
        images_points2d: Dict mapping image_id to list of 2D points
        images_points2d_virtual: Dict mapping image_id to list of virtual 2D points
        images_points2d_virtual_isnegative: Dict mapping image_id to list of is_negative flags
    """
    indexes = range(len(tracks_dict["indexes"]))

    # Initialize data structures
    images_points2d = {i: [] for i in range(N)}  # (image_id, [point_2d, ...])
    images_points2d_virtual = {i: [] for i in range(N)}  # (image_id, [point_2d, ...])
    images_points2d_virtual_isnegative = {
        i: [] for i in range(N)
    }  # (image_id, [is_negative, ...])

    N_indexes = {idx: tracks_dict["tracks"][idx].shape[-3] for idx in indexes}
    pts2d_idx_all = {
        idx: np.ones([N_indexes[idx], tracks_dict["tracks"][idx].shape[-2]], dtype=int)
        * -1
        for idx in indexes
    }  # N x K

    if add_virtual_points or not add_tracks:
        pts2d_idx_virtual_all = {
            idx: np.zeros(
                [N_indexes[idx], tracks_dict["tracks_virtual"][idx].shape[-2]],
                dtype=int,
            )
            for idx in indexes
        }  # N x K
    else:
        pts2d_idx_virtual_all = None

    # Initialize inverse maps: image_id -> list of (idx, i, j) tuples
    pts2d_idx_inv = {i: [] for i in range(N)}  # inverse map for real tracks

    if add_virtual_points or not add_tracks:
        pts2d_idx_virtual_inv = {i: [] for i in range(N)}  # inverse map for virtual
    else:
        pts2d_idx_virtual_inv = None

    # Add the points
    logger.info("Adding points to dictionary...")
    for idx in tqdm(indexes):
        scores = tracks_dict["scores"][idx]
        if add_tracks:
            for i in range(tracks_dict["tracks"][idx].shape[1]):
                valid = scores[0, i, :] > 0.0

                valid_idx = np.where(valid)[0].tolist()

                idx_inner = tracks_dict["indexes"][idx][i]

                indexes_pts = list(
                    range(
                        len(images_points2d[idx_inner]),
                        len(images_points2d[idx_inner]) + len(valid_idx),
                    )
                )
                pts2d_idx_all[idx][i, valid_idx] = indexes_pts
                images_points2d[idx_inner] += [
                    tracks_dict["tracks"][idx][0, i, j] for j in valid_idx
                ]
                # Build inverse map: for each point added, record its origin (idx, i, j)
                for j in valid_idx:
                    pts2d_idx_inv[idx_inner].append((idx, i, j))

        if not add_virtual_points and add_tracks:
            continue

        for i in range(tracks_dict["tracks_virtual"][idx].shape[1]):
            scores = tracks_dict["valid_virtual"][idx]
            valid = scores[0, i, :] > 0.0

            valid_idx = np.where(valid)[0].tolist()
            idx_inner = tracks_dict["indexes"][idx][i]

            indexes_pts = list(
                range(
                    len(images_points2d_virtual[idx_inner]),
                    len(images_points2d_virtual[idx_inner]) + len(valid_idx),
                )
            )
            pts2d_idx_virtual_all[idx][i, valid_idx] = indexes_pts
            images_points2d_virtual[idx_inner] += [
                tracks_dict["tracks_virtual"][idx][0, i, j] for j in valid_idx
            ]
            images_points2d_virtual_isnegative[idx_inner] += (
                tracks_dict["isnegative_virtual"][idx][0, i, valid_idx]
                .cpu()
                .numpy()
                .astype(int)
                .tolist()
            )
            # Build inverse map for virtual points
            for j in valid_idx:
                pts2d_idx_virtual_inv[idx_inner].append((idx, i, j))

    # Build keypoints per image
    keypoints_per_image = {}

    logger.info("Constructing correspondences...")
    for idx in tqdm(range(N)):
        uf_pts2d = UnionFind()
        if len(images_points2d[idx]) == 0 and len(images_points2d_virtual[idx]) == 0:
            continue

        if len(images_points2d[idx]) == 0:
            images_points2d_virtual_tensor = torch.stack(
                images_points2d_virtual[idx], dim=0
            ).to(torch.float32)

            # Store keypoints for this image
            keypoints_per_image[idx] = images_points2d_virtual_tensor.cpu().numpy()

            continue

        # Use KDTree for efficient near-duplicate detection (O(N) memory vs O(N^2))
        images_points2d_tensor = (
            torch.stack(images_points2d[idx], dim=0).to(torch.float32)
        )

        # Find near-duplicate points using KDTree on CPU
        pts_np = images_points2d_tensor.numpy()
        tree = cKDTree(pts_np)
        pairs = tree.query_pairs(r=1e-3, output_type='ndarray')
        for i, j in pairs:
            uf_pts2d.union(int(i), int(j))

        if len(images_points2d_virtual[idx]) > 0:
            images_points2d_virtual_tensor = torch.stack(
                images_points2d_virtual[idx], dim=0
            ).to(torch.float32)
            images_points2d_tensor = torch.cat(
                [images_points2d_tensor, images_points2d_virtual_tensor], dim=0
            )

        # Store keypoints for this image
        keypoints_per_image[idx] = images_points2d_tensor.cpu().numpy()

        # Update the indexes in pts2d_idx_all and pts2d_idx_virtual_all
        for idx_inner in indexes:
            for i in range(tracks_dict["tracks"][idx_inner].shape[1]):
                if tracks_dict["indexes"][idx_inner][i] != idx:
                    continue
                for j in range(tracks_dict["tracks"][idx_inner].shape[2]):
                    if pts2d_idx_all[idx_inner][i, j] == -1:
                        continue
                    pts2d_idx_all[idx_inner][i, j] = uf_pts2d.find(
                        pts2d_idx_all[idx_inner][i, j]
                    )

        uf_pts2d.clear()

    # Establish the concrete correspondences
    # Just change the point index a bit if two points are very close to each other
    correspondences = (
        {}
    )  # key is (image_id1, image_id2), value is a list of (point2d_id1, point2d_id2)
    logger.info("Concatenating tracks...")
    for idx in tqdm(indexes):

        # Note: since in GLOMAP, blind concatenation is used, so we only need to consider the pairs with the center image as one of the image
        for i, idx_inner in enumerate(tracks_dict["indexes"][idx]):
            if i == 0:
                continue
            if idx_inner < tracks_dict["indexes"][idx][0]:
                i1 = i
                i2 = 0
                idx_inner1 = idx_inner
                idx_inner2 = tracks_dict["indexes"][idx][0]
            else:
                i1 = 0
                i2 = i
                idx_inner1 = tracks_dict["indexes"][idx][0]
                idx_inner2 = idx_inner

            offset = 1 if use_1_indexed else 0
            key = (idx_inner1 + offset, idx_inner2 + offset)

            if add_tracks:
                scores = tracks_dict["scores"][idx].cpu()
                common_points = (scores[0, i1, :] * scores[0, i2, :]) > 0.0
                index_j = np.where(common_points)[0]

                corres_real = np.stack(
                    [pts2d_idx_all[idx][i1, index_j], pts2d_idx_all[idx][i2, index_j]],
                    axis=1,
                )
            else:
                corres_real = []

            # Add the virtual correspondences
            if add_virtual_points:
                scores_virtual = tracks_dict["valid_virtual"][idx]
                len1 = len(images_points2d[idx_inner1])
                len2 = len(images_points2d[idx_inner2])
                common_points_virtual = (
                    scores_virtual[0, i1, :] * scores_virtual[0, i2, :]
                ) > 0.0
                index_virtual_j = np.where(common_points_virtual)[0]
                corres_virtual = np.stack(
                    [
                        pts2d_idx_virtual_all[idx][i1, index_virtual_j] + len1,
                        pts2d_idx_virtual_all[idx][i2, index_virtual_j] + len2,
                    ],
                    axis=1,
                )
            else:
                corres_virtual = []

            if key not in correspondences:
                correspondences[key] = []

            if len(corres_real) > 0:
                correspondences[key].append(corres_real)
            if len(corres_virtual) > 0:
                correspondences[key].append(corres_virtual)

    for key in correspondences.keys():
        if len(correspondences[key]) == 0:
            continue
        correspondences[key] = np.unique(
            np.concatenate(correspondences[key]), axis=0
        ).tolist()

    return (
        keypoints_per_image,
        correspondences,
        images_points2d,
        images_points2d_virtual,
        images_points2d_virtual_isnegative,
        pts2d_idx_all,
        pts2d_idx_virtual_all,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
    )


# poses_rel, poses_rel_scores, intrinsics_all, neighbors, tracks_dict
def prepare_glomap_prior(
    dir_write,
    images_shape_ori,
    images_list,
    global_rotations,
    global_centers,
    poses_rel,
    poses_rel_scores,
    global_intrinsics,
    tracks_dict,
    intrinsics_mapping,
    camera_model="SIMPLE_RADIAL",
    add_tracks=True,
    add_virtual_points=True,
    database_name="database.db",
):
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)

    # Perform intrinsics averaging here
    indexes = range(len(tracks_dict["indexes"]))
    members = [tracks_dict["indexes"][idx] for idx in indexes]

    # Write colmap database + prior
    prepare_database_prior(
        dir_write,
        images_shape_ori,
        tracks_dict,
        global_rotations,
        global_centers,
        global_intrinsics,
        intrinsics_mapping,
        poses_rel,
        poses_rel_scores,
        images_list,
        camera_model=camera_model,
        add_tracks=add_tracks,
        add_virtual_points=add_virtual_points,
        database_name=database_name,
    )


def prepare_database_prior(
    dir_write,
    images_shape_ori,
    tracks_dict,
    global_rotations,
    global_centers,
    global_intrinsics,
    intrinsics_mapping,
    poses_rel,
    poses_rel_scores,
    images_list,
    camera_model="SIMPLE_RADIAL",
    add_tracks=True,
    add_virtual_points=True,
    database_name="database.db",
    device="cuda",
):
    database_path = dir_write + "/" + database_name

    if os.path.exists(database_path):
        os.remove(database_path)
    database = pycolmap.Database.open(database_path)

    camera_sizes = [None for _ in range(len(global_intrinsics))]
    for i in range(len(intrinsics_mapping)):
        camera_id = intrinsics_mapping[i]
        if camera_sizes[camera_id] is None:
            camera_sizes[camera_id] = images_shape_ori[i]

    # Write cameras
    cameras_colmap = {}
    for i in range(len(global_intrinsics)):
        if global_intrinsics[i] is None:
            continue

        camera = camera_from_intrinsics_matrix(
            global_intrinsics[i][0], camera_model, camera_sizes[i][-1], camera_sizes[i][0], i + 1
        )

        database.write_camera(camera)
        cameras_colmap[i] = camera

    logger.info("Write cameras to database done")

    N = len(images_shape_ori)

    # Establish keypoints and correspondences
    (
        keypoints_per_image,
        correspondences,
        images_points2d,
        images_points2d_virtual,
        images_points2d_virtual_isnegative,
        _,
        _,
        _,
        _,
    ) = establish_keypoints_and_correspondences(
        tracks_dict,
        N,
        add_tracks,
        add_virtual_points,
        device,
    )

    # Add images
    logger.info("Add images to database...")
    for idx in tqdm(range(N)):
        image = pycolmap.Image()
        image.camera_id = intrinsics_mapping[idx] + 1

        if images_list is not None:
            image.name = images_list[idx]
        else:
            image.name = str(idx)
        image.image_id = idx + 1

        database.write_image(image, use_image_id=True)

    # Write keypoints to database
    for idx, keypoints in keypoints_per_image.items():
        database.write_keypoints(idx + 1, keypoints)

    # For each correspondence, collect into a numpy array and write to the database
    logger.info("Write matches to database...")
    for key in tqdm(correspondences.keys()):
        image_id1 = key[0]
        image_id2 = key[1]

        if len(correspondences[key]) < 3:
            continue

        two_view_geometry = pycolmap.TwoViewGeometry()
        two_view_geometry.inlier_matches = np.array(correspondences[key])
        two_view_geometry.config = 2

        database.write_two_view_geometry(image_id1, image_id2, two_view_geometry)
        database.write_matches(image_id1, image_id2, np.array(correspondences[key]))

    database.close()

    # Required data structures
    #    - num_image num_pair
    #    - N lines of image prior info:
    #      IMAGE_NAME VIRTUAL_START
    #    - M lines of image pair info:
    #      IMAGE_NAME_1 IMAGE_NAME_2 QW QX QY QZ TX TY TZ WEIGHT

    # Need to calculate the number of valid poses, so we prepare the relative poses to be written first
    lines = []
    for key in poses_rel.keys():
        idx1 = key[0]
        idx2 = key[1]
        if idx1 >= idx2 and (idx2, idx1) in poses_rel.keys():
            continue

        cam2_from_cam1 = pycolmap.Rigid3d(poses_rel[key])
        score = poses_rel_scores[key]
        # If both way is available, write the average as the prior
        if (idx2, idx1) in poses_rel.keys():

            cam1_from_cam2 = pycolmap.Rigid3d(poses_rel[(idx2, idx1)])
            cam2_from_cam1_v2 = cam1_from_cam2.inverse()

            # Take the average of the two as the prior
            quat_sum = cam2_from_cam1.rotation.quat + cam2_from_cam1_v2.rotation.quat
            cam2_from_cam1.rotation.quat = quat_sum / np.linalg.norm(quat_sum)

            # Since the translation are not with the same scale, so need to scale both to unit length
            trans_sum = cam2_from_cam1.translation / np.linalg.norm(
                cam2_from_cam1.translation
            ) + cam2_from_cam1_v2.translation / np.linalg.norm(
                cam2_from_cam1_v2.translation
            )
            cam2_from_cam1.translation = trans_sum / 2

            score = (score + poses_rel_scores[(idx2, idx1)]) / 2

        if score <= 0:
            continue
        qx, qy, qz, qw = cam2_from_cam1.rotation.quat
        tx, ty, tz = cam2_from_cam1.translation
        weight = score

        lines.append(
            images_list[idx1]
            + " "
            + images_list[idx2]
            + " "
            + str(qw)
            + " "
            + str(qx)
            + " "
            + str(qy)
            + " "
            + str(qz)
            + " "
            + str(tx)
            + " "
            + str(ty)
            + " "
            + str(tz)
            + " "
            + str(weight)
            + "\n"
        )

    num_valid_pose = len(lines)
    file = open(dir_write + "/prior.txt", "w")

    file.write(str(len(images_list)) + " " + str(num_valid_pose) + "\n")
    # if add_virtual_points:
    # else:
    #     file.write("0 " + str(num_valid_pose) + "\n")

    for i in range(len(images_list)):
        # file.write(images_list[i] + " " + str(len(images_points2d[i])) + "\n")

        cam_from_world = pycolmap.Rigid3d(
            np.concatenate(
                [global_rotations[i][:3, :3].T, global_centers[i].reshape(3, 1)], axis=1
            )
        ).inverse()
        qx, qy, qz, qw = cam_from_world.rotation.quat
        tx, ty, tz = cam_from_world.translation
        if add_virtual_points or not add_tracks:
            file.write(
                images_list[i]
                + " "
                + str(qw)
                + " "
                + str(qx)
                + " "
                + str(qy)
                + " "
                + str(qz)
                + " "
                + str(tx)
                + " "
                + str(ty)
                + " "
                + str(tz)
                + " "
                + str(len(images_points2d[i]))
                + "\n"
            )
            is_negative_index = (
                np.arange(len(images_points2d_virtual[i]))[
                    np.array(images_points2d_virtual_isnegative[i]) == 1
                ]
                + len(images_points2d[i])
            ).tolist()
            file.write(
                " ".join(map(str, [len(is_negative_index)] + is_negative_index)) + "\n"
            )
        else:
            file.write(
                images_list[i]
                + " "
                + str(qw)
                + " "
                + str(qx)
                + " "
                + str(qy)
                + " "
                + str(qz)
                + " "
                + str(tx)
                + " "
                + str(ty)
                + " "
                + str(tz)
                + " -1\n"
            )
            file.write("0\n")

    for line in lines:
        file.write(line)

    file.close()
    # return database


# for the case pairs are directly given, set the selected_seeds to None and neighbors to the pairs
def prepare_sift_database(
    dir_write,
    images_path,
    images_list,
    intrinsics_mapping,
    neighbors,
    selected_seeds,
    device="cuda",
    extraction_method="sift",
    camera_model="SIMPLE_RADIAL",
    skip_matching=False,
    remove_existing=True,
):

    datbase_dir = dir_write + f"/database_{extraction_method}.db"
    if os.path.exists(datbase_dir) and remove_existing:
        os.remove(datbase_dir)

    # camera_mode = "PER_FOLDER" if use_same_camera else "PER_IMAGE"
    camera_mode = "PER_IMAGE"

    images_list = [str(Path(p)) for p in images_list]

    # Extract the features for all images
    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model
    pycolmap.extract_features(
        datbase_dir, images_path, images_list, camera_mode, reader_opts,
    )  # use the same camera model for all images

    database = pycolmap.Database.open(datbase_dir)
    idx_to_image_id = remap_cameras_to_intrinsics(database, images_list, intrinsics_mapping)

    if skip_matching:
        return True

    # Run matching on all the matched neighbors
    matched_pairs = []
    if not selected_seeds is None:
        for idx in selected_seeds:
            for i in range(len(neighbors[idx])):
                for j in range(len(neighbors[idx])):
                    if neighbors[idx][i] >= neighbors[idx][j]:
                        continue
                    matched_pairs.append(
                        (neighbors[idx][i].item(), neighbors[idx][j].item())
                    )
    else:
        matched_pairs = neighbors.tolist()
        matched_pairs = [(i, j) for i, j in matched_pairs]  # Ensure unique pairs

    matched_pairs = list(set(matched_pairs))

    # Write the match pair to a file
    with open(dir_write + "/pairs.txt", "w") as f:
        for i, j in matched_pairs:
            f.write(f"{images_list[i]} {images_list[j]}\n")

    if extraction_method == "sift":
        # Run matching
        # pycolmap.verify_matches(datbase_dir, dir_write + "/pairs.txt")
        cmd = (
            "colmap matches_importer --match_type pairs --database_path "
            + datbase_dir
            + " --match_list_path "
            + dir_write
            + "/pairs.txt --FeatureMatching.gpu_index "
        )
        # Get the available GPU index
        gpus = [idx for idx in range(torch.cuda.device_count())]
        cmd += str(",".join(map(str, gpus)))

        ite_num = 0
        while ite_num < 10:
            logger.info(cmd)
            status = os.system(cmd)
            if status == 0:
                break
            ite_num += 1

    return True

