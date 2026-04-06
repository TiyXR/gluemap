"""
Track establishment module using pycolmap.

Converts correspondences from establish_keypoints_and_correspondences()
into pycolmap Point3D tracks using Union-Find.

This implements the GLOMAP EstablishTracks algorithm from:
/home/linpan/workspace/colmap/src/glomap/sfm/global_mapper.cc (lines 122-256)
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import numpy as np

import pycolmap

from gluemap.math.union_find import UnionFind
from gluemap.utils.prepare_prior import establish_keypoints_and_correspondences

import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackEstablishmentOptions:
    """Options for track establishment."""

    # Max pixel distance between observations of same track within one image
    track_intra_image_consistency_threshold: float = 10.0
    # Minimum number of views per track
    track_min_num_views_per_track: int = 3
    # Required number of tracks per view before early stopping (inf = no limit)
    track_required_tracks_per_view: float = float("inf")


def establish_tracks(
    keypoints_per_image: Dict[int, np.ndarray],
    correspondences: Dict[Tuple[int, int], List],
    options: TrackEstablishmentOptions = None,
) -> Tuple[Dict[int, pycolmap.Point3D], Dict[int, Dict[int, int]]]:
    """
    Establish 3D point tracks from keypoints and correspondences.

    This implements the GLOMAP EstablishTracks algorithm:
    1. Union all matching observations using Union-Find
    2. Group observations by their root to form tracks
    3. Validate tracks for intra-image consistency
    4. Filter by minimum views
    5. Greedily select tracks by length

    Args:
        keypoints_per_image: Dict[image_id, np.ndarray(N, 2)]
        correspondences: Dict[(img_id1, img_id2), List[(pt_idx1, pt_idx2)]]
                        Must use same indexing as keypoints_per_image
        options: Track establishment options

    Returns:
        Tuple of:
            - points3D: Dictionary mapping point3D_id to pycolmap.Point3D objects
            - image_to_point3D: Dict[image_id, Dict[pt_idx, point3D_id]] mapping 2D to 3D
    """
    if options is None:
        options = TrackEstablishmentOptions()

    # Step 1: Union all matching observations
    uf = UnionFind()

    for (image_id1, image_id2), matches in correspondences.items():
        for match in matches:
            pt_idx1, pt_idx2 = match[0], match[1]
            obs1 = (image_id1, pt_idx1)
            obs2 = (image_id2, pt_idx2)

            # Consistent ordering for union (smaller first)
            if obs2 < obs1:
                uf.union(obs1, obs2)
            else:
                uf.union(obs2, obs1)

    # Step 2: Group observations by their root
    track_map = defaultdict(list)
    for obs in uf.parent.keys():
        root = uf.find(obs)
        track_map[root].append(obs)

    logger.info(f"Established {len(track_map)} tracks from {len(uf.parent)} observations")

    # Step 3: Validate tracks and check consistency
    candidate_points3D = {}
    track_lengths = []
    discarded_counter = 0
    next_point3D_id = 0

    sq_threshold = options.track_intra_image_consistency_threshold**2

    for track_id, observations in track_map.items():
        image_id_set = defaultdict(list)
        point3D = pycolmap.Point3D()
        is_consistent = True

        for image_id, feature_id in observations:
            # Skip if image not in keypoints or feature_id out of range
            if image_id not in keypoints_per_image:
                continue
            if feature_id >= len(keypoints_per_image[image_id]):
                continue

            xy = keypoints_per_image[image_id][feature_id]

            if image_id in image_id_set:
                # Check consistency with existing observations in same image
                for existing_xy in image_id_set[image_id]:
                    if np.sum((existing_xy - xy) ** 2) > sq_threshold:
                        is_consistent = False
                        break

                if not is_consistent:
                    discarded_counter += 1
                    break

                image_id_set[image_id].append(xy)
            else:
                image_id_set[image_id].append(xy)

            # Add to track
            point3D.track.add_element(image_id, feature_id)

        if not is_consistent:
            continue

        num_images = len(image_id_set)
        if num_images < options.track_min_num_views_per_track:
            continue

        point3D_id = next_point3D_id
        next_point3D_id += 1
        track_lengths.append((point3D.track.length(), point3D_id))
        candidate_points3D[point3D_id] = point3D

    logger.info(
        f"Kept {len(candidate_points3D)} tracks, discarded {discarded_counter} due to inconsistency"
    )

    # Step 4: Sort tracks by length (descending) and select greedily
    track_lengths.sort(reverse=True)

    tracks_per_image = defaultdict(int)
    images_with_keypoints = [k for k, v in keypoints_per_image.items() if len(v) > 0]
    images_left = len(images_with_keypoints)

    final_points3D = {}
    # Initialize mapping from (image_id, pt_idx) -> point3D_id
    image_to_point3D = {img_id: {} for img_id in keypoints_per_image.keys()}

    for track_length, point3D_id in track_lengths:
        point3D = candidate_points3D[point3D_id]

        # Check if any image in this track still needs more observations
        should_add = any(
            tracks_per_image[elem.image_id] <= options.track_required_tracks_per_view
            for elem in point3D.track.elements
        )

        if not should_add:
            continue

        # Update image counts
        for elem in point3D.track.elements:
            count = tracks_per_image[elem.image_id]
            if count == options.track_required_tracks_per_view:
                images_left -= 1
            tracks_per_image[elem.image_id] += 1

        # Add track and build 2D->3D mapping
        final_points3D[point3D_id] = point3D
        for elem in point3D.track.elements:
            image_to_point3D[elem.image_id][elem.point2D_idx] = point3D_id

        if images_left == 0:
            break

    logger.info(
        f"Before filtering: {len(candidate_points3D)}, after filtering: {len(final_points3D)}"
    )

    return final_points3D, image_to_point3D


def establish_tracks_from_tracks_dict(
    tracks_dict: Dict,
    num_images: int,
    options: TrackEstablishmentOptions = None,
    add_tracks: bool = True,
    add_virtual_points: bool = False,
    device: str = "cuda",
) -> Tuple[Dict[int, pycolmap.Point3D], Dict[int, np.ndarray]]:
    """
    Convenience function to establish tracks directly from tracks_dict.

    Args:
        tracks_dict: Dictionary containing tracks, scores, indexes, etc.
        num_images: Total number of images
        options: Track establishment options
        add_tracks: Whether to include real tracks
        add_virtual_points: Whether to include virtual points
        device: CUDA device for tensor operations

    Returns:
        Tuple of (points3D, keypoints_per_image)
    """
    # Step 1: Get keypoints and correspondences using existing function
    # Use 0-indexed for consistency with keypoints_per_image
    (
        keypoints_per_image,
        correspondences,
        images_points2d,
        images_points2d_virtual,
        images_points2d_virtual_isnegative,
        pts2d_idx_all,
        pts2d_idx_virtual_all,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
    ) = establish_keypoints_and_correspondences(
        tracks_dict=tracks_dict,
        N=num_images,
        add_tracks=add_tracks,
        add_virtual_points=add_virtual_points,
        device=device,
        use_1_indexed=False,  # Use 0-indexed to match keypoints_per_image
    )

    # Step 2: Establish tracks from correspondences
    points3D, image_to_point3D = establish_tracks(
        keypoints_per_image=keypoints_per_image,
        correspondences=correspondences,
        options=options,
    )

    return (
        points3D,
        keypoints_per_image,
        pts2d_idx_all,
        pts2d_idx_virtual_all,
        pts2d_idx_inv,
        pts2d_idx_virtual_inv,
        image_to_point3D,
        images_points2d_virtual_isnegative,
    )
