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
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree

import pycolmap

from gluemap.math.union_find import UnionFind

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


class TrackEstablishment:
    def __init__(self, options: TrackEstablishmentOptions = None):
        self.options = options or TrackEstablishmentOptions()

    def run(
        self,
        tracks_dict: Dict,
        num_images: int,
        add_tracks: bool = True,
        add_virtual_points: bool = False,
        device: str = "cuda",
    ) -> Tuple[Dict[int, pycolmap.Point3D], Dict[int, np.ndarray]]:
        """
        Establish tracks directly from tracks_dict.

        Args:
            tracks_dict: Dictionary containing tracks, scores, indexes, etc.
            num_images: Total number of images
            add_tracks: Whether to include real tracks
            add_virtual_points: Whether to include virtual points
            device: CUDA device for tensor operations

        Returns:
            Tuple of (points3D, keypoints_per_image, ...)
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
        ) = self.establish_keypoints_and_correspondences(
            tracks_dict=tracks_dict,
            N=num_images,
            add_tracks=add_tracks,
            add_virtual_points=add_virtual_points,
            device=device,
            use_1_indexed=False,  # Use 0-indexed to match keypoints_per_image
        )

        # Step 2: Establish tracks from correspondences
        points3D, image_to_point3D = self._establish_tracks(
            keypoints_per_image=keypoints_per_image,
            correspondences=correspondences,
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

    def establish_keypoints_and_correspondences(
        self,
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

    def _establish_tracks(
        self,
        keypoints_per_image: Dict[int, np.ndarray],
        correspondences: Dict[Tuple[int, int], List],
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

        Returns:
            Tuple of:
                - points3D: Dictionary mapping point3D_id to pycolmap.Point3D objects
                - image_to_point3D: Dict[image_id, Dict[pt_idx, point3D_id]] mapping 2D to 3D
        """
        options = self.options

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
    Lightweight wrapper that creates a TrackEstablishment instance and calls run().
    """
    return TrackEstablishment(options=options).run(
        tracks_dict=tracks_dict,
        num_images=num_images,
        add_tracks=add_tracks,
        add_virtual_points=add_virtual_points,
        device=device,
    )
