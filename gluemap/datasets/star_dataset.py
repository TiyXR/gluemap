import logging

import networkx as nx
import numpy as np
from lightglue import ALIKED
from tqdm import tqdm

from gluemap.datasets.base_dataset import DemoBaseDataset
from gluemap.estimators.feature_extraction import (
    get_query_points_from_extractors,
)
from gluemap.utils.load_fn import calculate_image_shapes

logger = logging.getLogger(__name__)


class BaseStarDataset(DemoBaseDataset):
    def __init__(self, args):
        self.image_size = 518
        self.patch_size = 14

        self.num_tracks = (
            args.num_track_per_img
            if hasattr(args, "num_track_per_img")
            else 1024
        )

        # Build the keypoint extractor once and reuse it across __getitem__ calls.
        # Moved to the query image's device on first use.
        # detection_threshold=0.005 matches the det_thres default used in the
        # original implementation.
        self.query_extractors = [
            ALIKED(
                max_num_keypoints=self.num_tracks, detection_threshold=0.005
            ).eval()
        ]
        self._extractors_device = None

        self.query_points = None  # needs to be initialized
        self.valid_edges = None  # needs to be initialized
        self.edge_scores = None  # per-edge scores for score-based pruning
        self.sequential_edges = None  # immediate sequential neighbor edges

        self.N = None  # Number of nodes in the star structure, needs to be initialized
        self.images_list = None
        self.images_path = None

        self.global_rotations = None
        self.global_centers = None
        self.global_intrinsics = None
        self.intrinsics_mapping = None

    def __post_init__(self):
        # Initialize any additional attributes or configurations after the object is created
        # Collect the indexes from the pairs

        # Initialize the star structure from the valid edges
        valid_edges = np.array(self.valid_edges, dtype=np.int64)
        valid_edges = np.concatenate(
            [valid_edges, valid_edges[:, ::-1]], axis=0
        )
        valid_edges = np.unique(valid_edges, axis=0)

        # Collect the neighbors for each node
        neighbors_all = {i: set() for i in range(self.N)}
        for i in range(len(valid_edges)):
            start, end = valid_edges[i]
            neighbors_all[start].add(end)
            neighbors_all[end].add(start)

        # Prune neighbors to top-K by score, then ensure connectivity
        MAX_NEIGHBORS = getattr(self, "max_neighbors", 25)
        MAX_SEQ = min(20, MAX_NEIGHBORS - 5)
        MIN_NONSEQ = 5

        # Build sequential edge lookup set
        seq_edge_set = (
            set(self.sequential_edges) if self.sequential_edges else set()
        )
        has_sequential = len(seq_edge_set) > 0

        if self.edge_scores is not None:
            for i in range(self.N):
                if len(neighbors_all[i]) <= MAX_NEIGHBORS:
                    continue

                if has_sequential:
                    # Split neighbors into sequential and non-sequential
                    seq_nbrs = []
                    nonseq_nbrs = []
                    for j in neighbors_all[i]:
                        edge = (min(i, j), max(i, j))
                        if edge in seq_edge_set:
                            seq_nbrs.append(j)
                        else:
                            nonseq_nbrs.append(j)

                    # Sample sequential neighbors: keep first & last, uniform middle
                    seq_nbrs.sort()
                    if len(seq_nbrs) > MAX_SEQ:
                        first, last = seq_nbrs[0], seq_nbrs[-1]
                        middle = seq_nbrs[1:-1]
                        n_middle = MAX_SEQ - 2
                        indices = np.round(
                            np.linspace(0, len(middle) - 1, n_middle)
                        ).astype(int)
                        sampled_middle = [
                            middle[idx] for idx in np.unique(indices)
                        ]
                        seq_nbrs = [first] + sampled_middle + [last]

                    # Fill remaining slots with top non-sequential by score
                    remaining = max(MAX_NEIGHBORS - len(seq_nbrs), MIN_NONSEQ)
                    nonseq_scored = sorted(
                        [
                            (
                                j,
                                self.edge_scores.get(
                                    (min(i, j), max(i, j)), 0.0
                                ),
                            )
                            for j in nonseq_nbrs
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    selected_nonseq = [x[0] for x in nonseq_scored[:remaining]]
                    neighbors_all[i] = set(seq_nbrs + selected_nonseq)
                else:
                    # No sequential edges: top-K by score
                    scored = [
                        (j, self.edge_scores.get((min(i, j), max(i, j)), 0.0))
                        for j in neighbors_all[i]
                    ]
                    scored.sort(key=lambda x: x[1], reverse=True)
                    neighbors_all[i] = set(x[0] for x in scored[:MAX_NEIGHBORS])

            # Check connectivity; add back pruned edges if disconnected
            G = nx.Graph()
            G.add_nodes_from(range(self.N))
            for i in range(self.N):
                for j in neighbors_all[i]:
                    G.add_edge(i, j)

            if not nx.is_connected(G):
                # Collect pruned edges sorted by score descending
                original_set = set()
                for e in np.array(self.valid_edges, dtype=np.int64):
                    original_set.add(
                        (min(int(e[0]), int(e[1])), max(int(e[0]), int(e[1])))
                    )
                current_set = set()
                for i in range(self.N):
                    for j in neighbors_all[i]:
                        current_set.add((min(i, j), max(i, j)))
                pruned_edges = original_set - current_set
                pruned_with_scores = sorted(
                    [(e, self.edge_scores.get(e, 0.0)) for e in pruned_edges],
                    key=lambda x: x[1],
                    reverse=True,
                )

                for (u, v), score in pruned_with_scores:
                    if nx.is_connected(G):
                        break
                    # Only add the edge if it bridges two different components
                    if nx.node_connected_component(
                        G, u
                    ) != nx.node_connected_component(G, v):
                        G.add_edge(u, v)
                        neighbors_all[u].add(v)
                        neighbors_all[v].add(u)
                logger.info("Re-added edges to restore connectivity")
            else:
                logger.info(
                    f"Graph is already connected after top-{MAX_NEIGHBORS} pruning"
                )
        else:
            # Fallback: no scores available, random selection
            for i in range(self.N):
                if len(neighbors_all[i]) > MAX_NEIGHBORS:
                    neighbors = list(neighbors_all[i])
                    index = np.random.choice(
                        len(neighbors), size=MAX_NEIGHBORS, replace=False
                    )
                    neighbors_all[i] = set(np.array(neighbors)[index])

        # Filter sequential_edges to only keep edges that survived pruning
        if self.sequential_edges is not None:
            kept_edges = set()
            for i in range(self.N):
                for j in neighbors_all[i]:
                    kept_edges.add((min(i, j), max(i, j)))
            self.sequential_edges = [
                e for e in self.sequential_edges if e in kept_edges
            ]

        # Build stars from pruned neighbors
        stars = []
        for i in tqdm(range(self.N)):
            neighbors = neighbors_all[i]
            if len(neighbors) > 0:
                neighbors = np.array(list(neighbors), dtype=np.int64)
                stars.append(np.concatenate([[i], neighbors], axis=0))

        neighbors_lens = [len(x) for x in stars]
        # sort by the length in descending order
        sorted_idx = np.argsort(neighbors_lens)[::-1]
        stars = [stars[i] for i in sorted_idx]

        # Add a mapping between star index and image index
        self.image_index_to_star_index = {
            stars[i][0]: i for i in range(len(stars))
        }
        self.stars = stars

        # calculate the image_changes
        if self.force_square:
            new_shape_hw = (518, 518)
        else:
            # find the shape
            ori_shape = self.images_shape_ori[0]
            height, width = ori_shape

            if width > height:
                new_width = 518

                # Calculate height maintaining aspect ratio, divisible by 14
                new_height = round(height * (new_width / width))

            else:
                new_height = 518

                # Calculate width maintaining aspect ratio, divisible by 14
                new_width = round(width * (new_height / height))

            new_shape_hw = (new_height, new_width)

        self.images_change = calculate_image_shapes(
            self.images_shape_ori, new_shape_hw
        )

        valid_edges = valid_edges[
            valid_edges[:, 0] < valid_edges[:, 1]
        ]  # only keep valid_edges where the first index is less than the second
        valid_edges = np.unique(valid_edges, axis=0)  # remove duplicates
        self.pairs = valid_edges

    def __len__(self):
        return len(self.stars)

    def __getitem__(self, idx):
        indexes = self.stars[idx]
        (
            images,
            images_ori,
            images_change,
            images_1024,
            images_change_1024,
            image_paths,
        ) = self.load_images(indexes, load_1024=True)

        query_image = images_1024[:1]
        if self._extractors_device != query_image.device:
            self.query_extractors = [
                e.to(query_image.device) for e in self.query_extractors
            ]
            self._extractors_device = query_image.device
        query_points = get_query_points_from_extractors(
            query_image,
            self.query_extractors,
            max_query_num=self.num_tracks,
            strict_num=False,
        )[0]

        global_rotations = None
        global_centers = None
        global_intrinsics = None
        if self.global_rotations is not None:
            # subsample the rotations, centers, intrinsics
            global_rotations = [self.global_rotations[i] for i in indexes]
            global_centers = [self.global_centers[i] for i in indexes]
            global_intrinsics = [
                self.global_intrinsics[self.intrinsics_mapping[i]][0]
                for i in indexes
            ]

        # extract the query points for the star structure
        batch = {
            "images": images,
            "images_change": np.array(images_change),
            "images_shape_ori": np.array(
                [images_ori[i].shape[-2:] for i in range(len(images_ori))]
            ),
            "images_1024": images_1024,
            "images_change_1024": np.array(images_change_1024),
            "image_paths": image_paths,
            "star_indexes": idx,
            "indexes": indexes.astype(np.int64),
            "query_points": query_points,
        }

        if global_rotations is not None:
            batch["global_rotations"] = np.array(global_rotations)
            batch["global_centers"] = np.array(global_centers)
            batch["global_intrinsics"] = np.array(global_intrinsics)

        return batch
