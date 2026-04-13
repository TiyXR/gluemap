import logging
import os

import numpy as np
import torch

from gluemap.datasets.base_dataset import DemoBaseDataset

logger = logging.getLogger(__name__)

from gluemap.datasets.utils import (
    establish_neighbors_sequential,
    get_image_list,
    retrieve_global_neighbors,
)


class MultiSequencePairs(DemoBaseDataset):
    def __init__(self, args, datasets):
        self.images_path_root = args.images_path
        self.datasets = datasets

        super().__init__(args, patch_size=16)

        # Load all the images, and generate SALAD descriptors for matching
        img_list = []
        for dataset in self.datasets:
            img_list.extend(
                get_image_list(os.path.join(args.images_path, dataset))
            )

        # Now, use all the images in the folder
        self.images_list = [
            x.replace(self.images_path_root, "").strip("/") for x in img_list
        ]

        self.images_path = [
            self.images_path_root for _ in range(len(self.images_list))
        ]

        self._construct_pairs(args, img_list)

        # get the image size of all the images and preload images if necessary
        self._preload_images(args)

        # prepare camera model
        self.camera_model = args.camera_model

    def __len__(self):
        return self.pairs.shape[0]

    def __getitem__(self, idx):
        indexes = self.pairs[idx]
        images, images_ori, images_change, image_paths = self.load_images(
            indexes, load_1024=False
        )

        return {
            "images": images,
            "images_change": np.array(images_change),
            "images_shape_ori": np.array(
                [images_ori[i].shape[-2:] for i in range(len(images_ori))]
            ),
            "image_paths": image_paths,
            "pair_indexes": idx,
            "indexes": indexes.astype(np.int64),
        }

    # -----------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------
    # Override the function to construct pairs
    def _construct_pairs(self, args, img_list):
        # Note that, here, it is a special case for LaMAR dataset
        datasets = self.datasets

        image_count = 0
        neighbors_sequential = []
        descriptors_all = []
        self.sequential_edges = []
        for dataset in datasets:
            logger.info(f"Constructing pairs for dataset: {dataset}")

            # For each subfolder, load the SALAD descriptors and construct sequential pairs
            img_list_dataset = [
                x
                for x in img_list
                if x.startswith(os.path.join(self.images_path_root, dataset))
            ]
            neighbors, descriptors_db = (
                self._retrieve_descriptors_and_neighbors(
                    args, dataset, img_list_dataset
                )
            )
            neighbors = torch.tensor(neighbors, dtype=torch.int64)
            neighbors = neighbors + image_count
            # if the number of neighbor is too few, pad it with the last value
            # it is fine since only unique value will remain
            if neighbors.shape[1] < args.num_neighbors_sequential + 1:
                diff = args.num_neighbors_sequential + 1 - neighbors.shape[1]
                neighbors = torch.cat(
                    [neighbors, neighbors[:, -1:].repeat(1, diff)], dim=1
                )

            # All sequential neighbors within this subfolder
            for k in range(neighbors.shape[0]):
                center = int(neighbors[k, 0])
                for m in range(1, neighbors.shape[1]):
                    nbr = int(neighbors[k, m])
                    self.sequential_edges.append(
                        (min(center, nbr), max(center, nbr))
                    )

            image_count += len(img_list_dataset)
            neighbors_sequential.append(neighbors)
            descriptors_all.append(descriptors_db)

        # Deduplicate sequential edges
        self.sequential_edges = list(set(self.sequential_edges))

        # Establish global neighbors based on the collected descriptors and sequential neighbors
        self.pairs = retrieve_global_neighbors(
            args, neighbors_sequential, descriptors_all
        )

    def _retrieve_descriptors_and_neighbors(self, args, dataset, img_list):

        descriptors_path = os.path.join(
            args.curr_processed, dataset, "salad_descriptors.pt"
        )

        # TODO: add subsampling logic here if necessary
        if os.path.exists(descriptors_path):
            descriptors_db = torch.load(descriptors_path, weights_only=False)
        else:
            descriptors_db = None

        neighbors = establish_neighbors_sequential(
            img_list, num_neighbors=args.num_neighbors_sequential
        )  # N, num_neighbors + 1

        return neighbors, descriptors_db
