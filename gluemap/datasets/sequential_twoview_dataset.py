import torch
import os
import numpy as np
import faiss

from gluemap.datasets.twoview_dataset import BaseTwoViewDataset
from gluemap.datasets.utils import (
    establish_neighbors_sequential,
    retrieve_global_neighbors,
)


class SequentialTwoViewDataset(BaseTwoViewDataset):
    # -----------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------
    def _get_image_list(self, args):
        img_list_full, img_list_used = super()._get_image_list(args)
        if args.sample_frequency > 1:
            img_list_used = img_list_used[:: args.sample_frequency]
            
        return img_list_full, img_list_used
    
    def _construct_pairs(self, args, img_list):
        descriptors_path = os.path.join(args.curr_processed, "salad_descriptors.pt")

        if os.path.exists(descriptors_path):
            descriptors_db = torch.load(descriptors_path, weights_only=False)
            # find the corresponding images and only keep the short list
            image_indexes = [
                img_list.index(os.path.join(args.images_path, img))
                for img in self.images_list
            ]
            descriptors_db = descriptors_db[image_indexes]

        else:
            raise FileNotFoundError(
                f"SALAD descriptors not found at {descriptors_path}. "
                "Please run SALAD descriptor extraction first."
            )

        neighbors = establish_neighbors_sequential(
            self.images_list, num_neighbors=args.num_neighbors_sequential
        )  # N, num_neighbors + 1

        # retrieve global neighbors based on SALAD descriptors
        self.pairs = retrieve_global_neighbors(args, [neighbors], [descriptors_db])

        # All sequential neighbors within the window
        seq_set = set()
        for k in range(neighbors.shape[0]):
            center = int(neighbors[k, 0])
            for m in range(1, neighbors.shape[1]):
                nbr = int(neighbors[k, m])
                seq_set.add((min(center, nbr), max(center, nbr)))
        self.sequential_edges = list(seq_set)