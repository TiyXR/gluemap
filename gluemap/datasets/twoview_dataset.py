import logging
import os

import faiss
import numpy as np
import torch

from gluemap.datasets.base_dataset import DemoBaseDataset
from gluemap.datasets.utils import get_image_list

logger = logging.getLogger(__name__)


class BaseTwoViewDataset(DemoBaseDataset):
    def __init__(self, args):
        super().__init__(args, patch_size=16)

        # Load all the images, and generate SALAD descriptors for matching
        img_list_full, img_list_used = self._get_image_list(args)

        # Now, use all the images in the folder
        self.images_list = [
            x.replace(args.images_path, "").strip("/") for x in img_list_used
        ]

        self.images_list_full = [
            x.replace(args.images_path, "").strip("/") for x in img_list_full
        ]

        self.images_path = [
            args.images_path for _ in range(len(self.images_list))
        ]

        # if salad descriptors are available, use them to construct pairs, else, use exhaustive matching
        self._construct_pairs(args, img_list_full)

        # get the image size of all the images and preload images if necessary
        self._preload_images(args)

        # prepare camera model
        self.camera_model = args.camera_model

        self.__post_init__()

    def __post_init__(self):
        neighbors = {i: set() for i in range(len(self.images_list))}
        for i in range(len(self.pairs)):
            neighbors[self.pairs[i][0]].add(self.pairs[i][1])
            neighbors[self.pairs[i][1]].add(self.pairs[i][0])

        self.neighbors = {k: sorted(list(v)) for k, v in neighbors.items()}
        pass

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
    # Return: full image list, used image list
    def _get_image_list(self, args):
        img_list = get_image_list(args.images_path)

        # Filter by images_list if provided (for experiment mode)
        if hasattr(args, "images_list") and args.images_list is not None:
            images_set = set(args.images_list)
            img_list_used = [
                x for x in img_list if os.path.basename(x) in images_set
            ]
        else:
            img_list_used = img_list

        return img_list, img_list_used

    def _construct_pairs(self, args, img_list):
        # TODO: extract the SALAD descriptors on the fly

        descriptors_path = os.path.join(
            args.curr_processed, "salad_descriptors.pt"
        )
        # args.num_neighbors = 100

        logger.info(f"Number of neighbors for retrieval: {args.num_neighbors}")

        if (
            os.path.exists(descriptors_path)
            and len(self.images_list) > args.num_neighbors
        ):
            descriptors_db = torch.load(descriptors_path, weights_only=False)
            # find the corresponding images and only keep the short list
            image_indexes = [
                img_list.index(os.path.join(args.images_path, img))
                for img in self.images_list
            ]
            descriptors_db = descriptors_db[image_indexes]

            embed_size = descriptors_db.shape[1]
            faiss_index = faiss.IndexFlatL2(embed_size)
            faiss_index.add(descriptors_db)  # add vectors to the index

            num_neighbors = min(args.num_neighbors, len(self.images_list) - 1)
            distance_extend, predictions_extend = faiss_index.search(
                descriptors_db, num_neighbors + 1
            )
        else:
            # consider the case for exhaustive matching
            predictions_extend = np.tile(
                np.arange(0, len(self.images_list)).reshape(1, -1),
                (len(self.images_list), 1),
            )

        # Construct the pairs
        chosen_indexes = np.arange(len(self.images_list))
        pairs = np.stack(
            [
                np.tile(
                    chosen_indexes[..., None], (1, predictions_extend.shape[1])
                ),
                predictions_extend[chosen_indexes],
            ],
            axis=-1,
        ).reshape(-1, 2)

        # collect back and forth
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        pairs = pairs[
            pairs[:, 0] < pairs[:, 1]
        ]  # only keep pairs where the first index is less than the second
        pairs = np.unique(pairs, axis=0)  # remove duplicates

        self.pairs = pairs
