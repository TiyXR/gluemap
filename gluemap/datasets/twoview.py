import argparse
import logging
import os
from typing import Any

import faiss
import numpy as np
import torch

from gluemap.datasets.base import DemoBaseDataset
from gluemap.datasets.utils import get_image_list

logger = logging.getLogger(__name__)


class BaseTwoViewDataset(DemoBaseDataset):
    """Two-view dataset that yields image pairs for pairwise inference.

    Discovers images under ``args.images_path``, optionally filters to a
    user-supplied list, builds candidate pairs from cached SALAD
    descriptors (or falls back to exhaustive matching), and exposes the
    pairs as PyTorch-style ``(images, ...)`` items.
    """

    def __init__(self, args: argparse.Namespace) -> None:
        """Discover images, build pairs, and preload tensors.

        Args:
            args: Parsed CLI/config namespace. Reads ``images_path``,
                ``camera_model``, ``num_neighbors``, ``curr_processed``,
                and optional ``images_list``.
        """
        super().__init__(args, patch_size=16)

        # Load all the images, and generate SALAD descriptors for matching
        img_list_full, img_list_used = self._get_image_list(args)

        # Now, use all the images in the folder
        self.images_list = [
            os.path.relpath(x, args.images_path) for x in img_list_used
        ]

        self.images_list_full = [
            os.path.relpath(x, args.images_path) for x in img_list_full
        ]

        self.images_path = [
            args.images_path for _ in range(len(self.images_list))
        ]

        # if salad descriptors are available, use them to construct pairs,
        # else, use exhaustive matching
        self._construct_pairs(args, img_list_full)

        # get the image size of all the images and preload images if necessary
        self._preload_images(args)

        # prepare camera model
        self.camera_model = args.camera_model

        self.__post_init__()

    def __post_init__(self) -> None:
        """Build the per-image neighbor lookup from ``self.pairs``."""
        neighbors = {i: set() for i in range(len(self.images_list))}
        for i in range(len(self.pairs)):
            neighbors[self.pairs[i][0]].add(self.pairs[i][1])
            neighbors[self.pairs[i][1]].add(self.pairs[i][0])

        self.neighbors = {k: sorted(list(v)) for k, v in neighbors.items()}
        pass

    def __len__(self) -> int:
        return self.pairs.shape[0]

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return the ``idx``-th image pair and its preprocessing metadata.

        Args:
            idx: Pair index in ``[0, len(self))``.

        Returns:
            Dict with keys ``images`` (stacked tensor), ``images_change``
            (per-image preprocessing transforms), ``images_shape_ori``
            (original H, W), ``image_paths`` (on-disk paths),
            ``pair_indexes`` (the input ``idx``), and ``indexes``
            (the two image indices as int64).
        """
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
    def _get_image_list(
        self, args: argparse.Namespace
    ) -> tuple[list[str], list[str]]:
        """Discover images and apply the optional ``images_list`` filter.

        Args:
            args: Parsed CLI/config namespace. Reads ``images_path`` and
                optional ``images_list`` (a sequence of basenames to keep).

        Returns:
            ``(img_list_full, img_list_used)`` — the full discovery
            order and the (possibly filtered) subset to operate on.
        """
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

    def _construct_pairs(
        self, args: argparse.Namespace, img_list: list[str]
    ) -> None:
        """Populate ``self.pairs`` with undirected unique pair indices.

        Uses a FAISS L2 search over cached SALAD descriptors at
        ``args.curr_processed/salad_descriptors.pt`` when available and
        the descriptor count is large enough. Otherwise falls back to
        exhaustive matching.

        Args:
            args: Parsed CLI/config namespace. Reads ``curr_processed``,
                ``num_neighbors``, and ``images_path``.
            img_list: Full list of image paths discovered during
                preprocessing — used to map current ``self.images_list``
                entries back to descriptor rows.
        """
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
