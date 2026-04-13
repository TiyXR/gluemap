import logging
import os

import imagesize
import torch
from tqdm import tqdm

from gluemap.utils.load_fn import (
    load_and_preprocess_images,
    load_and_preprocess_images_1024,
    load_and_preprocess_images_inner,
)

logger = logging.getLogger(__name__)


class DemoBaseDataset:
    def __init__(self, args, patch_size=16):
        self.image_size = 512
        self.patch_size = patch_size
        if patch_size == 14:
            self.image_size = 518
        else:
            self.image_size = 512

        self.camera_model = (
            args.camera_model
            if hasattr(args, "camera_model")
            else "SIMPLE_PINHOLE"
        )

        self.sequential_edges = []  # list of (i, j) tuples, i < j, for immediate sequential neighbors

    def load_images(self, indexes, load_1024=False):
        image_paths = [
            os.path.join(self.images_path[i], self.images_list[i])
            for i in indexes
        ]

        # for the case with few images, load images beforehand is more efficient
        if hasattr(self, "has_preloaded"):
            images = torch.stack(
                [self.images[i] for i in indexes], dim=0
            ).float()
            images_ori = [self.images_ori[i] for i in indexes]
            images_change = [self.images_change[i] for i in indexes]

            if load_1024:
                images_1024 = torch.stack(
                    [self.images_1024[i] for i in indexes], dim=0
                ).float()
                images_change_1024 = [
                    self.images_change_1024[i] for i in indexes
                ]

        else:
            if not hasattr(self, "images_ori"):
                images, images_ori, images_change = load_and_preprocess_images(
                    image_paths,
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    force_square=self.force_square,
                )
            else:
                images, images_ori, images_change = (
                    load_and_preprocess_images_inner(
                        [self.images_ori[i] for i in indexes],
                        image_size=self.image_size,
                        patch_size=self.patch_size,
                        force_square=self.force_square,
                    )
                )

            if load_1024:
                images_1024, images_change_1024 = (
                    load_and_preprocess_images_1024(images_ori)
                )

        if not load_1024:
            return images, images_ori, images_change, image_paths
        else:
            return (
                images,
                images_ori,
                images_change,
                images_1024,
                images_change_1024,
                image_paths,
            )

    # -----------------------------------------------------------------
    # Utility functions
    # -----------------------------------------------------------------
    def _preload_images(self, args):
        # get the image size of all the images
        images_shape_ori = []
        for img_path in tqdm(self.images_list):
            img_size = imagesize.get(os.path.join(args.images_path, img_path))
            images_shape_ori.append(
                img_size[::-1]
            )  # (width, height) -> (height, width)

        self.images_shape_ori = images_shape_ori

        # Check whether all images have the same size
        # If not, force all images to be square
        unique_shapes = set(images_shape_ori)
        if (
            len(unique_shapes) == 1
            and images_shape_ori[0][0] < images_shape_ori[0][1]
        ):
            self.force_square = False
        else:
            self.force_square = True

        self.intrinsics_mapping = {}
        # If share_intrinsics, then for all images with the same shape, use the same intrinsics
        if hasattr(args, "share_intrinsics") and args.share_intrinsics:
            shape_to_intrinsics_idx = {}
            intrinsics_idx = 0
            for i, shape in enumerate(self.images_shape_ori):
                if shape not in shape_to_intrinsics_idx:
                    shape_to_intrinsics_idx[shape] = intrinsics_idx
                    intrinsics_idx += 1
                self.intrinsics_mapping[i] = shape_to_intrinsics_idx[shape]
        # Otherwise, each image has its own intrinsics
        else:
            self.intrinsics_mapping = {
                i: i for i in range(len(self.images_list))
            }

        # if the number of images is fewer than 200 images, directly load the images
        if len(self.images_list) < 200:
            self.has_preloaded = True
            self.images, self.images_ori, self.images_change = (
                load_and_preprocess_images(
                    [
                        os.path.join(self.images_path[i], self.images_list[i])
                        for i in range(len(self.images_list))
                    ],
                    image_size=self.image_size,
                    patch_size=self.patch_size,
                    force_square=self.force_square,
                )
            )
            self.images_1024, self.images_change_1024 = (
                load_and_preprocess_images_1024(self.images_ori)
            )
        else:
            logger.warning("images are not preloaded!")

        logger.info("preload image done")
