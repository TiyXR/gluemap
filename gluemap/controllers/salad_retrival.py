import torch
from tqdm import tqdm

from gluemap.utils.load_fn import (
    load_and_preprocess_images,
)

from gluemap.datasets.utils import get_image_list


class SALADRetrieval:
    def __init__(self, model, args, device="cuda", dtype=torch.bfloat16):
        self.model = model
        self.args = args
        self.device = device
        self.dtype = dtype

        self.batch_size = args.retrieval_batch_size

        self.images_list = get_image_list(args.images_path)

    @torch.no_grad()
    def main(self):
        descriptors = []
        N = len(self.images_list)
        for i in tqdm(range(0, N, self.batch_size)):
            num_img = min(N - i, self.batch_size)
            images, _, _ = load_and_preprocess_images(
                self.images_list[i : i + num_img],
                image_size=322,  # use the fixed 322 size for SALAD retrieval
                patch_size=14,
                force_square=True,
            )
            output = self.model(images.to(self.device)).cpu()

            descriptors.append(output)

        descriptors = torch.cat(descriptors)
        return descriptors
