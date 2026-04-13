import glob
import logging
import os

import faiss
import numpy as np
import torch

logger = logging.getLogger(__name__)


def get_image_list(images_path):
    img_list = sorted(glob.glob(images_path + "/**", recursive=True))
    # keep only the images
    img_list = [
        x
        for x in img_list
        if os.path.isfile(x)
        and x.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff"))
    ]
    return img_list


def establish_neighbors_sequential(
    image_names, num_neighbors=30, return_raw=False
):
    N = len(image_names)
    num_neighbors = min(num_neighbors, N - 1)

    neighbors_all = []
    for i in range(N):
        neighbors = []
        left = max(0, i - num_neighbors // 2)
        right = min(N, i + num_neighbors // 2 + 1)

        for j in range(left, right):
            if j != i:
                neighbors.append(j)

        # If not enough neighbors, pad from the other side
        while len(neighbors) < num_neighbors:
            if left > 0:
                left -= 1
                neighbors.insert(0, left)
            elif right < N:
                neighbors.append(right)
                right += 1
            else:
                break

        neighbors_all.append(neighbors)

    if return_raw:
        return neighbors_all

    neighbors_ori = torch.Tensor(neighbors_all).long()
    neighbors = torch.cat(
        [torch.arange(N).unsqueeze(1).to(neighbors_ori.device), neighbors_ori],
        dim=1,
    )

    return neighbors


def retrieve_global_neighbors(args, neighbors_sequential, descriptors_all):
    logger.info("Establishing global neighbors...")
    neighbors_local = [
        x[i].cpu().numpy()
        for x in neighbors_sequential
        for i in range(x.shape[0])
    ]
    if args.num_neighbors > 0:
        descriptors_db = torch.cat(descriptors_all, dim=0)
        embed_size = descriptors_db.shape[1]
        faiss_index = faiss.IndexFlatL2(embed_size)
        faiss_index.add(descriptors_db)  # add vectors to the index

        chosen_indexes = np.arange(descriptors_db.shape[0])
        sampled_neighbors = [
            set(neighbors_local[i].tolist())
            for i in range(len(neighbors_local))
        ]
        distance_extend, predictions_extend = faiss_index.search(
            descriptors_db[chosen_indexes],
            args.num_neighbors + args.num_neighbors_sequential + 1,
        )

        # exclude itself and invalid FAISS results (-1 when k > index size)
        predictions_extend = predictions_extend[:, 1:]

        # exclude invalid FAISS results (-1) and ones already in neighbors_local
        chosen_indexes_item = [
            [
                j
                for j, x in enumerate(predictions_extend[i].tolist())
                if x >= 0 and x not in sampled_neighbors[i]
            ][: args.num_neighbors]
            for i in range(len(predictions_extend))
        ]

        # Build retrieval pairs directly (rows may have different lengths after filtering)
        pairs_retrieval = (
            np.array(
                [
                    (i, predictions_extend[i][j])
                    for i in range(len(predictions_extend))
                    for j in chosen_indexes_item[i]
                ],
                dtype=int,
            ).reshape(-1, 2)
            if any(len(c) > 0 for c in chosen_indexes_item)
            else np.zeros((0, 2), dtype=int)
        )
    else:
        pairs_retrieval = np.zeros((0, 2), dtype=int)

    pairs_local = np.concatenate(
        [
            np.stack(
                [
                    np.ones(neighbors_local[i].shape, dtype=int) * i,
                    neighbors_local[i],
                ],
                axis=-1,
            )
            for i in range(len(neighbors_local))
        ],
        axis=0,
    )

    pairs = np.array(
        sorted(
            list(
                set(tuple(x) for x in pairs_retrieval).union(
                    tuple(x) for x in pairs_local
                )
            )
        )
    )

    # collect back and forth
    pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
    pairs = pairs[
        (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0) & (pairs[:, 0] < pairs[:, 1])
    ]  # only keep valid pairs where the first index is less than the second
    pairs = np.unique(pairs, axis=0)  # remove duplicates
    return pairs
