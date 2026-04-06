# from e2esfm.pipeline.tracks_util import track_snapping
# from e2esfm.models.utils import *
import logging
from pathlib import Path
import pycolmap
from tqdm import tqdm
import torch
import numpy as np
import faiss

import pyceres

from gluemap.utils.misc import get_tracks_dict_indexes

logger = logging.getLogger(__name__)


def track_snapping(tracks_dict, query_points_full, merge_thres=10.0, tolerance=3.0):
    tracks_dict["scores"] = {}
    for idx in get_tracks_dict_indexes(tracks_dict):
        tracks_dict["scores"][idx] = torch.where(
            tracks_dict["vis"][idx] > 0.05, tracks_dict["vis"][idx], 0.0
        )

    N = len(query_points_full)
    merge_thres_dict = {}
    if isinstance(merge_thres, float):
        for i in range(N):
            merge_thres_dict[i] = merge_thres
    elif not isinstance(merge_thres, dict):
        raise ValueError("Merge threshold should be a float or a dictionary")
    else:
        merge_thres_dict = merge_thres

    indexes_all = []
    for idx in range(N):
        index = faiss.IndexFlatL2(2)  # build the index
        index.add(query_points_full[idx][0])  # add vectors to the index
        indexes_all.append(index)

    counter = 0
    # merge_thres = merge_thres ** 2 # since we are using L2 distance
    for key in merge_thres_dict.keys():
        merge_thres_dict[key] = (
            merge_thres_dict[key] ** 2
        )  # since we are using L2 distance

    indexes = get_tracks_dict_indexes(tracks_dict)
    counter_valid = 0
    tracks_dict["is_close"] = {}
    logger.info("Indexing and snapping points...")
    for idx in tqdm(indexes):
        is_close_all = []
        for i, idx_inner in enumerate(tracks_dict["indexes"][idx]):
            D, I = indexes_all[idx_inner].search(tracks_dict["tracks"][idx][0, i], 1)
            if query_points_full[idx_inner].shape == (1, 0, 2):
                is_close = np.zeros(
                    (tracks_dict["tracks"][idx][0, i].shape[0],), dtype=bool
                )
            else:
                is_close = (D < merge_thres_dict[idx_inner]).reshape(
                    -1,
                )
                counter += (
                    is_close * (tracks_dict["scores"][idx][0, i] > 0).cpu().numpy()
                ).sum()

            tracks_dict["tracks"][idx][0, i, is_close] = query_points_full[idx_inner][
                0, I[is_close, 0]
            ]

            counter_valid += (tracks_dict["scores"][idx][0, i] > 0).sum()

            # Set those not close to be weaker
            tracks_dict["scores"][idx][0, i, ~is_close] /= 2
            # tracks_dict["scores"][idx][0, i, ~is_close] = 0

            is_close_all.append(is_close)

            # # rescale the tracks
            # tracks_dict["tracks"][idx][0, i] = rescale_tracks_single(tracks_dict["tracks"][idx][0, i], query_points_full[index_inner][1])

        tracks_dict["is_close"][idx] = torch.from_numpy(
            np.stack(is_close_all, axis=0)
        ).unsqueeze(0)

    logger.info(f"Total points snapped {counter} / {counter_valid}")


def refine_tracks_database(
    database_dir, predictions_dict, images_shape_ori, images_list, snapping_thres
):

    # Collect the query points from the database
    database = pycolmap.Database.open(database_dir)
    images = database.read_all_images()

    image_name_to_idx_db = {images[idx].name: idx for idx in range(len(images))}
    # Convert the name to the standard path
    image_name_to_standard_path = [str(Path(x)) for x in images_list]

    query_points_full = []
    for idx in range(len(image_name_to_standard_path)):
        image_name = image_name_to_standard_path[idx]
        # idx_ori_to_db[idx] = image_name_to_idx_db[image_name]
        image_id = images[image_name_to_idx_db[image_name]].image_id
        query_keypoints = database.read_keypoints(image_id)
        query_points_full.append(
            torch.from_numpy(query_keypoints)[:, :2].float().unsqueeze(0)
        )

    logger.info("Finish loading keypoints from the database")
    # print([query_points_full[i].shape for i in range(len(query_points_full))])

    snapping_thres_dict = {}
    for image_idx in range(len(images_shape_ori)):
        scaling_factor = (
            images_shape_ori[image_idx][0] + images_shape_ori[image_idx][1]
        ) / 1024
        snapping_thres_dict[image_idx] = snapping_thres * scaling_factor

    logger.info("Start track snapping")
    track_snapping(predictions_dict, query_points_full, merge_thres=snapping_thres_dict)
