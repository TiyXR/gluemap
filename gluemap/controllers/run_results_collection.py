import numpy as np
import torch
import networkx as nx


# from e2esfm.pipeline.tracks_util import *
from gluemap.datasets.star_dataset import BaseStarDataset
from gluemap.utils.misc import get_tracks_dict_indexes


def construct_covisibility_graph(pairs, scores, N, threshold=0.8):
    # Collect the valid edges based on the scores
    # valid_edges = pairs[scores > threshold]
    scores = scores.numpy()
    valid_edges = pairs[(scores > threshold)]

    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    G.add_edges_from(valid_edges)

    # Connect the unconnected components
    components = list(nx.connected_components(G))
    if len(components) > 1:
        while threshold > 0.0:
            # Collect the cluster index for each node
            cluster_index = np.zeros(N, dtype=int)
            for idx, component in enumerate(components):
                cluster_index[list(component)] = idx

            index = cluster_index[pairs[:, 0]] != cluster_index[pairs[:, 1]]

            threshold -= 0.1
            # Take the edges across the components
            G = nx.Graph()
            G.add_nodes_from(np.arange(N))
            G.add_edges_from(
                np.concatenate(
                    [valid_edges, pairs[index * (scores > threshold)]], axis=0
                )
            )
            components = list(nx.connected_components(G))

            print(f"Reducing threshold to {threshold:.2f} to connect components")
            valid_edges = np.concatenate(
                [valid_edges, pairs[index * (scores > threshold)]], axis=0
            )

            if len(components) == 1:
                break

    return valid_edges


def generate_dataset_from_outputs(
    dataset_ori, global_outputs, args, device="cuda", dtype=torch.bfloat16
):
    """
    Generate a dataset from the global outputs.
    """
    dataset = BaseStarDataset(args)

    scores = global_outputs["scores"].clone()
    sequential_edges = getattr(dataset_ori, "sequential_edges", [])

    # extract the graph structure from the global outputs
    valid_edges = construct_covisibility_graph(
        global_outputs["pairs"],
        scores,
        len(dataset_ori.images_list),
        threshold=args.valid_dg_threshold,
    )

    # Build edge score lookup from all pairs
    pairs_np = global_outputs["pairs"].numpy() if torch.is_tensor(global_outputs["pairs"]) else global_outputs["pairs"]
    scores_np = scores.numpy() if torch.is_tensor(scores) else scores
    edge_scores = {}
    for k in range(len(pairs_np)):
        i, j = int(pairs_np[k, 0]), int(pairs_np[k, 1])
        key = (min(i, j), max(i, j))
        edge_scores[key] = float(scores_np[k])

    # Initialize the dataset with the global outputs
    print("Initializing dataset with valid edges:", valid_edges.shape)
    dataset.valid_edges = valid_edges
    dataset.edge_scores = edge_scores
    dataset.N = len(dataset_ori.images_list)
    dataset.images_list = dataset_ori.images_list
    dataset.images_path = dataset_ori.images_path
    dataset.query_points = [None] * len(
        valid_edges
    )  # Initialize query points for each star structure
    dataset.images_shape_ori = dataset_ori.images_shape_ori
    dataset.force_square = dataset_ori.force_square
    dataset.sequential_edges = sequential_edges

    if hasattr(dataset_ori, "has_preloaded"):
        dataset.images_ori = dataset_ori.images_ori

    # Post initialization to set up the star structure
    dataset.__post_init__()

    print("Dataset initialized done.")

    return dataset


# def generate_dataset_from_star_outputs(
#     dataset_ori,
#     predictions_dict,
#     args,
#     device="cuda",
#     dtype=torch.bfloat16,
#     global_rotations=None,
#     global_centers=None,
#     global_intrinsics=None,
# ):
#     """
#     Generate a dataset from the global outputs.
#     """
#     dataset = BaseStarDataset(args)

#     # # # extract the graph structure from the global outputs
#     # valid_edges = construct_covisibility_graph(global_outputs["pairs"], global_outputs["scores"], len(dataset_ori.images_list), threshold=args.valid_dg_threshold)

#     # get the valid_edges
#     valid_edges = []
#     indexes = get_tracks_dict_indexes(predictions_dict)
#     for idx in indexes:
#         if len(predictions_dict["indexes"][idx]) == 1:
#             continue
#         valid_edges_curr_list = (
#             torch.where(predictions_dict["pose_scores"][idx][0] > 0)[0]
#         ).tolist()
#         valid_edges_curr_pair = [
#             (predictions_dict["indexes"][idx][0], predictions_dict["indexes"][idx][i])
#             for i in valid_edges_curr_list
#         ]
#         valid_edges.extend(valid_edges_curr_pair)

#     valid_edges = np.array(valid_edges)

#     # Initialize the dataset with the global outputs
#     dataset.valid_edges = valid_edges
#     dataset.N = len(dataset_ori.images_list)
#     dataset.images_list = dataset_ori.images_list
#     dataset.images_path = dataset_ori.images_path
#     dataset.query_points = [None] * len(
#         valid_edges
#     )  # Initialize query points for each star structure
#     dataset.images_shape_ori = dataset_ori.images_shape_ori

#     if hasattr(dataset_ori, "has_preloaded"):
#         dataset.images_ori = dataset_ori.images_ori

#     dataset.global_rotations = global_rotations
#     dataset.global_centers = global_centers
#     dataset.global_intrinsics = global_intrinsics
#     dataset.intrinsics_mapping = dataset_ori.intrinsics_mapping

#     # Post initialization to set up the star structure
#     dataset.__post_init__()

#     return dataset
