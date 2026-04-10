import logging
import numpy as np
import torch
import networkx as nx

from gluemap.datasets.star_dataset import BaseStarDataset

logger = logging.getLogger(__name__)


class StarCollector:
    """
    Collects two-view outputs and constructs a star dataset for multi-view inference.

    Builds a covisibility graph from pairwise scores and initializes a
    BaseStarDataset with valid edges and metadata from the original dataset.
    """

    def __init__(self, args):
        self.args = args

    def run(self, dataset_pair, global_outputs):
        """
        Generate a star dataset from two-view global outputs.
        """
        args = self.args
        dataset = BaseStarDataset(args)

        scores = global_outputs["scores"].clone()
        sequential_edges = getattr(dataset_pair, "sequential_edges", [])

        # extract the graph structure from the global outputs
        valid_edges = self._construct_covisibility_graph(
            global_outputs["pairs"],
            scores,
            len(dataset_pair.images_list),
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
        logger.info(f"Initializing dataset with valid edges: {valid_edges.shape}")
        dataset.valid_edges = valid_edges
        dataset.edge_scores = edge_scores
        dataset.N = len(dataset_pair.images_list)
        dataset.images_list = dataset_pair.images_list
        dataset.images_path = dataset_pair.images_path
        dataset.query_points = [None] * len(
            valid_edges
        )  # Initialize query points for each star structure
        dataset.images_shape_ori = dataset_pair.images_shape_ori
        dataset.force_square = dataset_pair.force_square
        dataset.sequential_edges = sequential_edges

        if hasattr(dataset_pair, "has_preloaded"):
            dataset.images_ori = dataset_pair.images_ori

        # Post initialization to set up the star structure
        dataset.__post_init__()

        logger.info("Dataset initialized done.")

        return dataset

    def _construct_covisibility_graph(self, pairs, scores, N, threshold=0.8):
        # Collect the valid edges based on the scores
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

                logger.info(f"Reducing threshold to {threshold:.2f} to connect components")
                valid_edges = np.concatenate(
                    [valid_edges, pairs[index * (scores > threshold)]], axis=0
                )

                if len(components) == 1:
                    break

        return valid_edges



def run_star_collection(dataset_pair, global_outputs, args):
    """Convenience wrapper — instantiates StarCollector and runs it."""
    return StarCollector(args).run(dataset_pair, global_outputs)
