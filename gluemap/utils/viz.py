import os
from collections import defaultdict
from typing import Dict, Tuple

import networkx as nx
import pycolmap


def draw_covisibility_graph(
    pair_count: Dict[Tuple[int, int], int],
    reconstruction: pycolmap.Reconstruction,
    save_path: str = "covisibility_graph.png",
    min_shared_tracks: int = 0,
) -> None:
    """
    Draw and save a covisibility graph from pair_count.

    Nodes are images (labeled by name), edges are weighted by the number of
    shared tracks between each pair.

    Args:
        pair_count: Dict[(image_id_i, image_id_j), count] of shared tracks.
        reconstruction: pycolmap.Reconstruction for image name lookup.
        save_path: Path to save the figure.
        min_shared_tracks: Only draw edges with count > min_shared_tracks.
    """
    import matplotlib.pyplot as plt

    G = nx.Graph()

    # Add nodes for all images in reconstruction
    id_to_label = {}
    for image_id, image in reconstruction.images.items():
        short_name = os.path.splitext(os.path.basename(image.name))[0]
        id_to_label[image_id] = short_name
        G.add_node(image_id, label=short_name)

    # Add weighted edges from pair_count
    for (id_i, id_j), count in pair_count.items():
        if count > min_shared_tracks and id_i in id_to_label and id_j in id_to_label:
            G.add_edge(id_i, id_j, weight=count)

    if len(G.edges) == 0:
        print("No edges in covisibility graph, skipping drawing.")
        return

    # Extract edge weights for visualization
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights)
    min_w = min(weights)
    # Normalize widths to [0.5, 4.0]
    if max_w > min_w:
        norm_widths = [0.5 + 3.5 * (w - min_w) / (max_w - min_w) for w in weights]
    else:
        norm_widths = [2.0] * len(weights)

    labels = nx.get_node_attributes(G, "label")

    # Scale figure size based on number of nodes
    n_nodes = len(G.nodes)
    fig_size = max(10, n_nodes * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, n_nodes**0.5))
    nx.draw_networkx_edges(G, pos, width=norm_widths, alpha=0.4, edge_color="gray", ax=ax)
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="steelblue", ax=ax)
    # Only show labels if graph is not too large
    if n_nodes <= 200:
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    ax.set_title(f"Covisibility Graph ({len(G.nodes)} images, {len(G.edges)} edges)")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Covisibility graph saved to {save_path}")
    

def draw_covisibility_graph_from_predictions(
    predictions_dict: Dict,
    save_path: str = "covisibility_graph_predictions.png",
    min_score: float = 0.0,
    images_list: list = None,
) -> None:
    """
    Draw and save a covisibility graph from predictions_dict using pose scores.

    Extracts image pairs and their pose scores from the star-based predictions_dict,
    builds a networkx graph, and saves a visualization.

    Args:
        predictions_dict: Star inference output containing "indexes" and "pose_scores".
        save_path: Path to save the figure.
        min_score: Only draw edges with score > min_score.
        images_list: Optional list of image filenames indexed by image ID for node labels.
    """
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    # Extract pairs and scores from star configurations
    pair_scores: Dict[Tuple[int, int], float] = {}
    for star_idx in range(len(predictions_dict["indexes"])):
        indexes = predictions_dict["indexes"][star_idx]
        scores = predictions_dict["pose_scores"][star_idx]
        center = indexes[0]
        for i in range(1, len(indexes)):
            neighbor = indexes[i]
            score = float(scores[0, i])
            pair = (min(center, neighbor), max(center, neighbor))
            # Keep max score if pair appears in multiple stars
            if pair not in pair_scores or score > pair_scores[pair]:
                pair_scores[pair] = score

    G = nx.Graph()

    # Add nodes
    all_image_ids = set()
    for star_idx in range(len(predictions_dict["indexes"])):
        all_image_ids.update(predictions_dict["indexes"][star_idx])

    id_to_label = {}
    for img_id in sorted(all_image_ids):
        if images_list is not None and img_id < len(images_list):
            short_name = os.path.splitext(os.path.basename(images_list[img_id]))[0]
        else:
            short_name = str(img_id)
        id_to_label[img_id] = short_name
        G.add_node(img_id, label=short_name)

    # Add weighted edges
    for (id_i, id_j), score in pair_scores.items():
        if score > min_score:
            G.add_edge(id_i, id_j, weight=score)

    if len(G.edges) == 0:
        print("No edges in covisibility graph, skipping drawing.")
        return

    # Edge weights for width and color
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights)
    min_w = min(weights)
    if max_w > min_w:
        norm_weights = [(w - min_w) / (max_w - min_w) for w in weights]
        norm_widths = [0.5 + 3.5 * nw for nw in norm_weights]
    else:
        norm_weights = [1.0] * len(weights)
        norm_widths = [2.0] * len(weights)

    # Map scores to colors (red=low, green=high)
    cmap = cm.RdYlGn
    edge_colors = [cmap(nw) for nw in norm_weights]

    labels = nx.get_node_attributes(G, "label")

    n_nodes = len(G.nodes)
    fig_size = max(10, n_nodes * 0.3)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    pos = nx.spring_layout(G, seed=42, k=2.0 / max(1, n_nodes**0.5))
    nx.draw_networkx_edges(
        G, pos, width=norm_widths, alpha=0.6, edge_color=edge_colors, ax=ax
    )
    nx.draw_networkx_nodes(G, pos, node_size=80, node_color="steelblue", ax=ax)
    if n_nodes <= 200:
        nx.draw_networkx_labels(G, pos, labels, font_size=6, ax=ax)

    # Add colorbar for score
    sm = plt.cm.ScalarMappable(
        cmap=cmap, norm=plt.Normalize(vmin=min_w, vmax=max_w)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Pose Score")

    ax.set_title(
        f"Covisibility Graph ({len(G.nodes)} images, {len(G.edges)} edges, "
        f"score range [{min_w:.3f}, {max_w:.3f}])"
    )
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Covisibility graph saved to {save_path}")
    
