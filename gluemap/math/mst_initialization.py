import numpy as np
import torch
import networkx as nx

from gluemap.utils.misc import get_tracks_dict_indexes

MIN_TRI_ANGLE = 1  # degrees, minimum median triangulation angle for reliable scale estimation


def initialize_mst_structures(predictions_dict, global_rotations):
    N = max(global_rotations.keys()) + 1
    indexes = get_tracks_dict_indexes(predictions_dict)

    rel_poses = {}
    scales = {}  # (i,j): s_j / s_i
    node_idx_to_star_idx = {}
    predictions_dict["median_tri_angle"] = {}

    for star_idx, idx in enumerate(indexes):
        node_idx_to_star_idx[predictions_dict["indexes"][idx][0]] = star_idx

        # Compute max of median triangulation angle across edges
        points3d = predictions_dict["points3d_virtual"][idx][0]  # (K, 3)
        extr = predictions_dict["extrinsics"][idx]  # (1, N, 3, 4)
        N_views = extr.shape[1]

        ray_center = points3d / torch.clamp(
            points3d.norm(dim=-1, keepdim=True), min=1e-8
        )  # (K, 3)

        # Vectorized over all neighbor views
        R_all = extr[0, 1:, :3, :3]  # (M, 3, 3)
        t_all = extr[0, 1:, :3, 3]   # (M, 3)
        c_all = -torch.einsum('mji,mj->mi', R_all, t_all)  # (M, 3)

        ray_all = points3d.unsqueeze(0) - c_all.unsqueeze(1)  # (M, K, 3)
        ray_all = ray_all / torch.clamp(
            ray_all.norm(dim=-1, keepdim=True), min=1e-8
        )
        cos_angles = torch.clamp(
            torch.einsum('kd,mkd->mk', ray_center, ray_all),
            -1.0 + 1e-6, 1.0 - 1e-6,
        )  # (M, K)
        angles = torch.acos(cos_angles)  # (M, K)
        median_angles = angles.median(dim=-1).values if angles.numel() > 0 else torch.tensor([])  # (M,)

        predictions_dict["median_tri_angle"][idx] = np.rad2deg(median_angles) if median_angles.numel() > 0 else np.array([])


    for idx in indexes:
        poses = predictions_dict["extrinsics"][idx]
        pose_scores = predictions_dict["pose_scores"][idx]
        N_poses = poses.shape[1]
        idx_i = predictions_dict["indexes"][idx][0]
        for i in range(N_poses):
            if i == 0:
                continue
            idx_j = predictions_dict["indexes"][idx][i]
            status = not (idx_j, idx_i) in rel_poses
            star_idx_i = node_idx_to_star_idx[idx_i]
            star_idx_j = node_idx_to_star_idx[idx_j]
            score = pose_scores[0, i].item()
            if not status:
                # If either edge has a small triangulation angle, scale is unreliable
                angle_current = predictions_dict["median_tri_angle"][star_idx_i][i - 1]
                reverse_pos = rel_poses[(idx_j, idx_i)][2]
                angle_reverse = predictions_dict["median_tri_angle"][star_idx_j][reverse_pos - 1]
                if angle_current < MIN_TRI_ANGLE or angle_reverse < MIN_TRI_ANGLE:
                    scales[(star_idx_j, star_idx_i)] = 1.0
                    scales[(star_idx_i, star_idx_j)] = 1.0
                else:
                    scales[(star_idx_j, star_idx_i)] = (
                        poses[0, i, :3, 3:].norm().item()
                        / rel_poses[(idx_j, idx_i)][0][:3, 3:].norm().item()
                    )
                    scales[(star_idx_i, star_idx_j)] = (
                        1.0 / scales[(star_idx_j, star_idx_i)]
                    )
                score *= min(20, angle_current, angle_reverse) / 20  # downweight the edge if the triangulation angle is small

            rel_poses[(idx_i, idx_j)] = (
                poses[0, i].cpu(),
                idx,
                i,
                score
            )

    # Only consider the two side edges
    invalid_edges = []
    for (i, j), (_, _, _, score) in rel_poses.items():
        if (node_idx_to_star_idx[i], node_idx_to_star_idx[j]) not in scales:
            invalid_edges.append((i, j))

    for i, j in invalid_edges:
        del rel_poses[(i, j)]

    G = nx.Graph()
    G.add_nodes_from(np.arange(N))
    for (i, j), (pose, idx, i_pos, score) in rel_poses.items():
        G.add_edge(i, j, weight=score)
    nx.set_edge_attributes(
        G,
        {(i, j): score for (i, j), (_, _, _, score) in rel_poses.items()},
        "weight",
    )
    mst = nx.maximum_spanning_tree(G)

    global_centers = {}
    global_scales = {}
    visited = set()

    # Iterative DFS to avoid RecursionError on large graphs
    global_centers[0] = np.zeros((3,), dtype=np.float64)
    global_scales[node_idx_to_star_idx[0]] = 1.0
    visited.add(0)
    stack = [(0, iter(mst.neighbors(0)))]

    while stack:
        node, neighbors_iter = stack[-1]
        try:
            neighbor = next(neighbors_iter)
        except StopIteration:
            stack.pop()
            continue

        if neighbor in visited:
            continue

        visited.add(neighbor)
        idx_node = node_idx_to_star_idx[neighbor]
        idx_parent = node_idx_to_star_idx[node]
        pose, idx, i_pos, _ = rel_poses[(neighbor, node)]

        if idx_node in global_scales:
            global_scales[idx_parent] = (
                global_scales[idx_node] * scales[(idx_node, idx_parent)]
            )
        else:
            global_scales[idx_node] = (
                global_scales[idx_parent] * scales[(idx_parent, idx_node)]
            )

        # s_i * (c_j - c_i) = -R_j^T * t_ij
        # ==> c_i = c_j + R_j^T * t_ij / s_i
        global_centers[neighbor] = (
            global_centers[node]
            + (global_rotations[node].T @ pose[:3, 3:].numpy()).flatten()
            / global_scales[idx_node]
        )

        stack.append((neighbor, iter(mst.neighbors(neighbor))))

    return global_centers, global_scales
