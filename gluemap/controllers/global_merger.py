
import numpy as np
import torch
import networkx as nx

from gluemap.estimators.rotation_averaging import rotation_averaging, rotation_averaging_pycolmap
from gluemap.estimators.intrinsics_averaging import intrinsics_averaging
from gluemap.estimators.similarity_averaging import (
    similarity_averaging,
    similarity_averaging_with_depth,
)
from gluemap.estimators.bundle_adjustment import bundle_adjustment
from gluemap.controllers.bundle_adjustment import initialize_world_points
from gluemap.estimators.establish_tracks import (
    establish_tracks_from_tracks_dict,
    TrackEstablishmentOptions,
)

from gluemap.utils.misc import (
    get_tracks_dict_indexes,
    restore_identity,
)

import logging
logger = logging.getLogger(__name__)

MIN_TRI_ANGLE = 1 # degrees, minimum median triangulation angle for reliable scale estimation


# This class is purely optimization based, so we do not need to store the model
class GlobalGluer:
    def __init__(self, args):
        self.max_rot_error = 5  # degrees

        # self.valid_threshold = args.valid_threshold
        self.valid_threshold_pose = (
            args.valid_pose_threshold if hasattr(args, "valid_pose_threshold") else 0.05
        )

        # self.skip_optimization = False
        # self.global_rotations = None
        # self.global_centers = None
        # self.global_intrinsics = None

        self.thres_consistency = np.deg2rad(10.0)  # degrees
        self.angle_threshold = 5.0  # degrees
        self.boost_sequential = (
            True if hasattr(args, "is_sequential") and args.is_sequential else False
        )
        self.use_ceres_rotation_averaging = getattr(args, "use_ceres_rotation_averaging", True)

    def main(self, predictions_dict, intrinsics_mapping, camera_model, num_img):
        self.N = num_img
        # Refine the graph structure
        predictions_dict, valid_edges = self._refine_graph_structure(predictions_dict)

        # Estimate the intrinsics
        global_intrinsics = self._estimate_intrinsics(
            predictions_dict, intrinsics_mapping, camera_model
        )

        global_rotations, global_centers = self._global_structure_estimation(
            predictions_dict,
        )

        return (
            global_rotations,
            global_centers,
            global_intrinsics,
            valid_edges,
            predictions_dict,
        )

    # Suppress edges which are too weak, and connect the missing edges
    def _refine_graph_structure(self, predictions_dict):
        predictions_dict["scores"] = {}
        for idx in get_tracks_dict_indexes(predictions_dict):
            predictions_dict["scores"][idx] = torch.where(
                predictions_dict["vis"][idx] > 0.05, predictions_dict["vis"][idx], 0.0
            )
        # Perform two way check for filtering simple outliers
        self._filter_inconsistent_edges(predictions_dict)

        # First, we want to collect the valid edges
        valid_edges = self._collect_valid_edges(predictions_dict)

        # Then, connect the missing edges
        self._connect_missing(valid_edges, predictions_dict)

        # Prune invisible pairs
        self._prune_invisible_pairs(predictions_dict)

        return predictions_dict, valid_edges

    def _filter_inconsistent_edges(self, predictions_dict):
        indexes = get_tracks_dict_indexes(predictions_dict)

        rel_poses = {}
        # inconsistent_edges = set()
        counter = 0
        for idx in indexes:
            # pose_scores = predictions_dict["pose_scores"][idx][0]
            poses = predictions_dict["extrinsics"][idx]
            N = poses.shape[1]
            idx_i = predictions_dict["indexes"][idx][0]
            for i in range(N):
                if i == 0:
                    continue
                idx_j = predictions_dict["indexes"][idx][i]
                if (idx_j, idx_i) in rel_poses:
                    pose = rel_poses[(idx_j, idx_i)][0]
                    # Compare the two relative poses
                    R12 = pose[:3, :3]
                    R21 = poses[0, i, :3, :3]

                    error_r = torch.acos(
                        torch.clamp(
                            ((R12 @ R21).trace() - 1) / 2, -1.0 + 1e-6, 1.0 - 1e-6
                        )
                    )

                    # R21 * t12_normed = -t21_normed
                    t12_normed = pose[:3, 3:] / torch.linalg.norm(pose[:3, 3:])
                    t21_normed = poses[0, i, :3, 3:] / torch.linalg.norm(
                        poses[0, i, :3, 3:]
                    )
                    error_t = torch.acos(
                        torch.clamp(
                            -torch.sum(t12_normed * (R12 @ t21_normed)),
                            -1.0 + 1e-6,
                            1.0 - 1e-6,
                        )
                    )

                    if (
                        error_r > self.thres_consistency
                        or error_t > 3 * self.thres_consistency
                    ):
                        # Inconsistent, suppress both directions
                        predictions_dict["pose_scores"][idx][0, i] = 0.0
                        idx_1 = rel_poses[(idx_j, idx_i)][1]
                        j_pos = rel_poses[(idx_j, idx_i)][2]

                        predictions_dict["pose_scores"][idx_1][0, j_pos] = 0.0
                        logger.debug(
                            f"Filtered inconsistent edge between {idx_i} and {idx_j}, rotation error: {np.rad2deg(error_r)} degrees, translation error: {np.rad2deg(error_t)} degrees"
                        )
                        counter += 1
                else:
                    rel_poses[(idx_i, idx_j)] = (poses[0, i].cpu(), idx, i)

        logger.info(f"Total number of inconsistent edges filtered: {counter}")

    # TODO: debug this part
    def _collect_valid_edges(self, predictions_dict):
        # Here, the score already considers the n^2 visibility, so we can just use the pose scores
        valid_edges = set()
        indexes = get_tracks_dict_indexes(predictions_dict)
        for idx in indexes:
            valid_j = torch.where(
                predictions_dict["pose_scores"][idx][0] > self.valid_threshold_pose
            )[0]
            valid_edges.update(
                set(
                    [
                        (
                            predictions_dict["indexes"][idx][0],
                            predictions_dict["indexes"][idx][j],
                        )
                        for j in valid_j[1:]
                    ]
                )
            )

        return valid_edges

    def _connect_missing(self, valid_edges, predictions_dict):
        N = self.N

        # Establish tree with the valid edges
        G = nx.Graph()
        G.add_edges_from(list(valid_edges))

        # Find the connected components
        components = list(nx.connected_components(G))
        if (len(components) == 1) and (len(components[0]) == N):
            logger.info(f"Edge connectivity of the graph: {nx.edge_connectivity(G)}")
            return

        components = [list(x) for x in components]

        image_id_to_cluster_id = {}
        for i, component in enumerate(components):
            for image_id in component:
                image_id_to_cluster_id[image_id] = i

        curr_component_num = len(components)
        for i in range(N):
            if i not in image_id_to_cluster_id:
                image_id_to_cluster_id[i] = curr_component_num
                curr_component_num += 1

        # For edges across the components, we just set all scores to be 1e-2
        for idx in range(len(predictions_dict["indexes"])):
            idx1 = predictions_dict["indexes"][idx][0]
            for i, idx_inner in enumerate(predictions_dict["indexes"][idx]):
                if i == 0:
                    continue

                if image_id_to_cluster_id[idx1] != image_id_to_cluster_id[idx_inner]:
                    predictions_dict["pose_scores"][idx][0, i] += 1e-2
                    logger.debug(f"{idx1} {idx_inner} cross component")
                    valid_edges.add((idx1, idx_inner))

    def _global_structure_estimation(
        self, predictions_dict,
    ):
        # Double sequential edge weights (neighboring frames with index diff <= 10)
        if self.boost_sequential:
            self._boost_sequential_edges(predictions_dict, boost_factor=5.0)

        if self.use_ceres_rotation_averaging:
            # Original two-pass: RA → filter → RA → filter
            global_rotations = rotation_averaging(predictions_dict)
            self._filter_invalid_edges(predictions_dict, global_rotations)
            global_rotations = rotation_averaging(predictions_dict, global_rotations)
            self._filter_invalid_edges(predictions_dict, global_rotations)
        else:
            global_rotations = rotation_averaging_pycolmap(
                predictions_dict, max_rotation_error_deg=self.max_rot_error
            )
            self._filter_invalid_edges(predictions_dict, global_rotations)

        self._prune_invisible_pairs(predictions_dict)

        # Initialize the structures by maximum spanning tree
        global_centers, global_scales = self._initialize_mst_structures(
            predictions_dict, global_rotations
        )

        global_centers = similarity_averaging(
            predictions_dict,
            global_rotations,
            global_centers=global_centers,
            global_scales=global_scales,
            max_num_iterations=200,
        )

        # Prune the edges by the global rotations
        self._mark_inconsistent_edges(
            predictions_dict, global_rotations, global_centers
        )

        # Check whether all images are estimated
        logger.info(f"Number of images: {self.N}")
        logger.info(f"Number of global rotations: {len(global_rotations)}")
        logger.info(f"Number of global centers: {len(global_centers)}")
        if len(global_rotations) != self.N or len(global_centers) != self.N:
            global_rotations = {
                i: (
                    global_rotations[i]
                    if i in global_rotations
                    else np.eye(3, dtype=np.float64)
                )
                for i in range(self.N)
            }
            global_centers = {
                i: (
                    global_centers[i]
                    if i in global_centers
                    else np.zeros(3, dtype=np.float64)
                )
                for i in range(self.N)
            }

        # # Establish tracks for bundle adjustment
        # track_options = TrackEstablishmentOptions(track_min_num_views_per_track=2)
        # (
        #     points3D,
        #     keypoints_per_image,
        #     pts2d_idx_all,
        #     pts2d_idx_virtual_all,
        #     pts2d_idx_inv,
        #     pts2d_idx_virtual_inv,
        #     image_to_point3D,
        #     images_points2d_virtual_isnegative,
        # ) = establish_tracks_from_tracks_dict(
        #     tracks_dict=predictions_dict,
        #     num_images=self.N,
        #     options=track_options,
        #     add_tracks=True,
        #     add_virtual_points=True,
        #     device="cuda",
        # )

        # # Initialize 3D world points
        # points3D = initialize_world_points(
        #     predictions_dict,
        #     global_rotations,
        #     global_centers,
        #     points3D,
        #     pts2d_idx_inv,
        #     pts2d_idx_virtual_inv,
        #     keypoints_per_image=keypoints_per_image,
        #     intrinsics_params=intrinsics_params,
        #     intrinsics_mapping=intrinsics_mapping,
        #     reproj_threshold=10.0,
        # )

        # global_rotations, global_centers, global_intrinsics, points3D = bundle_adjustment(
        #     predictions_dict,
        #     global_rotations,
        #     global_centers,
        #     global_intrinsics,
        #     intrinsics_mapping,
        #     points3D=points3D,
        #     keypoints_per_image=keypoints_per_image,
        #     pts2d_idx_inv=pts2d_idx_inv,
        #     pts2d_idx_virtual_inv=pts2d_idx_virtual_inv,
        #     images_points2d_virtual_isnegative=images_points2d_virtual_isnegative,
        #     max_num_iterations=200,
        #     camera_model=camera_model,
        #     skip_world_points_init=True,  # Already initialized above
        # )

        return global_rotations, global_centers

    def _estimate_intrinsics(self, predictions_dict, intrinsics_mapping, camera_model):
        indexes = get_tracks_dict_indexes(predictions_dict)
        intrinsics_all = [predictions_dict["intrinsics"][idx] for idx in indexes]
        members = [predictions_dict["indexes"][idx] for idx in indexes]

        global_intrinsics = intrinsics_averaging(
            intrinsics_all, members, intrinsics_mapping, camera_model
        )

        return global_intrinsics

    def _prune_invisible_pairs(self, predictions_dict):
        indexes = get_tracks_dict_indexes(predictions_dict)
        for idx in indexes:
            if len(predictions_dict["indexes"][idx]) == 1:
                continue
            valid_edges_curr = (
                torch.where(predictions_dict["pose_scores"][idx][0] > 0)[0]
            ).tolist()
            if len(valid_edges_curr) == predictions_dict["pose_scores"][idx].shape[1]:
                continue
            for key in predictions_dict.keys():
                if key == "indexes":
                    predictions_dict[key][idx] = [
                        predictions_dict[key][idx][x] for x in valid_edges_curr
                    ]
                elif (
                    key == "points3d_virtual"
                    or key == "scales"
                    or key == "star_indexes"
                    or key == "image_index_to_star_index"
                ):
                    continue
                else:
                    if predictions_dict[key][idx] is None:
                        continue
                    predictions_dict[key][idx] = predictions_dict[key][idx][
                        :, valid_edges_curr
                    ]
                    # except:
                    #     breakpoint()

    def _filter_invalid_edges(self, predictions_dict, global_rotations):
        indexes = get_tracks_dict_indexes(predictions_dict)
        thres = np.deg2rad(self.max_rot_error)
        num_filtered = 0
        num_total = 0
        filtered_index = []
        for idx in indexes:
            rotation_local = predictions_dict["extrinsics"][idx][0, :, :3, :3]
            rotation_js = torch.stack(
                [
                    torch.from_numpy(global_rotations[idx_inner])
                    for idx_inner in predictions_dict["indexes"][idx]
                ],
                dim=0,
            )
            rotations_global = rotation_js @ torch.from_numpy(
                global_rotations[predictions_dict["indexes"][idx][0]].T
            ).unsqueeze(0)

            diff = (
                rotations_global @ rotation_local.transpose(-1, -2).cpu().double()
            )  # (N, 3, 3)

            errors = torch.acos(
                torch.clamp(
                    (torch.einsum("bii -> b", diff) - 1) / 2,
                    min=-1.0 + 1e-6,
                    max=1.0 - 1e-6,
                )
            )
            invalid_mask = errors > thres

            invalid_mask = invalid_mask * (
                predictions_dict["pose_scores"][idx][0].cpu() > 0
            )

            num_total += rotation_local.shape[0] - 1

            if not invalid_mask.any():
                continue

            num_filtered += invalid_mask.sum().item()

            filtered_index.extend(
                [
                    (
                        idx,
                        i,
                        predictions_dict["pose_scores"][idx][0, i].item(),
                        errors[i],
                    )
                    for i in range(len(predictions_dict["indexes"][idx]))
                    if invalid_mask[i]
                ]
            )
            predictions_dict["pose_scores"][idx][0, invalid_mask] = 0.0

        if num_filtered > 0:
            logger.info(
                f"Number of filtered edges by the rotation error / total: "
                f"{num_filtered} / {num_total}"
            )
        return filtered_index

    # Filter edges by both rotation and translation consistency
    # TODO: refactor this part so that it can reuse the rotation filtering code
    def _mark_inconsistent_edges(
        self, predictions_dict, global_rotations, global_centers
    ):
        indexes = get_tracks_dict_indexes(predictions_dict)
        thres_rot = np.deg2rad(1.0)
        thres_trans = np.deg2rad(5.0)
        num_filtered = 0
        num_total = 0
        predictions_dict["pose_inconsistent"] = {}
        for idx in indexes:
            # Compute the relative translations
            translation_local = predictions_dict["extrinsics"][idx][
                0, :, :3, 3
            ].double()

            rotations = torch.from_numpy(
                np.stack(
                    [global_rotations[idx] for idx in predictions_dict["indexes"][idx]]
                )
            ).unsqueeze(0)
            translations = torch.from_numpy(
                np.stack(
                    [
                        -global_rotations[idx] @ global_centers[idx].reshape(3, 1)
                        for idx in predictions_dict["indexes"][idx]
                    ]
                )
            ).unsqueeze(0)
            extrinsics = torch.cat([rotations, translations], dim=-1)  # (B, N, 3, 4)
            extrinsics = restore_identity(extrinsics)
            translation_global = extrinsics[0, :, :3, 3]

            # Normalize
            safe_norm_global = torch.clamp(
                torch.linalg.norm(translation_global, dim=-1, keepdim=True), min=1e-6
            )
            translation_global = translation_global / safe_norm_global
            safe_norm_local = torch.clamp(
                torch.linalg.norm(translation_local, dim=-1, keepdim=True), min=1e-6
            )
            translation_local = translation_local / safe_norm_local

            # Compare the directions
            errors_trans = torch.acos(
                torch.clamp(
                    torch.einsum("bi,bi->b", translation_global, translation_local),
                    min=-1.0 + 1e-6,
                    max=1.0 - 1e-6,
                )
            )

            # compare the rotation
            rotations_local = predictions_dict["extrinsics"][idx][0, :, :3, :3]
            diff = (
                extrinsics[0, :, :3, :3]
                @ rotations_local.transpose(-1, -2).cpu().double()
            )  # (N, 3, 3)

            errors_rot = torch.acos(
                torch.clamp(
                    (torch.einsum("bii -> b", diff) - 1) / 2,
                    min=-1.0 + 1e-6,
                    max=1.0 - 1e-6,
                )
            )
            invalid_mask = (errors_rot > thres_rot) | (errors_trans > thres_trans)
            invalid_mask = invalid_mask * (
                predictions_dict["pose_scores"][idx][0].cpu() > 0
            )
            invalid_mask[0] = False  # never filter the first one

            predictions_dict["pose_inconsistent"][idx] = invalid_mask

            num_total += translation_local.shape[0] - 1

            num_filtered += invalid_mask.sum().item()

        if num_filtered > 0:
            logger.info(
                f"Number of filtered edges by the rotation error / total: "
                f"{num_filtered} / {num_total}"
            )

    def _boost_sequential_edges(self, predictions_dict, boost_factor=2.0):
        """
        Boost weights of sequential edges (neighboring frames).

        Args:
            predictions_dict: Dictionary containing pose_scores and indexes
            boost_factor: Factor to multiply the weight by (default: 2.0)
        """
        indexes = get_tracks_dict_indexes(predictions_dict)
        seq_edges = getattr(self, "sequential_edges", set())
        num_boosted = 0

        for idx in indexes:
            center_idx = predictions_dict["indexes"][idx][0]
            for i, neighbor_idx in enumerate(predictions_dict["indexes"][idx]):
                if i == 0:
                    continue
                edge = (min(center_idx, neighbor_idx), max(center_idx, neighbor_idx))
                if edge in seq_edges:
                    # predictions_dict["pose_scores"][idx][0, i] *= 1 + boost_factor * (1 / abs(center_idx - neighbor_idx))
                    predictions_dict["pose_scores"][idx][0, i] *= 2
                    num_boosted += 1
                else:
                    predictions_dict["pose_scores"][idx][0, i] /= 1
                    
        logger.info(f"Boosted {num_boosted} sequential edges by factor {boost_factor}")

    def _initialize_mst_structures(self, predictions_dict, global_rotations):
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
            # max_median_angle = median_angles.max().item()

            predictions_dict["median_tri_angle"][idx] = np.rad2deg(median_angles) if median_angles.numel() > 0 else np.array([])


        for idx in indexes:
            poses = predictions_dict["extrinsics"][idx]
            pose_scores = predictions_dict["pose_scores"][idx]
            N = poses.shape[1]
            idx_i = predictions_dict["indexes"][idx][0]
            for i in range(N):
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
                        # score *= 0.1  # downweight the edge if the scale is unreliable
                    else:
                        scales[(star_idx_j, star_idx_i)] = (
                            poses[0, i, :3, 3:].norm().item()
                            / rel_poses[(idx_j, idx_i)][0][:3, 3:].norm().item()
                        )
                        scales[(star_idx_i, star_idx_j)] = (
                            1.0 / scales[(star_idx_j, star_idx_i)]
                        )
                    score *= min(20, angle_current, angle_reverse) / 20  # downweight the edge if the triangulation angle is small

                    # if pose_scores[0, i] < rel_poses[(idx_j, idx_i)][3]:
                    #     continue
                    # status = True
                    # rel_poses.pop((idx_j, idx_i))

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
                # del rel_poses[(i, j)]
                invalid_edges.append((i, j))

        for i, j in invalid_edges:
            del rel_poses[(i, j)]

        G = nx.Graph()
        G.add_nodes_from(np.arange(self.N))
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
