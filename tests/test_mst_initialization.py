"""Tests for initialize_mst_structures using pycolmap synthetic datasets."""

import logging

import numpy as np
import torch

from gluemap.math.mst_initialization import initialize_mst_structures
from tests.helpers import (
    create_synthetic_reconstruction,
    extract_gt,
    build_star_topology_full,
    build_star_topology_sparse,
    build_predictions_dict,
    evaluate_centers,
    max_center_error,
    remap_to_original_ids,
)

logger = logging.getLogger(__name__)


class TestMSTInitialization:
    """Basic correctness tests for MST-based center/scale initialization."""

    def test_fully_connected_clean(self):
        """Clean data, fully connected stars, uniform scales."""
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=100)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec, zero_indexed=True)
        stars = build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values,
            generate_points3d_virtual=True,
        )

        global_centers, global_scales = initialize_mst_structures(
            predictions_dict, gt_rotations
        )

        assert len(global_centers) == len(image_ids)
        assert len(global_scales) == len(stars)

        orig_rotations = remap_to_original_ids(gt_rotations, gt_rec)
        orig_centers = remap_to_original_ids(global_centers, gt_rec)
        errors = evaluate_centers(gt_rec, orig_rotations, orig_centers)
        max_err = max_center_error(errors)
        logger.info(f"fully_connected_clean max center error: {max_err:.6e}")
        assert max_err < 1e-4, f"Max center error {max_err:.6e} >= 1e-4"

    def test_sparse_topology(self):
        """Clean data, sparse (k=3) neighbors, uniform scales."""
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=101)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec, zero_indexed=True)
        stars = build_star_topology_sparse(image_ids, gt_centers, k=3)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values,
            generate_points3d_virtual=True,
        )

        global_centers, global_scales = initialize_mst_structures(
            predictions_dict, gt_rotations
        )

        assert len(global_centers) == len(image_ids)

        orig_rotations = remap_to_original_ids(gt_rotations, gt_rec)
        orig_centers = remap_to_original_ids(global_centers, gt_rec)
        errors = evaluate_centers(gt_rec, orig_rotations, orig_centers)
        max_err = max_center_error(errors)
        logger.info(f"sparse_topology max center error: {max_err:.6e}")
        assert max_err < 1e-3, f"Max center error {max_err:.6e} >= 1e-3"

    def test_non_uniform_scales(self):
        """Clean data, fully connected, non-uniform GT scales."""
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=200)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec, zero_indexed=True)
        stars = build_star_topology_full(image_ids)

        rng = np.random.default_rng(200)
        gt_scales_values = rng.uniform(0.5, 3.0, size=len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values,
            generate_points3d_virtual=True, rng=rng,
        )

        global_centers, global_scales = initialize_mst_structures(
            predictions_dict, gt_rotations
        )

        # Centers should still align to GT via Sim3
        orig_rotations = remap_to_original_ids(gt_rotations, gt_rec)
        orig_centers = remap_to_original_ids(global_centers, gt_rec)
        errors = evaluate_centers(gt_rec, orig_rotations, orig_centers)
        max_err = max_center_error(errors)
        logger.info(f"non_uniform_scales max center error: {max_err:.6e}")
        assert max_err < 1e-3, f"Max center error {max_err:.6e} >= 1e-3"

        # Scale ratios should be consistent with GT.
        # global_scales[i] / global_scales[j] ~ gt_scales[i] / gt_scales[j]
        # up to a global multiplier. Check the ratio-of-ratios is constant.
        star_indices = sorted(global_scales.keys())
        if len(star_indices) >= 2:
            ref = star_indices[0]
            ratios = []
            for si in star_indices[1:]:
                gt_ratio = gt_scales_values[si] / gt_scales_values[ref]
                rec_ratio = global_scales[si] / global_scales[ref]
                if gt_ratio > 1e-8:
                    ratios.append(rec_ratio / gt_ratio)
            if ratios:
                ratios = np.array(ratios)
                # All ratio-of-ratios should be approximately the same constant
                spread = ratios.max() / ratios.min() if ratios.min() > 1e-12 else float("inf")
                logger.info(f"Scale ratio spread: {spread:.4f}")
                assert spread < 1.5, f"Scale ratio spread {spread:.4f} >= 1.5"


class TestMSTInitializationTriangulationAngle:
    """Tests for triangulation angle computation and small-angle fallback."""

    def test_small_triangulation_angle_fallback(self):
        """Stars with near-collinear 3D points should trigger scale=1.0 fallback."""
        gt_rec = create_synthetic_reconstruction(num_frames=4, seed=300)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec, zero_indexed=True)
        stars = build_star_topology_full(image_ids)

        rng = np.random.default_rng(300)
        gt_scales_values = rng.uniform(0.5, 3.0, size=len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values,
            generate_points3d_virtual=True, rng=rng,
        )

        # Override points3d_virtual for all stars with near-collinear points
        # (very small x/y, large z) so triangulation angles are tiny
        for idx_star in predictions_dict["points3d_virtual"]:
            n_pts = 50
            collinear = np.zeros((n_pts, 3))
            collinear[:, 0] = rng.uniform(-0.0001, 0.0001, size=n_pts)
            collinear[:, 1] = rng.uniform(-0.0001, 0.0001, size=n_pts)
            collinear[:, 2] = rng.uniform(1000, 2000, size=n_pts)
            predictions_dict["points3d_virtual"][idx_star] = (
                torch.from_numpy(collinear).float().unsqueeze(0)
            )

        global_centers, global_scales = initialize_mst_structures(
            predictions_dict, gt_rotations
        )

        # Verify median_tri_angle was computed and is small
        for idx_star in predictions_dict["median_tri_angle"]:
            angles = predictions_dict["median_tri_angle"][idx_star]
            if len(angles) > 0:
                assert angles.max() < 1.0, (
                    f"Star {idx_star}: expected small tri angles, got max={angles.max():.2f}"
                )

        # Function should still produce valid output (no crash)
        assert len(global_centers) > 0
        assert len(global_scales) > 0


class TestMSTInitializationDisconnected:
    """Tests with disconnected graph components."""

    def test_disconnected_graph(self):
        """Nodes in a disconnected component should not appear in global_centers."""
        gt_rec = create_synthetic_reconstruction(num_frames=6, seed=400)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec, zero_indexed=True)

        # Build two separate clusters: {0,1,2} fully connected, {3,4,5} fully connected
        # No cross-edges between the clusters
        cluster_a = image_ids[:3]
        cluster_b = image_ids[3:]
        stars = {}
        for i, center_id in enumerate(cluster_a):
            neighbors = [img_id for img_id in cluster_a if img_id != center_id]
            stars[i] = [center_id] + neighbors
        for i, center_id in enumerate(cluster_b):
            neighbors = [img_id for img_id in cluster_b if img_id != center_id]
            stars[len(cluster_a) + i] = [center_id] + neighbors

        gt_scales_values = np.ones(len(stars))
        rng = np.random.default_rng(400)
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values,
            generate_points3d_virtual=True, rng=rng,
        )

        global_centers, global_scales = initialize_mst_structures(
            predictions_dict, gt_rotations
        )

        # Node 0 is the DFS root, so cluster_a (containing node 0) should be recovered
        # Cluster_b nodes should NOT be reachable from node 0
        for img_id in cluster_a:
            assert img_id in global_centers, f"Image {img_id} (cluster A) missing from global_centers"
        for img_id in cluster_b:
            assert img_id not in global_centers, (
                f"Image {img_id} (cluster B) should not be reachable from node 0"
            )
