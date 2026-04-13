"""Tests for similarity_averaging using pycolmap synthetic datasets."""

import logging

import numpy as np

from gluemap.estimators.similarity_averaging import similarity_averaging
from tests.helpers import (
    build_predictions_dict,
    build_star_topology_full,
    build_star_topology_sparse,
    create_synthetic_reconstruction,
    evaluate_centers,
    extract_gt,
    max_center_error,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimilarityAveragingFixedScales:
    """Tests with fix_scales=True (scales provided at GT values)."""

    def test_fully_connected(self):
        """Clean data, fully connected stars, fixed GT scales."""
        logger.info("=== Test: FixedScales / fully_connected ===")
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=0)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_full(image_ids)

        rng = np.random.default_rng(123)
        gt_scales_values = rng.uniform(0.5, 3.0, size=len(stars))

        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values
        )

        # Perturbed initial centers
        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }
        init_scales = [
            np.ones(1, dtype=np.float64) * gt_scales_values[i]
            for i in range(len(stars))
        ]

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=init_scales,
            max_num_iterations=200,
            fix_scales=True,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"

    def test_sparse_neighbors(self):
        """Clean data, sparse (k=3) neighbors, fixed GT scales."""
        logger.info("=== Test: FixedScales / sparse_neighbors ===")
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=1)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_sparse(image_ids, gt_centers, k=3)

        rng = np.random.default_rng(456)
        gt_scales_values = rng.uniform(0.5, 3.0, size=len(stars))

        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values
        )

        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }
        init_scales = [
            np.ones(1, dtype=np.float64) * gt_scales_values[i]
            for i in range(len(stars))
        ]

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=init_scales,
            max_num_iterations=200,
            fix_scales=True,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"


class TestSimilarityAveragingFreeScales:
    """Tests with fix_scales=False (scales optimized)."""

    def test_fully_connected(self):
        """Clean data, fully connected, free scales (uniform GT scale=1)."""
        logger.info("=== Test: FreeScales / fully_connected ===")
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=2)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values
        )

        rng = np.random.default_rng(789)
        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"

    def test_sparse_neighbors(self):
        """Clean data, sparse (k=3), free scales."""
        logger.info("=== Test: FreeScales / sparse_neighbors ===")
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=3)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_sparse(image_ids, gt_centers, k=3)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values
        )

        rng = np.random.default_rng(101)
        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"


class TestSimilarityAveragingRobustness:
    """Tests with noise and outliers."""

    def test_with_noise(self):
        """Translation noise (stddev=0.01), fully connected, free scales."""
        logger.info("=== Test: Robustness / with_noise ===")
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=4)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        rng = np.random.default_rng(202)
        predictions_dict = build_predictions_dict(
            gt_rotations,
            gt_centers,
            stars,
            gt_scales_values,
            translation_noise_std=0.01,
            rng=rng,
        )

        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-2, f"Max center error {max_err:.6e} >= 1e-2"

    def test_with_outliers(self):
        """~10% outlier edges with low scores, fully connected, free scales."""
        logger.info("=== Test: Robustness / with_outliers ===")
        gt_rec = create_synthetic_reconstruction(num_frames=8, seed=5)
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        rng = np.random.default_rng(303)
        predictions_dict = build_predictions_dict(
            gt_rotations,
            gt_centers,
            stars,
            gt_scales_values,
            outlier_ratio=0.1,
            outlier_score=0.1,
            rng=rng,
        )

        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-3, f"Max center error {max_err:.6e} >= 1e-3"


class TestSimilarityAveragingMultiRig:
    """Tests with multiple camera rigs (more images, diverse baselines)."""

    def test_multi_rig_fully_connected(self):
        """3 rigs x 4 frames = 12 images, fully connected, free scales."""
        logger.info("=== Test: MultiRig / fully_connected ===")
        gt_rec = create_synthetic_reconstruction(
            num_frames=4, num_points3D=100, num_rigs=3, seed=10
        )
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values
        )

        rng = np.random.default_rng(404)
        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"

    def test_multi_rig_sparse(self):
        """3 rigs x 4 frames = 12 images, sparse (k=3), free scales."""
        logger.info("=== Test: MultiRig / sparse ===")
        gt_rec = create_synthetic_reconstruction(
            num_frames=4, num_points3D=100, num_rigs=3, seed=11
        )
        image_ids, gt_rotations, gt_centers = extract_gt(gt_rec)
        stars = build_star_topology_sparse(image_ids, gt_centers, k=3)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = build_predictions_dict(
            gt_rotations, gt_centers, stars, gt_scales_values
        )

        rng = np.random.default_rng(505)
        init_centers = {
            img_id: c + rng.normal(0, 0.5, size=3)
            for img_id, c in gt_centers.items()
        }

        recovered = similarity_averaging(
            predictions_dict,
            gt_rotations,
            global_centers={
                k: v.astype(np.float64) for k, v in init_centers.items()
            },
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"
