"""Tests for similarity_averaging using pycolmap synthetic datasets."""

import logging
import tempfile
from pathlib import Path

import numpy as np
import pycolmap
import pytest
import torch

from gluemap.estimators.similarity_averaging import similarity_averaging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_synthetic_reconstruction(num_frames=8, num_points3D=100, num_rigs=1, seed=0):
    """Create a pycolmap synthetic reconstruction with known ground truth."""
    pycolmap.set_random_seed(seed)
    opts = pycolmap.SyntheticDatasetOptions()
    opts.num_rigs = num_rigs
    opts.num_cameras_per_rig = 1
    opts.num_frames_per_rig = num_frames
    opts.num_points3D = num_points3D
    return pycolmap.synthesize_dataset(opts)


def _extract_gt(reconstruction):
    """Extract ground-truth rotations and centers from a reconstruction."""
    image_ids = sorted(reconstruction.reg_image_ids())
    gt_rotations = {}
    gt_centers = {}
    for img_id in image_ids:
        img = reconstruction.image(img_id)
        pose = img.cam_from_world()
        gt_rotations[img_id] = np.array(pose.rotation.matrix())
        gt_centers[img_id] = np.array(img.projection_center())
    return image_ids, gt_rotations, gt_centers


def _build_star_topology_full(image_ids):
    """One star per image, fully connected to all others."""
    stars = {}
    for i, center_id in enumerate(image_ids):
        neighbors = [img_id for img_id in image_ids if img_id != center_id]
        stars[i] = [center_id] + neighbors
    return stars


def _build_star_topology_sparse(image_ids, gt_centers, k=3):
    """One star per image, connected to k nearest neighbors."""
    centers_array = np.array([gt_centers[i] for i in image_ids])
    stars = {}
    for i, center_id in enumerate(image_ids):
        dists = np.linalg.norm(centers_array - centers_array[i], axis=1)
        # Exclude self (dist=0), take k nearest
        neighbor_indices = np.argsort(dists)[1 : k + 1]
        neighbors = [image_ids[j] for j in neighbor_indices]
        stars[i] = [center_id] + neighbors
    return stars


def _build_predictions_dict(
    gt_rotations,
    gt_centers,
    stars,
    gt_scales,
    translation_noise_std=0.0,
    outlier_ratio=0.0,
    outlier_score=0.1,
    rng=None,
):
    """Build predictions_dict from ground-truth data and star topology.

    Args:
        gt_rotations: dict image_id -> (3,3) rotation matrix
        gt_centers: dict image_id -> (3,) center
        stars: dict idx_star -> list of image_ids (first is center)
        gt_scales: list of floats, one per star
        translation_noise_std: stddev of Gaussian noise on translations
        outlier_ratio: fraction of edges to corrupt with random translations
        outlier_score: pose_score assigned to outlier edges
        rng: numpy random generator
    """
    if rng is None:
        rng = np.random.default_rng(42)

    predictions_dict = {
        "indexes": {},
        "pose_scores": {},
        "extrinsics": {},
        "median_tri_angle": {},
    }

    for idx_star, img_ids in stars.items():
        n = len(img_ids)
        center_id = img_ids[0]
        scale = gt_scales[idx_star]

        extrinsics = torch.zeros(1, n, 3, 4)
        scores = torch.ones(1, n)
        tri_angles = torch.full((n - 1,), 15.0)

        for j in range(1, n):
            neighbor_id = img_ids[j]
            R_j = gt_rotations[neighbor_id]
            direction = gt_centers[neighbor_id] - gt_centers[center_id]
            t_ij = -R_j @ (scale * direction)

            # Add noise
            if translation_noise_std > 0:
                t_ij = t_ij + rng.normal(0, translation_noise_std, size=3)

            # Outlier corruption
            if outlier_ratio > 0 and rng.random() < outlier_ratio:
                t_ij = rng.normal(0, 1, size=3)
                scores[0, j] = outlier_score

            extrinsics[0, j, :3, :3] = torch.from_numpy(
                R_j @ gt_rotations[center_id].T
            ).float()
            extrinsics[0, j, :3, 3] = torch.from_numpy(t_ij).float()

        predictions_dict["indexes"][idx_star] = img_ids
        predictions_dict["pose_scores"][idx_star] = scores
        predictions_dict["extrinsics"][idx_star] = extrinsics
        predictions_dict["median_tri_angle"][idx_star] = tri_angles

    return predictions_dict


def _evaluate_centers(gt_reconstruction, gt_rotations, recovered_centers):
    """Align recovered centers to GT via compare_reconstructions and return errors.

    Writes GT to disk, reads a copy, overwrites poses with recovered centers
    (keeping GT rotations), then uses pycolmap.compare_reconstructions to
    align via Sim3 and compute per-image errors.

    Returns list of ImageAlignmentError objects.
    """
    image_ids = sorted(recovered_centers.keys())

    with tempfile.TemporaryDirectory() as tmpdir:
        gt_path = Path(tmpdir) / "gt"
        gt_path.mkdir()
        gt_reconstruction.write(gt_path)

        # Read a copy and overwrite poses with recovered centers
        rec = pycolmap.Reconstruction(gt_path)
        for img_id in image_ids:
            img = rec.image(img_id)
            R = gt_rotations[img_id]
            c = recovered_centers[img_id]
            t = -R @ c
            mat = np.hstack([R, t.reshape(3, 1)])
            new_pose = pycolmap.Rigid3d(mat)
            img.frame.set_cam_from_world(img.camera_id, new_pose)

        rec_path = Path(tmpdir) / "rec"
        rec_path.mkdir()
        rec.write(rec_path)

        rec_loaded = pycolmap.Reconstruction(rec_path)
        gt_loaded = pycolmap.Reconstruction(gt_path)

        result = pycolmap.compare_reconstructions(
            rec_loaded,
            gt_loaded,
            alignment_error="proj_center",
            max_proj_center_error=100.0,
        )

    assert result is not None, "compare_reconstructions returned None (alignment failed)"
    return result["errors"]


def _max_center_error(errors):
    """Extract max projection center error from alignment errors."""
    return max(e.proj_center_error for e in errors)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSimilarityAveragingFixedScales:
    """Tests with fix_scales=True (scales provided at GT values)."""

    def test_fully_connected(self):
        """Clean data, fully connected stars, fixed GT scales."""
        logger.info("=== Test: FixedScales / fully_connected ===")
        gt_rec = _create_synthetic_reconstruction(num_frames=8, seed=0)
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_full(image_ids)

        rng = np.random.default_rng(123)
        gt_scales_values = rng.uniform(0.5, 3.0, size=len(stars))

        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=init_scales,
            max_num_iterations=200,
            fix_scales=True,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"

    def test_sparse_neighbors(self):
        """Clean data, sparse (k=3) neighbors, fixed GT scales."""
        logger.info("=== Test: FixedScales / sparse_neighbors ===")
        gt_rec = _create_synthetic_reconstruction(num_frames=8, seed=1)
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_sparse(image_ids, gt_centers, k=3)

        rng = np.random.default_rng(456)
        gt_scales_values = rng.uniform(0.5, 3.0, size=len(stars))

        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=init_scales,
            max_num_iterations=200,
            fix_scales=True,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"


class TestSimilarityAveragingFreeScales:
    """Tests with fix_scales=False (scales optimized)."""

    def test_fully_connected(self):
        """Clean data, fully connected, free scales (uniform GT scale=1)."""
        logger.info("=== Test: FreeScales / fully_connected ===")
        gt_rec = _create_synthetic_reconstruction(num_frames=8, seed=2)
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"

    def test_sparse_neighbors(self):
        """Clean data, sparse (k=3), free scales."""
        logger.info("=== Test: FreeScales / sparse_neighbors ===")
        gt_rec = _create_synthetic_reconstruction(num_frames=8, seed=3)
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_sparse(image_ids, gt_centers, k=3)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"


class TestSimilarityAveragingRobustness:
    """Tests with noise and outliers."""

    def test_with_noise(self):
        """Translation noise (stddev=0.01), fully connected, free scales."""
        logger.info("=== Test: Robustness / with_noise ===")
        gt_rec = _create_synthetic_reconstruction(num_frames=8, seed=4)
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        rng = np.random.default_rng(202)
        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-2, f"Max center error {max_err:.6e} >= 1e-2"

    def test_with_outliers(self):
        """~10% outlier edges with low scores, fully connected, free scales."""
        logger.info("=== Test: Robustness / with_outliers ===")
        gt_rec = _create_synthetic_reconstruction(num_frames=8, seed=5)
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        rng = np.random.default_rng(303)
        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-3, f"Max center error {max_err:.6e} >= 1e-3"


class TestSimilarityAveragingMultiRig:
    """Tests with multiple camera rigs (more images, diverse baselines)."""

    def test_multi_rig_fully_connected(self):
        """3 rigs x 4 frames = 12 images, fully connected, free scales."""
        logger.info("=== Test: MultiRig / fully_connected ===")
        gt_rec = _create_synthetic_reconstruction(
            num_frames=4, num_points3D=100, num_rigs=3, seed=10
        )
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_full(image_ids)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"

    def test_multi_rig_sparse(self):
        """3 rigs x 4 frames = 12 images, sparse (k=3), free scales."""
        logger.info("=== Test: MultiRig / sparse ===")
        gt_rec = _create_synthetic_reconstruction(
            num_frames=4, num_points3D=100, num_rigs=3, seed=11
        )
        image_ids, gt_rotations, gt_centers = _extract_gt(gt_rec)
        stars = _build_star_topology_sparse(image_ids, gt_centers, k=3)

        gt_scales_values = np.ones(len(stars))
        predictions_dict = _build_predictions_dict(
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
            global_centers={k: v.astype(np.float64) for k, v in init_centers.items()},
            global_scales=None,
            max_num_iterations=200,
            fix_scales=False,
        )

        errors = _evaluate_centers(gt_rec, gt_rotations, recovered)
        max_err = _max_center_error(errors)
        logger.info(f"Max center error: {max_err:.6e}")
        assert max_err < 1e-7, f"Max center error {max_err:.6e} >= 1e-7"
