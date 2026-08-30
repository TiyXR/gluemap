import numpy as np

from gluemap.estimators.rotation_averaging import rotation_averaging
from tests.helpers import (
    build_predictions_dict,
    build_star_topology_full,
    create_synthetic_reconstruction,
    extract_gt,
)


def test_fixed_anchor_rotation_remains_in_canonical_gauge() -> None:
    reconstruction = create_synthetic_reconstruction(num_frames=8, seed=41)
    image_ids, ground_truth_rotations, ground_truth_centers = extract_gt(
        reconstruction
    )
    predictions = build_predictions_dict(
        ground_truth_rotations,
        ground_truth_centers,
        build_star_topology_full(image_ids),
        np.ones(len(image_ids)),
    )
    fixed_id = image_ids[0]

    recovered = rotation_averaging(
        predictions,
        init_rotations={
            image_id: rotation.copy()
            for image_id, rotation in ground_truth_rotations.items()
        },
        fixed_rotation_ids={fixed_id},
    )

    assert np.allclose(
        recovered[fixed_id], ground_truth_rotations[fixed_id], atol=1e-12
    )
