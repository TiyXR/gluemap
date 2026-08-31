from __future__ import annotations

import numpy as np
import pytest
import torch

from gluemap.estimators.dense_pose_track_inference import DensePoseTrackInference


class FakeTracker(torch.nn.Module):
    def process_images_to_fmaps(self, images: torch.Tensor) -> torch.Tensor:
        return images.mean(dim=1, keepdim=True)

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor,
        *,
        fmaps: torch.Tensor | None,
        inference: bool,
    ):
        frame_count = images.shape[1]
        offsets = torch.arange(
            frame_count, device=images.device, dtype=query_points.dtype
        ).reshape(1, frame_count, 1, 1)
        tracks = query_points[:, None, :, :].expand(-1, frame_count, -1, -1)
        tracks = tracks + torch.cat((offsets, offsets * 2.0), dim=-1)
        visibility = torch.ones(tracks.shape[:-1], device=images.device)
        confidence = torch.ones_like(visibility)
        return tracks, None, visibility, confidence


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tracks_reference_landmarks_across_query_microbatch_on_gpu() -> None:
    tracker = DensePoseTrackInference(
        FakeTracker().cuda(), feature_cache_frames=4
    )
    images_518 = torch.zeros((3, 3, 32, 32), device="cuda")
    images_1024 = torch.zeros((3, 3, 64, 64), device="cuda")
    changes = np.tile(np.array([1.0, 1.0, 0.0, 0.0]), (3, 1))
    reference = torch.tensor([[10.0, 20.0], [30.0, 40.0]], device="cuda")

    values = tracker.predict(
        images_518=images_518,
        images_1024=images_1024,
        image_changes_518=changes,
        image_changes_1024=changes,
        frame_uids=["a", "b", "c"],
        reference_xy_518=reference,
        minimum_visibility=0.5,
        minimum_confidence=0.5,
    )

    assert len(values) == 2
    torch.testing.assert_close(
        values[0].query_xy,
        reference + torch.tensor([1.0, 2.0], device="cuda"),
    )
    torch.testing.assert_close(
        values[1].query_xy,
        reference + torch.tensor([2.0, 4.0], device="cuda"),
    )
    assert values[0].visible.all()
    report = tracker.cache_report()
    assert report is not None
    assert report["extractedFrameCount"] == 3
