import time

import torch

from gluemap.estimators.covisibility_extraction import CovisibilityExtraction
from gluemap.estimators.local_inference import create_local_inference
from gluemap.estimators.track_inference import TrackInference


class BatchInferenceStar:
    def __init__(
        self,
        model,
        model_type="pi3",
        model_track=None,
        device="cuda",
        dtype=torch.bfloat16,
        pointmap_dir=None,
    ):
        self.model = model
        self.model_type = model_type
        self.model_track = model_track
        self.device = device
        self.dtype = dtype
        self.pointmap_dir = pointmap_dir

        self.local_inference = create_local_inference(model, model_type, device, dtype)
        self.track_inference = TrackInference(model_track, device)
        self.covisibility_extraction = CovisibilityExtraction()

    def main(self, batch, disable_track=False, include_track=True):
        predictions, forward_time, track_time = self._predict_images(
            batch, disable_track=disable_track, include_track=include_track
        )

        (
            extrinsics,
            intrinsics,
            scores,
            tracks_virtual,
            points3d_virtual,
            valid_virtual,
            cam_points,
            cam_points_conf,
            depth_transformed,
        ) = self.covisibility_extraction.main(
            predictions,
            batch["indexes"],
            batch["images_change"],
        )

        result_dict = {
            "indexes": batch["indexes"][0].tolist(),
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "pose_scores": scores,
            "tracks_virtual": tracks_virtual,
            "points3d_virtual": points3d_virtual,
            "valid_virtual": valid_virtual,
            "_forward_time": forward_time,
            "_track_time": track_time,
        }

        if include_track:
            result_dict["tracks"] = predictions["track"].cpu()
            result_dict["vis"] = predictions["vis"].cpu()
            result_dict["conf"] = predictions["conf"].cpu()

        return result_dict

    @torch.no_grad()
    def _predict_images(self, batch, disable_track=False, include_track=True):
        if not disable_track and not include_track:
            raise ValueError("if not disable_track, include_track should be True")

        # Local inference (timed)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        predictions = self.local_inference.predict(batch)
        torch.cuda.synchronize()
        forward_time = time.perf_counter() - t0

        # Track inference
        track_preds, track_time = self._run_track_inference(
            batch, disable_track, include_track
        )
        predictions.update(track_preds)

        return predictions, forward_time, track_time

    def _run_track_inference(self, batch, disable_track, include_track):
        """Returns (track_preds_dict, track_time)."""
        if not include_track:
            return {}, 0.0

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        track_preds = self.track_inference.predict(
            batch=batch,
            disable_track=disable_track,
        )
        torch.cuda.synchronize()
        track_time = time.perf_counter() - t0
        return track_preds, track_time

