import time

import torch

from gluemap.estimators.local_inference import create_local_inference
from gluemap.estimators.track_inference import TrackInference
from gluemap.math.scaling import rescale_tracks_single
from gluemap.math.verification import (
    verify_by_reprojection_n2,
    convert_from_depth_to_world_points,
)
from gluemap.math.virtual_tracks import (
    calculate_virtual_tracks,
    extract_cam_points_depth,
)


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

        self.include_track = True
        self.check_consistency = False

        self.local_inference = create_local_inference(model, model_type, device, dtype)
        self.track_inference = TrackInference(model_track, device)

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
        ) = self._covisiblity_extraction(
            predictions,
            batch["indexes"],
            batch["images_change"],
            batch["images_shape_ori"],
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

    def _covisiblity_extraction(
        self, predictions, indexes, images_change, images_shape_ori
    ):
        extrinsics = predictions["extrinsics"]
        intrinsics = predictions["intrinsics"]

        depth_transformed = convert_from_depth_to_world_points(
            predictions["depth"], extrinsics, intrinsics
        )

        if self.check_consistency:
            scores, valid_mask = verify_by_reprojection_n2(
                depth_transformed,
                extrinsics,
                intrinsics,
                conf=predictions["depth_conf"],
            )
        else:
            scores, valid_mask = verify_by_reprojection_n2(
                depth_transformed, extrinsics, intrinsics
            )

        tracks_virtual, points3d_virtual, isnegative_virtual, valid_virtual = (
            calculate_virtual_tracks(
                predictions["depth"], extrinsics, intrinsics, valid_mask
            )
        )

        if "track" in predictions and self.include_track:
            cam_points, cam_points_conf = extract_cam_points_depth(
                intrinsics, predictions
            )

            for i in range(indexes.shape[0]):
                for j, idx_inner in enumerate(indexes[i].tolist()):
                    tracks_virtual[i, j] = rescale_tracks_single(
                        tracks_virtual[i, j], images_change[i][j]
                    )

            return (
                extrinsics.cpu(),
                intrinsics.cpu(),
                scores.cpu(),
                tracks_virtual.cpu(),
                points3d_virtual.cpu(),
                valid_virtual.cpu(),
                cam_points.cpu(),
                cam_points_conf.cpu(),
                depth_transformed.cpu(),
            )
        else:
            return (
                extrinsics.cpu(),
                intrinsics.cpu(),
                scores.cpu(),
                tracks_virtual.cpu(),
                points3d_virtual.cpu(),
                valid_virtual.cpu(),
                None,
                None,
                depth_transformed.cpu(),
            )
