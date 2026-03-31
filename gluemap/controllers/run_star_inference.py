import os
import time
import numpy as np
import torch
from gluemap.math.scaling import rescale_tracks_single, standardize_query_points
from gluemap.math.verification import (
    verify_by_reprojection_n2,
    convert_from_depth_to_world_points,
)
from gluemap.math.virtual_tracks import (
    calculate_virtual_tracks,
    extract_cam_points_depth,
)
from gluemap.utils.pi3_utils import get_pi3d_calibration
from gluemap.utils.mapanything_utils import mapanything_inference


class BatchInferenceStar:
    def __init__(
        self,
        model,
        model_type="pi3",
        model_track=None,
        device="cuda",
        dtype=torch.bfloat16,
        pointmap_dir=None,
        repredict_verified=False,
        repredict_threshold=0.05,
    ):
        self.model = model
        self.model_type = model_type
        self.model_track = model_track
        self.device = device
        self.dtype = dtype
        self.pointmap_dir = pointmap_dir

        self.repredict_verified = repredict_verified
        self.repredict_threshold = repredict_threshold
        self.include_track = True
        self.check_consistency = False

    def main(self, batch, disable_track=False, include_track=True):
        if self.repredict_verified:
            return self._main_repredict(batch, include_track=include_track)

        predictions, images, forward_time, track_time = self._predict_images(
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
            images.shape[-2:],
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

    def _main_repredict(self, batch, include_track=True):
        # Round 1: full prediction with tracks
        predictions, images, forward_time, track_time = self._predict_images(
            batch, disable_track=False, include_track=include_track
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
            images.shape[-2:],
            batch["indexes"],
            batch["images_change"],
            batch["images_shape_ori"],
        )

        # Select verified views (always include center view at index 0)
        verified_mask = scores[0] > self.repredict_threshold
        verified_mask[0] = True
        verified_indices = torch.where(verified_mask)[0]
        num_views = scores.shape[1]

        # If all views pass or only center survives, return round 1 as-is
        if verified_indices.numel() >= num_views or verified_indices.numel() < 2:
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

        # Round 2: re-predict geometry with only verified views (skip tracks)
        vi = verified_indices
        subset_batch = {
            "images": batch["images"][:, vi],
            "indexes": batch["indexes"][:, vi],
            "images_change": batch["images_change"][:, vi],
            "images_shape_ori": batch["images_shape_ori"][:, vi],
        }

        predictions2, images2, forward_time2, _ = self._predict_images(
            subset_batch, disable_track=True, include_track=False
        )
        forward_time += forward_time2

        (
            extrinsics2,
            intrinsics2,
            scores2,
            tracks_virtual2,
            points3d_virtual2,
            valid_virtual2,
            cam_points2,
            cam_points_conf2,
            depth_transformed2,
        ) = self._covisiblity_extraction(
            predictions2,
            images2.shape[-2:],
            subset_batch["indexes"],
            subset_batch["images_change"],
            subset_batch["images_shape_ori"],
        )

        result_dict = {
            "indexes": subset_batch["indexes"][0].tolist(),
            "extrinsics": extrinsics2,
            "intrinsics": intrinsics2,
            "pose_scores": scores2,
            "tracks_virtual": tracks_virtual2,
            "points3d_virtual": points3d_virtual2,
            "valid_virtual": valid_virtual2,
            "_forward_time": forward_time,
            "_track_time": track_time,
        }

        # Subsample tracks from round 1 to match verified views
        if include_track:
            result_dict["tracks"] = predictions["track"][:, vi].cpu()
            result_dict["vis"] = predictions["vis"][:, vi].cpu()
            result_dict["conf"] = predictions["conf"][:, vi].cpu()

        return result_dict

    @torch.no_grad()
    def _predict_images(self, batch, disable_track=False, include_track=True):
        images = batch["images"].to(self.device).contiguous()
        if include_track:
            query_points = batch["query_points"].to(self.device)
        else:
            query_points = None

        if not disable_track and not include_track:
            raise ValueError("if not disable_track, include_track should be True")

        torch.cuda.synchronize()
        t_forward_start = time.perf_counter()

        if disable_track or not self.model_track is None:
            if self.model_type in ("map_anything", "map_anything_v1.1"):
                predictions = mapanything_inference(
                    self.model, images, device=self.device, dtype=self.dtype
                )
            else:
                with torch.cuda.amp.autocast(dtype=self.dtype):
                    predictions = self.model(images)
        else:
            with torch.cuda.amp.autocast(dtype=self.dtype):
                predictions = self.model(images, query_points)

        torch.cuda.synchronize()
        t_forward_end = time.perf_counter()
        forward_time = t_forward_end - t_forward_start

        # Rename depth confidence to avoid collision with tracker confidence
        # (MapAnything already returns "depth_conf" directly)
        if "conf" in predictions and "depth_conf" not in predictions:
            predictions["depth_conf"] = predictions.pop("conf")

        track_time = 0.0
        if disable_track:
            if include_track:
                # Use the query points as track. Since we are not doing the BA, it is okay that the correspondences are incorrect
                predictions["track"] = query_points.unsqueeze(1).expand(-1, images.shape[1], -1, -1)
                predictions["vis"] = torch.ones_like(
                    predictions["track"][..., 0:1]
                ).squeeze(-1)
                predictions["conf"] = torch.ones_like(
                    predictions["track"][..., 0:1]
                ).squeeze(-1)
        else:
            images_1024 = batch["images_1024"].to(self.device)

            tracks_all = []
            vis_all = []
            scores_all = []

            torch.cuda.synchronize()
            t_track_start = time.perf_counter()

            for i in range(images_1024.shape[0]):
                fine_pred_track, _, pred_vis, pred_score = self.model_track(
                    images_1024[i : i + 1], query_points[i : i + 1]
                )

                tracks_all.append(fine_pred_track)
                vis_all.append(pred_vis)
                scores_all.append(pred_score)

            torch.cuda.synchronize()
            t_track_end = time.perf_counter()
            track_time = t_track_end - t_track_start

            # Concatenate the results
            fine_pred_track = torch.cat(tracks_all, dim=0)
            pred_vis = torch.cat(vis_all, dim=0)
            pred_score = torch.cat(scores_all, dim=0)

            predictions["track"] = fine_pred_track
            predictions["vis"] = pred_vis
            predictions["conf"] = pred_score

            indexes = batch["indexes"]

            for i in range(indexes.shape[0]):
                for j, idx_inner in enumerate(indexes[i].tolist()):
                    predictions["track"][i : i + 1, j : j + 1] = (
                        standardize_query_points(
                            rescale_tracks_single(
                                predictions["track"][i : i + 1, j : j + 1],
                                batch["images_change_1024"][i][j],
                            ),
                            batch["images_change"][i][j],
                        )
                    )

        return predictions, images, forward_time, track_time

    # TODO: implement this function
    def _covisiblity_extraction(
        self, predictions, image_size_hw, indexes, images_change, images_shape_ori
    ):
        # Extract extrinsics and intrinsics based on model type
        if self.model_type == "vggt":
            from thirdparty.vggt.utils.pose_enc import pose_encoding_to_extri_intri
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                predictions["pose_enc"], image_size_hw=image_size_hw
            )
        elif self.model_type in ("pi3", "pi3x"):
            extrinsics, intrinsics = get_pi3d_calibration(predictions)
        elif self.model_type in ("map_anything", "map_anything_v1.1"):
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
