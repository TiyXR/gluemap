import torch

from gluemap.math.scaling import rescale_tracks_single, standardize_query_points


class TrackInference:
    """Point tracking using the VGGSfM tracker."""

    def __init__(self, model_track, device="cuda"):
        self.model_track = model_track
        self.device = device

    def predict(self, batch, disable_track=False):
        """Run VGGSfM tracker or produce dummy tracks.

        Args:
            batch: dict with keys images (or images_1024), query_points,
                   indexes, images_change_1024, images_change.
            disable_track: if True, create dummy tracks from query_points.

        Returns:
            dict with keys: "track", "vis", "conf".
        """
        query_points = batch["query_points"].to(self.device)
        num_frames = batch["images"].shape[1]

        if disable_track:
            track = query_points.unsqueeze(1).expand(-1, num_frames, -1, -1)
            return {
                "track": track,
                "vis": torch.ones_like(track[..., 0]),
                "conf": torch.ones_like(track[..., 0]),
            }

        images_1024 = batch["images_1024"].to(self.device)

        tracks_all = []
        vis_all = []
        scores_all = []

        for i in range(images_1024.shape[0]):
            fine_pred_track, _, pred_vis, pred_score = self.model_track(
                images_1024[i : i + 1], query_points[i : i + 1]
            )
            tracks_all.append(fine_pred_track)
            vis_all.append(pred_vis)
            scores_all.append(pred_score)

        track = torch.cat(tracks_all, dim=0)
        vis = torch.cat(vis_all, dim=0)
        conf = torch.cat(scores_all, dim=0)

        # Rescale tracks from 1024 space to original image coordinates
        indexes = batch["indexes"]
        images_change_1024 = batch["images_change_1024"]
        images_change = batch["images_change"]

        for i in range(indexes.shape[0]):
            for j, idx_inner in enumerate(indexes[i].tolist()):
                track[i : i + 1, j : j + 1] = standardize_query_points(
                    rescale_tracks_single(
                        track[i : i + 1, j : j + 1],
                        images_change_1024[i][j],
                    ),
                    images_change[i][j],
                )

        return {"track": track, "vis": vis, "conf": conf}
