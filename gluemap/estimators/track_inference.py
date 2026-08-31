from collections import OrderedDict
from dataclasses import dataclass

import torch

from gluemap.math.scaling import rescale_tracks_single, standardize_query_points


@dataclass(frozen=True)
class TrackerFeatureKey:
    model_identity: str
    preprocessing_identity: str
    frame_uid: str
    image_height: int
    image_width: int
    dtype: str
    device: str


class TrackerCoarseFeatureCache:
    """Bounded GPU LRU for VGGSfM's context-independent coarse fmaps."""

    PREPROCESSING_IDENTITY = "vggsfm-coarse-downsample-bilinear/v1"

    def __init__(
        self,
        model: torch.nn.Module,
        capacity_frames: int,
        device: str | torch.device,
    ) -> None:
        if capacity_frames < 1:
            raise ValueError("tracker feature cache capacity must be positive")
        self.model_identity = (
            f"{model.__class__.__module__}.{model.__class__.__qualname__}:"
            f"{id(model)}"
        )
        self.capacity_frames = capacity_frames
        self.device = str(device)
        self._entries: OrderedDict[TrackerFeatureKey, torch.Tensor] = (
            OrderedDict()
        )
        self.requested_frame_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.extracted_frame_count = 0
        self.eviction_count = 0
        self.microbatch_reuse_count = 0
        self.peak_entry_count = 0
        self.peak_logical_bytes = 0

    def _key(
        self,
        frame_uid: str,
        image_height: int,
        image_width: int,
        dtype: torch.dtype,
    ) -> TrackerFeatureKey:
        return TrackerFeatureKey(
            model_identity=self.model_identity,
            preprocessing_identity=self.PREPROCESSING_IDENTITY,
            frame_uid=frame_uid,
            image_height=image_height,
            image_width=image_width,
            dtype=str(dtype),
            device=self.device,
        )

    def features(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        frame_uids: list[str] | list[list[str]],
    ) -> torch.Tensor:
        if images.ndim != 5:
            raise ValueError("tracker feature cache requires B,N,C,H,W images")
        batch_size, frame_count = images.shape[:2]
        if frame_uids and isinstance(frame_uids[0], str):
            uid_rows = [list(frame_uids)]
        else:
            uid_rows = [list(row) for row in frame_uids]
        if len(uid_rows) != batch_size or any(
            len(row) != frame_count for row in uid_rows
        ):
            raise ValueError("tracker feature cache frame UID shape differs")
        if frame_count > self.capacity_frames:
            raise ValueError("tracker context exceeds feature cache capacity")

        height, width = int(images.shape[-2]), int(images.shape[-1])
        key_rows = [
            [self._key(value, height, width, images.dtype) for value in row]
            for row in uid_rows
        ]
        context_features: list[list[torch.Tensor | None]] = [
            [None] * frame_count for _ in range(batch_size)
        ]
        missing: OrderedDict[TrackerFeatureKey, tuple[int, int]] = OrderedDict()
        pending_positions: dict[TrackerFeatureKey, list[tuple[int, int]]] = {}
        self.requested_frame_count += batch_size * frame_count
        for batch_index, keys in enumerate(key_rows):
            for frame_index, key in enumerate(keys):
                feature = self._entries.pop(key, None)
                if feature is not None:
                    self.hit_count += 1
                    self._entries[key] = feature
                    context_features[batch_index][frame_index] = feature
                elif key in missing:
                    self.microbatch_reuse_count += 1
                    pending_positions[key].append((batch_index, frame_index))
                else:
                    self.miss_count += 1
                    missing[key] = (batch_index, frame_index)
                    pending_positions[key] = [(batch_index, frame_index)]

        if missing:
            missing_images = torch.stack(
                [
                    images[batch_index, frame_index]
                    for batch_index, frame_index in missing.values()
                ],
                dim=0,
            )
            extracted = model.process_images_to_fmaps(missing_images)
            if extracted.shape[0] != len(missing):
                raise ValueError("tracker returned an unexpected fmap count")
            self.extracted_frame_count += len(missing)
            for extracted_position, key in enumerate(missing):
                feature = extracted[extracted_position : extracted_position + 1]
                self._entries[key] = feature
                for batch_index, frame_index in pending_positions[key]:
                    context_features[batch_index][frame_index] = feature
                while len(self._entries) > self.capacity_frames:
                    self._entries.popitem(last=False)
                    self.eviction_count += 1

        if any(value is None for row in context_features for value in row):
            raise RuntimeError(
                "tracker feature cache failed to assemble a context"
            )
        result = torch.cat(
            [torch.cat(row, dim=0).unsqueeze(0) for row in context_features],
            dim=0,
        )
        self.peak_entry_count = max(self.peak_entry_count, len(self._entries))
        logical_bytes = sum(
            value.numel() * value.element_size()
            for value in self._entries.values()
        )
        self.peak_logical_bytes = max(self.peak_logical_bytes, logical_bytes)
        return result

    def report(self) -> dict[str, int | float | str]:
        requests = self.requested_frame_count
        dtype = next((key.dtype for key in self._entries), "unknown")
        return {
            "contractId": "jarailsense.vggsfm-coarse-feature-cache/v1",
            "capacityFrames": self.capacity_frames,
            "requestedFrameCount": requests,
            "cacheHitCount": self.hit_count,
            "cacheMissCount": self.miss_count,
            "microbatchReuseCount": self.microbatch_reuse_count,
            "extractedFrameCount": self.extracted_frame_count,
            "evictionCount": self.eviction_count,
            "residentFrameCount": len(self._entries),
            "peakResidentFrameCount": self.peak_entry_count,
            "peakResidentLogicalBytes": self.peak_logical_bytes,
            "hitRate": self.hit_count / requests if requests else 0.0,
            "preprocessingIdentity": self.PREPROCESSING_IDENTITY,
            "dtype": dtype,
            "device": self.device,
        }


class TrackInference:
    """Point tracking using the VGGSfM tracker."""

    def __init__(
        self,
        model_track: torch.nn.Module,
        device: str = "cuda",
        feature_cache_frames: int = 0,
    ) -> None:
        self.model_track = model_track
        self.device = device
        self.feature_cache = (
            TrackerCoarseFeatureCache(model_track, feature_cache_frames, device)
            if feature_cache_frames > 0
            and hasattr(model_track, "process_images_to_fmaps")
            else None
        )

    @staticmethod
    def _frame_uids(
        batch: dict, frame_count: int
    ) -> list[str] | list[list[str]]:
        images = batch.get("images")
        indexes = batch.get("indexes")
        batch_size = (
            int(images.shape[0])
            if isinstance(images, torch.Tensor)
            else int(indexes.shape[0])
            if isinstance(indexes, torch.Tensor) and indexes.ndim == 2
            else 1
        )
        raw = batch.get("frame_uids")
        if isinstance(raw, (list, tuple)) and len(raw) == frame_count:
            if all(isinstance(value, str) for value in raw) and batch_size == 1:
                return list(raw)
            if all(
                isinstance(value, (list, tuple))
                and len(value) == batch_size
                and all(isinstance(item, str) for item in value)
                for value in raw
            ):
                rows = [
                    [str(raw[frame][batch]) for frame in range(frame_count)]
                    for batch in range(batch_size)
                ]
                return rows[0] if batch_size == 1 else rows
        if isinstance(indexes, torch.Tensor) and indexes.shape == (
            batch_size,
            frame_count,
        ):
            rows = [
                [f"geometry-{int(value)}" for value in row]
                for row in indexes.tolist()
            ]
            return rows[0] if batch_size == 1 else rows
        raise ValueError(
            "tracker feature cache requires stable frame identities"
        )

    def feature_cache_report(self) -> dict | None:
        if self.feature_cache is None:
            return None
        return self.feature_cache.report()

    def predict(
        self,
        batch: dict[str, torch.Tensor],
        use_dummy_tracks: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Run VGGSfM tracker or produce dummy tracks.

        Args:
            batch: dict with keys images (or images_1024), query_points,
                   indexes, images_change_1024, images_change.
            use_dummy_tracks: if True, create dummy tracks from query_points.

        Returns:
            dict with keys: "track", "vis", "conf".
        """
        query_points = batch["query_points"].to(self.device)
        num_frames = batch["images"].shape[1]

        if use_dummy_tracks:
            track = query_points.unsqueeze(1).expand(-1, num_frames, -1, -1)
            return {
                "track": track,
                "vis": torch.ones_like(track[..., 0]),
                "conf": torch.ones_like(track[..., 0]),
            }

        images_1024 = batch["images_1024"].to(self.device)
        fmaps = None
        if self.feature_cache is not None:
            frame_uids = self._frame_uids(batch, images_1024.shape[1])
            fmaps = self.feature_cache.features(
                self.model_track, images_1024, frame_uids
            )

        tracks_all = []
        vis_all = []
        scores_all = []

        for i in range(images_1024.shape[0]):
            fine_pred_track, _, pred_vis, pred_score = self.model_track(
                images_1024[i : i + 1],
                query_points[i : i + 1],
                fmaps=None if fmaps is None else fmaps[i : i + 1],
                inference=False,
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
            for j, _idx_inner in enumerate(indexes[i].tolist()):
                track[i : i + 1, j : j + 1] = standardize_query_points(
                    rescale_tracks_single(
                        track[i : i + 1, j : j + 1],
                        images_change_1024[i][j],
                    ),
                    images_change[i][j],
                )

        return {"track": track, "vis": vis, "conf": conf}
