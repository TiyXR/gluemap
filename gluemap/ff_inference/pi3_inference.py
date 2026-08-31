# Functions are adpated from https://github.com/yyfz/Pi3/issues/29
from collections import OrderedDict
from dataclasses import dataclass
from functools import partial

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import least_squares

from gluemap.ff_inference.local_inference import LocalInference


@dataclass(frozen=True)
class EncoderTokenKey:
    model_identity: str
    preprocessing_identity: str
    frame_uid: str
    image_height: int
    image_width: int
    dtype: str
    device: str


class Pi3EncoderTokenCache:
    """Bounded GPU-resident LRU for context-independent Pi3 frame tokens."""

    PREPROCESSING_IDENTITY = "pi3-imagenet-normalize/v1"

    def __init__(
        self,
        model: torch.nn.Module,
        capacity_frames: int,
        device: str | torch.device,
        dtype: torch.dtype,
    ) -> None:
        if capacity_frames < 1:
            raise ValueError(
                "Pi3 encoder token cache capacity must be positive"
            )
        self.model_identity = (
            f"{model.__class__.__module__}.{model.__class__.__qualname__}:"
            f"{id(model)}"
        )
        self.capacity_frames = capacity_frames
        self.device = str(device)
        self.dtype = str(dtype)
        self._entries: OrderedDict[EncoderTokenKey, torch.Tensor] = (
            OrderedDict()
        )
        self.requested_frame_count = 0
        self.hit_count = 0
        self.miss_count = 0
        self.encoded_frame_count = 0
        self.eviction_count = 0
        self.microbatch_reuse_count = 0
        self.peak_entry_count = 0
        self.peak_logical_bytes = 0

    def _key(
        self, frame_uid: str, image_height: int, image_width: int
    ) -> EncoderTokenKey:
        return EncoderTokenKey(
            model_identity=self.model_identity,
            preprocessing_identity=self.PREPROCESSING_IDENTITY,
            frame_uid=frame_uid,
            image_height=image_height,
            image_width=image_width,
            dtype=self.dtype,
            device=self.device,
        )

    def encode_context(
        self,
        model: torch.nn.Module,
        images: torch.Tensor,
        frame_uids: list[str] | list[list[str]],
    ):
        if images.ndim != 5:
            raise ValueError("Pi3 token cache requires B,N,C,H,W images")
        batch_size, frame_count = images.shape[:2]
        if frame_uids and isinstance(frame_uids[0], str):
            uid_rows = [list(frame_uids)]
        else:
            uid_rows = [list(row) for row in frame_uids]
        if len(uid_rows) != batch_size or any(
            len(row) != frame_count for row in uid_rows
        ):
            raise ValueError("Pi3 token cache frame UID shape differs")
        if frame_count > self.capacity_frames:
            raise ValueError("Pi3 context exceeds encoder token cache capacity")

        height, width = int(images.shape[-2]), int(images.shape[-1])
        key_rows = [
            [self._key(value, height, width) for value in row]
            for row in uid_rows
        ]
        context_tokens: list[list[torch.Tensor | None]] = [
            [None] * frame_count for _ in range(batch_size)
        ]
        missing: OrderedDict[EncoderTokenKey, tuple[int, int]] = OrderedDict()
        pending_positions: dict[EncoderTokenKey, list[tuple[int, int]]] = {}
        self.requested_frame_count += batch_size * frame_count
        for batch_index, keys in enumerate(key_rows):
            for frame_index, key in enumerate(keys):
                token = self._entries.pop(key, None)
                if token is not None:
                    self.hit_count += 1
                    self._entries[key] = token
                    context_tokens[batch_index][frame_index] = token
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
            ).unsqueeze(0)
            encoded = model.encode_frames(missing_images)
            if encoded.tokens.shape[1] != len(missing):
                raise ValueError(
                    "Pi3 encoder returned an unexpected frame count"
                )
            self.encoded_frame_count += len(missing)
            for encoded_position, key in enumerate(missing):
                token = encoded.tokens[
                    :, encoded_position : encoded_position + 1
                ]
                self._entries[key] = token
                for batch_index, frame_index in pending_positions[key]:
                    context_tokens[batch_index][frame_index] = token
                while len(self._entries) > self.capacity_frames:
                    self._entries.popitem(last=False)
                    self.eviction_count += 1

        if any(value is None for row in context_tokens for value in row):
            raise RuntimeError("Pi3 token cache failed to assemble a context")
        tokens = torch.cat(
            [torch.cat(row, dim=1) for row in context_tokens], dim=0
        )
        encoded_type = type(encoded) if missing else None
        if encoded_type is None:
            from pi3.models.pi3 import Pi3EncodedFrames

            encoded_type = Pi3EncodedFrames
        result = encoded_type(
            tokens=tokens,
            image_height=height,
            image_width=width,
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
        return {
            "contractId": "jarailsense.pi3-encoder-token-cache/v1",
            "capacityFrames": self.capacity_frames,
            "requestedFrameCount": requests,
            "cacheHitCount": self.hit_count,
            "cacheMissCount": self.miss_count,
            "microbatchReuseCount": self.microbatch_reuse_count,
            "encodedFrameCount": self.encoded_frame_count,
            "evictionCount": self.eviction_count,
            "residentFrameCount": len(self._entries),
            "peakResidentFrameCount": self.peak_entry_count,
            "peakResidentLogicalBytes": self.peak_logical_bytes,
            "hitRate": self.hit_count / requests if requests else 0.0,
            "preprocessingIdentity": self.PREPROCESSING_IDENTITY,
            "dtype": self.dtype,
            "device": self.device,
        }


class Pi3LocalInference(LocalInference):
    """Local inference for Pi3 / Pi3X backbones."""

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        *,
        encoder_token_cache_frames: int = 0,
    ) -> None:
        super().__init__(model, device, dtype)
        self.encoder_token_cache = (
            Pi3EncoderTokenCache(
                model, encoder_token_cache_frames, device, dtype
            )
            if encoder_token_cache_frames > 0
            and hasattr(model, "encode_frames")
            and hasattr(model, "decode_context")
            and hasattr(model, "decode_heads")
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
        raise ValueError("Pi3 token cache requires stable frame identities")

    def encoder_token_cache_report(self) -> dict | None:
        if self.encoder_token_cache is None:
            return None
        return self.encoder_token_cache.report()

    def predict(self, batch: dict) -> dict:
        """Run the Pi3/Pi3X backbone on a batch of images.

        Args:
            batch: Dict with ``"images"`` of shape ``(B, N, 3, H, W)``.

        Returns:
            The model's prediction dict augmented with calibration outputs.
            Keys include ``depth`` (shape ``(B, N, H, W, 1)``), ``depth_conf``
            (shape ``(B, N, H, W)``), ``extrinsics`` (shape ``(B, N, 4, 4)``),
            ``intrinsics`` (shape ``(B, N, 3, 3)``), as well as the raw
            ``local_points``, ``camera_poses`` and recovered ``shift``.
        """
        images = batch["images"].to(self.device).contiguous()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            if self.encoder_token_cache is None:
                predictions = self.model(images)
            else:
                frame_uids = self._frame_uids(batch, images.shape[1])
                encoded = self.encoder_token_cache.encode_context(
                    self.model, images, frame_uids
                )
                decoded = self.model.decode_context(encoded)
                predictions = self.model.decode_heads(decoded)

        # Rename depth confidence to avoid collision with tracker confidence
        if "conf" in predictions and "depth_conf" not in predictions:
            predictions["depth_conf"] = predictions.pop("conf")

        # Calibrate: sets depth, shift, depth_conf in predictions and
        # returns extrinsics/intrinsics
        extrinsics, intrinsics = self._calibrate(predictions)
        predictions["extrinsics"] = extrinsics
        predictions["intrinsics"] = intrinsics

        return predictions

    @staticmethod
    def _calibrate(result: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover camera calibration from Pi3's local point map.

        Consumes ``local_points`` and ``camera_poses`` from ``result``, recovers
        a per-frame focal length and Z-shift via
        :meth:`_recover_focal_shift`, builds the corresponding intrinsics, and
        derives extrinsics relative to the first view. Mutates ``result`` in
        place to add ``shift`` and a shifted ``depth`` tensor, and to squeeze
        ``depth_conf`` to ``(B, N, H, W)``.

        Args:
            result: Pi3 prediction dict containing ``local_points`` of shape
                ``(B, N, H, W, 3)``, ``depth_conf`` of shape
                ``(B, N, H, W, 1)`` and ``camera_poses`` of shape
                ``(B, N, 4, 4)``.

        Returns:
            ``(extrinsics, intrinsics)`` of shape ``(B, N, 4, 4)`` and
            ``(B, N, 3, 3)`` respectively.
        """
        points = result["local_points"]  # Shape: (B, N, H, W, 3)
        result["depth_conf"] = result["depth_conf"][..., 0]
        masks = torch.sigmoid(result["depth_conf"]) > 0.1  # Shape: (B, N, H, W)

        focal, shift = Pi3LocalInference._recover_focal_shift(
            points, masks, downsample_size=(64, 64)
        )

        # Calculate fx, fy from focal
        original_height, original_width = points.shape[-3:-1]
        aspect_ratio = original_width / original_height

        fx = (
            focal
            / 2
            * (1 + aspect_ratio**2) ** 0.5
            / aspect_ratio
            * original_width
        )
        fy = focal / 2 * (1 + aspect_ratio**2) ** 0.5 * original_height

        cx = original_width // 2
        cy = original_height // 2

        batch_shape = fx.shape
        intrinsics = torch.zeros(
            *batch_shape, 3, 3, device=fx.device, dtype=fx.dtype
        )
        intrinsics[..., 0, 0] = fx
        intrinsics[..., 1, 1] = fy
        intrinsics[..., 0, 2] = cx
        intrinsics[..., 1, 2] = cy
        intrinsics[..., 2, 2] = 1.0

        result["shift"] = shift

        extrinsics = (
            torch.linalg.inv(result["camera_poses"])
            @ result["camera_poses"][:, :1]
        )
        depth = Pi3LocalInference._get_shifted_depth_map(result)
        result["depth"] = depth.unsqueeze(-1)

        return extrinsics, intrinsics

    @staticmethod
    def _get_shifted_depth_map(result: dict) -> torch.Tensor:
        """Build the per-pixel depth map by adding the recovered Z-shift.

        Args:
            result: Pi3 prediction dict containing ``local_points`` of shape
                ``(B, N, H, W, 3)`` and ``shift`` of shape ``(B, N)``.

        Returns:
            Shifted depth tensor of shape ``(B, N, H, W)``.
        """
        # Get the depth map for the reference camera
        depth_map = result["local_points"][
            ..., 2
        ]  # Shape: (H, W) - extract Z component

        # shift the depth
        depth_map = depth_map + result["shift"][..., None, None]
        return depth_map

    @staticmethod
    def _recover_focal_shift(
        points: torch.Tensor,
        mask: torch.Tensor | None = None,
        focal: torch.Tensor | None = None,
        downsample_size: tuple[int, int] = (64, 64),
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Recover focal length and Z-shift from a point map.

        Assumes that the optical center is at the center of the map, that the
        map is undistorted, and that it is isometric in x and y.

        Args:
            points: Point map of shape ``(..., H, W, 3)``.
            mask: Optional boolean mask of shape ``(..., H, W)`` selecting
                valid pixels. If ``None``, all pixels are used.
            focal: Optional pre-known focal of shape ``(...)``. When provided
                only the shift is solved.
            downsample_size: ``(height, width)`` of the downsampled map used
                during optimization. Downsampling yields an approximate but
                efficient solution on large maps.

        Returns:
            ``(focal, shift)`` tensors of shape ``(...)``. ``focal`` is
            expressed relative to the half-diagonal of the map, and ``shift``
            is the Z-axis offset translating the point map into camera space.
        """

        shape = points.shape
        height, width = points.shape[-3], points.shape[-2]

        points = points.reshape(-1, *shape[-3:])
        mask = None if mask is None else mask.reshape(-1, *shape[-3:-1])
        focal = focal.reshape(-1) if focal is not None else None
        uv = Pi3LocalInference._normalized_view_plane_uv(
            width, height, dtype=points.dtype, device=points.device
        )  # (H, W, 2)

        points_lr = F.interpolate(
            points.permute(0, 3, 1, 2), downsample_size, mode="nearest"
        ).permute(0, 2, 3, 1)
        uv_lr = (
            F.interpolate(
                uv.unsqueeze(0).permute(0, 3, 1, 2),
                downsample_size,
                mode="nearest",
            )
            .squeeze(0)
            .permute(1, 2, 0)
        )
        mask_lr = (
            None
            if mask is None
            else F.interpolate(
                mask.to(torch.float32).unsqueeze(1),
                downsample_size,
                mode="nearest",
            ).squeeze(1)
            > 0
        )

        uv_lr_np = uv_lr.cpu().numpy()
        points_lr_np = points_lr.detach().cpu().numpy()
        focal_np = focal.cpu().numpy() if focal is not None else None
        mask_lr_np = None if mask is None else mask_lr.cpu().numpy()
        optim_shift, optim_focal = [], []
        for i in range(points.shape[0]):
            points_lr_i_np = (
                points_lr_np[i]
                if mask is None
                else points_lr_np[i][mask_lr_np[i]]
            )
            uv_lr_i_np = uv_lr_np if mask is None else uv_lr_np[mask_lr_np[i]]
            if uv_lr_i_np.shape[0] < 2:
                optim_focal.append(1)
                optim_shift.append(0)
                continue
            if focal is None:
                optim_shift_i, optim_focal_i = (
                    Pi3LocalInference._solve_optimal_focal_shift(
                        uv_lr_i_np, points_lr_i_np
                    )
                )
                optim_focal.append(float(optim_focal_i))
            else:
                optim_shift_i = Pi3LocalInference._solve_optimal_shift(
                    uv_lr_i_np, points_lr_i_np, focal_np[i]
                )
            optim_shift.append(float(optim_shift_i))
        optim_shift = torch.tensor(
            optim_shift, device=points.device, dtype=points.dtype
        ).reshape(shape[:-3])

        if focal is None:
            optim_focal = torch.tensor(
                optim_focal, device=points.device, dtype=points.dtype
            ).reshape(shape[:-3])
        else:
            optim_focal = focal.reshape(shape[:-3])

        return optim_focal, optim_shift

    @staticmethod
    def _normalized_view_plane_uv(
        width: int,
        height: int,
        aspect_ratio: float | None = None,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        """Build a normalized view-plane UV grid.

        The top-left corner maps to
        ``(-width / diagonal, -height / diagonal)`` and the bottom-right
        corner to ``(width / diagonal, height / diagonal)``.
        """
        if aspect_ratio is None:
            aspect_ratio = width / height

        span_x = aspect_ratio / (1 + aspect_ratio**2) ** 0.5
        span_y = 1 / (1 + aspect_ratio**2) ** 0.5

        u = torch.linspace(
            -span_x * (width - 1) / width,
            span_x * (width - 1) / width,
            width,
            dtype=dtype,
            device=device,
        )
        v = torch.linspace(
            -span_y * (height - 1) / height,
            span_y * (height - 1) / height,
            height,
            dtype=dtype,
            device=device,
        )
        u, v = torch.meshgrid(u, v, indexing="xy")
        return torch.stack([u, v], dim=-1)

    @staticmethod
    def _solve_optimal_focal_shift(
        uv: np.ndarray, xyz: np.ndarray
    ) -> tuple[np.floating, np.floating]:
        """Solve ``min |focal * xy / (z + shift) - uv|`` over shift and focal.

        Args:
            uv: Normalized image coordinates of shape ``(..., 2)``.
            xyz: Corresponding 3D points of shape ``(..., 3)``.

        Returns:
            ``(optim_shift, optim_focal)`` as ``np.float32`` scalars.
        """
        uv, xy, z = (
            uv.reshape(-1, 2),
            xyz[..., :2].reshape(-1, 2),
            xyz[..., 2].reshape(-1),
        )

        def fn(
            uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray
        ):
            xy_proj = xy / (z + shift)[:, None]
            f = (xy_proj * uv).sum() / np.square(xy_proj).sum()
            return (f * xy_proj - uv).ravel()

        solution = least_squares(
            partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm"
        )
        optim_shift = solution["x"].squeeze().astype(np.float32)

        xy_proj = xy / (z + optim_shift)[:, None]
        optim_focal = (xy_proj * uv).sum() / np.square(xy_proj).sum()

        return optim_shift, optim_focal

    @staticmethod
    def _solve_optimal_shift(
        uv: np.ndarray, xyz: np.ndarray, focal: float
    ) -> np.floating:
        """Solve ``min |focal * xy / (z + shift) - uv|`` for ``shift`` only.

        Args:
            uv: Normalized image coordinates of shape ``(..., 2)``.
            xyz: Corresponding 3D points of shape ``(..., 3)``.
            focal: Fixed focal length used during the fit.

        Returns:
            Optimal shift as an ``np.float32`` scalar.
        """
        uv, xy, z = (
            uv.reshape(-1, 2),
            xyz[..., :2].reshape(-1, 2),
            xyz[..., 2].reshape(-1),
        )

        def fn(
            uv: np.ndarray, xy: np.ndarray, z: np.ndarray, shift: np.ndarray
        ):
            xy_proj = xy / (z + shift)[:, None]
            return (focal * xy_proj - uv).ravel()

        solution = least_squares(
            partial(fn, uv, xy, z), x0=0, ftol=1e-3, method="lm"
        )
        return solution["x"].squeeze().astype(np.float32)
