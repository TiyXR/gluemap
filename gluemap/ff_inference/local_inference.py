from abc import ABC, abstractmethod

import torch


class LocalInference(ABC):
    """Abstract wrapper around a multi-view feed-forward backbone.

    Subclasses adapt a specific backbone (Pi3/Pi3X, VGGT, MapAnything) to a
    shared :meth:`predict` interface returning depth, confidence, extrinsics,
    and intrinsics.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def predict(self, batch: dict) -> dict:
        """Run backbone model on a batch of images.

        Args:
            batch: dict with at minimum an "images" key of shape
                   (B, N, 3, H, W). Subclasses may use additional keys.

        Returns:
            dict with at minimum: depth, depth_conf, extrinsics, intrinsics.
        """
        ...


def create_local_inference(
    model: torch.nn.Module,
    model_type: str,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    encoder_token_cache_frames: int = 0,
) -> LocalInference:
    """Factory to create the appropriate :class:`LocalInference` subclass.

    Args:
        model: Loaded backbone network whose architecture matches
            ``model_type``.
        model_type: One of ``"pi3"``, ``"pi3x"``, ``"vggt"``,
            ``"map_anything"``.
        device: Device on which inputs are placed before the backbone runs.
        dtype: Autocast dtype used during the backbone forward pass.

    Returns:
        A concrete :class:`LocalInference` subclass wrapping ``model``.

    Raises:
        ValueError: If ``model_type`` is not one of the supported values.
    """
    if model_type in ("pi3", "pi3x"):
        from gluemap.ff_inference.pi3_inference import Pi3LocalInference

        return Pi3LocalInference(
            model,
            device,
            dtype,
            encoder_token_cache_frames=encoder_token_cache_frames,
        )
    elif model_type == "vggt":
        from gluemap.ff_inference.vggt_inference import VGGTLocalInference

        return VGGTLocalInference(model, device, dtype)
    elif model_type == "map_anything":
        from gluemap.ff_inference.mapanything_inference import (
            MapAnythingLocalInference,
        )

        return MapAnythingLocalInference(model, device, dtype)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
