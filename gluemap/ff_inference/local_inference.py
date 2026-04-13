from abc import ABC, abstractmethod

import torch


class LocalInference(ABC):
    def __init__(self, model, device="cuda", dtype=torch.bfloat16):
        self.model = model
        self.device = device
        self.dtype = dtype

    @abstractmethod
    def predict(self, batch: dict) -> dict:
        """Run backbone model on a batch of images.

        Args:
            batch: dict with at minimum an "images" key of shape (B, N, 3, H, W).
                   Subclasses may use additional keys.

        Returns:
            dict with at minimum: depth, depth_conf, extrinsics, intrinsics.
        """
        ...


def create_local_inference(
    model, model_type, device="cuda", dtype=torch.bfloat16
):
    """Factory to create the appropriate LocalInference subclass."""
    if model_type in ("pi3", "pi3x"):
        from gluemap.ff_inference.pi3_inference import Pi3LocalInference

        return Pi3LocalInference(model, device, dtype)
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
