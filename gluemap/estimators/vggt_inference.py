import torch

from gluemap.estimators.local_inference import LocalInference


class VGGTLocalInference(LocalInference):
    """Local inference for VGGT backbone."""

    def predict(self, batch):
        images = batch["images"].to(self.device).contiguous()

        with torch.cuda.amp.autocast(dtype=self.dtype):
            predictions = self.model(images)

        # Extract extrinsics and intrinsics from pose encoding
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri

        extrinsics, intrinsics = pose_encoding_to_extri_intri(
            predictions["pose_enc"], image_size_hw=images.shape[-2:]
        )
        predictions["extrinsics"] = extrinsics
        predictions["intrinsics"] = intrinsics

        return predictions
