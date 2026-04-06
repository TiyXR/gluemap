import torch

# from e2esfm.pipeline.tracks_util import *
from gluemap.utils.misc import get_tracks_dict_indexes
from gluemap.math.scaling import rescale_tracks, rescale_intrinsics, keep_inframes

# from gluemap.utils.tracks_util import keep_inframes, calculate
# from gluemap.math.

def restore_image_shape(predictions_dict, images_change, images_shape_ori, valid_threshold=0.05):
    index = get_tracks_dict_indexes(predictions_dict)
    for idx in index:
        # Keep the points in the imag
        # keep_inframes(predictions_dict, self.images_shape, [idx])
        
        # Rescale the tracks
        rescale_tracks(predictions_dict, images_change, [idx])
        
        # Keep the tracks in the frames
        keep_inframes(predictions_dict, images_shape_ori, [idx])
        
        # calculate_track_scores(predictions_dict, valid_threshold, indexes=[idx])
        
        
        # Rescale the intrinsics
        if "intrinsics" in predictions_dict:
            members = predictions_dict["indexes"][idx]
            scales_curr = [images_change[members[j]] for j in range(len(members))]
            intrinscs_curr = [predictions_dict["intrinsics"][idx][:, j] for j in range(len(members))]
            intrinscs_curr = torch.stack(rescale_intrinsics(intrinscs_curr, scales_curr), dim=1)
            predictions_dict["intrinsics"][idx] = intrinscs_curr.cpu()
            
    return predictions_dict

def restore_intrinsics(intrinsics, images_change, inverse=False):
    scales_curr = [images_change[j] for j in range(len(images_change))]
    intrinscs_curr = [intrinsics[:, j] for j in range(len(scales_curr))]
    intrinscs_curr = torch.stack(rescale_intrinsics(intrinscs_curr, scales_curr, inverse=inverse), dim=1)

    return intrinscs_curr