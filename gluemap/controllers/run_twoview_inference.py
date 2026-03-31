# import numpy as np
# import torch

# from pi3.models.pi3 import Pi3
# from vggt.models.vggt import VGGT

# from salad.vpr_model import VPRModel
# from e2esfm.utils.get_pi3_calibration import get_pi3d_calibration
# from e2esfm.utils.utils_mapanything import mapanything_inference, retrieve_mapanything_result

# from e2esfm.pipeline.run_star_base import StellarBase
# # from e2esfm.pipeline.run_ministar_vggt import MiniStellarVGGT
# from e2esfm.pipeline.tracks_util import *
# from vggt.utils.pose_enc import pose_encoding_to_extri_intri

# import faiss


from scipy.special import softmax
import torch
import numpy as np


class BatchInferenceDG:
    def __init__(self, model, store_dense=False, device="cuda", dtype=torch.bfloat16):

        self.model = model
        self.device = device
        self.dtype = dtype

        self.store_dense = store_dense

    def main(self, batch):
        images = batch["images"].to(self.device)

        view1 = {"img": images[:, 0], "instance": [i for i in range(images.shape[0])]}
        view2 = {"img": images[:, 1], "instance": [i for i in range(images.shape[0])]}

        res, pred1, pred2 = self.model(view1, view2, predict_extra=self.store_dense)

        if isinstance(pred1, list):
            pred1 = torch.stack(pred1, dim=0)
        else:
            pred1 = pred1

        if isinstance(pred2, list):
            pred2 = torch.stack(pred2, dim=0)
        else:
            pred2 = pred2

        score_s1 = softmax(pred1.detach().cpu().numpy(), axis=1)
        score_s2 = softmax(pred2.detach().cpu().numpy(), axis=1)
        vote_0 = (score_s1[:, 0] > score_s1[:, 1]).astype(int) + (
            score_s2[:, 0] > score_s2[:, 1]
        ).astype(int)
        vote_1 = (score_s1[:, 1] > score_s1[:, 0]).astype(int) + (
            score_s2[:, 1] > score_s2[:, 0]
        ).astype(int)
        index_max = vote_1 > vote_0
        index_min = vote_1 < vote_0
        index_equal = vote_1 == vote_0
        score = np.zeros_like(score_s1[:, 0])
        score[index_max] = np.max(
            (score_s1[index_max, 1], score_s2[index_max, 1]), axis=0
        )
        score[index_min] = np.min(
            (score_s1[index_min, 1], score_s2[index_min, 1]), axis=0
        )
        score[index_equal] = np.mean(
            (score_s1[index_equal, 1], score_s2[index_equal, 1]), axis=0
        )

        result_dict = {
            "scores": torch.from_numpy(score),
        }

        if self.store_dense:
            for i in range(4):
                for key in res[i].keys():
                    res[i][key] = res[i][key].cpu()
            # result_dict["res"] = res
            # a list of sublist, sublist has 4 dicts
            # each dict has keys 'pts3d', 'conf', 'desc', 'desc_conf'
            batch_size = images.shape[0]
            result_dict["res"] = [
                [
                    {
                        key: res[idx_tuple][key][idx_entry : idx_entry + 1]
                        for key in res[0].keys()
                    }
                    for idx_tuple in range(4)
                ]
                for idx_entry in range(batch_size)
            ]

        return result_dict
