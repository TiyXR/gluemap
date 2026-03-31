from thirdparty.mast3r.model_dg import AsymmetricMASt3R
from thirdparty.pi3.models.pi3 import Pi3
from thirdparty.pi3.models.pi3x import Pi3X
from thirdparty.vggt.models.vggt import VGGT
from thirdparty.mapanything.models import MapAnything
from thirdparty.mapanything.models.mapanything.model_v1_1 import MapAnythingV1_1
from thirdparty.salad.vpr_model import VPRModel
from thirdparty.vggsfm.vggsfm_tracker import TrackerPredictor

import numpy as np

import torch
import omegaconf
from safetensors.torch import load_file


def load_models(args, keys=set()):
    chosen_model = getattr(args, "chosen_model", "pi3")
    keys.add(chosen_model)

    models = {}

    # Load the chosen multi-view model
    if chosen_model == "pi3" and chosen_model in keys:
        models["pi3"] = Pi3()
        models["pi3"].load_state_dict(load_file(args.path_pi3))
    elif chosen_model == "pi3x" and chosen_model in keys:
        models["pi3x"] = Pi3X(use_multimodal=False)
        models["pi3x"].load_state_dict(load_file(args.path_pi3x), strict=False)
    elif chosen_model == "vggt" and chosen_model in keys:
        models["vggt"] = VGGT()
        models["vggt"].load_state_dict(
            torch.load(args.path_vggt, map_location="cpu")
        )
    elif chosen_model == "map_anything" and chosen_model in keys:
        models["map_anything"] = MapAnything.from_pretrained(args.path_map_anything)
    elif chosen_model == "map_anything_v1.1" and chosen_model in keys:
        models["map_anything_v1.1"] = MapAnythingV1_1.from_pretrained(
            args.path_map_anything_v1_1
        )

    # Load Doppelganger++
    if "dg" in keys:
        models["dg"] = AsymmetricMASt3R(
            pos_embed="RoPE100",
            patch_embed_cls="ManyAR_PatchEmbed",
            img_size=(512, 512),
            head_type="catmlp+dpt",
            head_type_dg="transformer",
            output_mode="pts3d+desc24",
            output_mode_dg="dg_score",
            depth_mode=("exp", -np.inf, np.inf),
            conf_mode=("exp", 1, np.inf),
            enc_embed_dim=1024,
            enc_depth=24,
            enc_num_heads=16,
            dec_embed_dim=768,
            dec_depth=12,
            dec_num_heads=12,
            two_confs=True,
            desc_conf_mode=("exp", 0, np.inf),
            add_dg_pred_head=True,
            freeze=["mask", "encoder", "decoder", "head"],
        ).from_pretrained(args.path_dg)

    # Load VGG-SfM Tracker
    if "vggsfm" in keys:
        models["vggsfm"] = TrackerPredictor()
        models["vggsfm"].load_state_dict(
            torch.load(args.path_vggsfm_tracker, map_location="cpu")
        )

    # Load SALAD
    if "salad" in keys:
        models["salad"] = VPRModel(
            backbone_arch="dinov2_vitb14",
            backbone_config={
                "num_trainable_blocks": 4,
                "return_token": True,
                "norm_layer": True,
            },
            agg_arch="SALAD",
            agg_config={
                "num_channels": 768,
                "num_clusters": 64,
                "cluster_dim": 128,
                "token_dim": 256,
            },
        )

        models["salad"].load_state_dict(
            torch.load(args.path_salad, map_location="cpu"), strict=False
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    for model_name, model in models.items():
        model.eval()
        model.to(device)

    if args.distributed:
        for model_name, model in models.items():
            models[model_name] = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args.gpu],
                find_unused_parameters=True,
                static_graph=True,
            )

    # Create uniform alias for the chosen multi-view model
    if chosen_model in models:
        models["mv"] = models[chosen_model]

    return models, device


def load_all_models(args):
    """Load all pipeline models at once for reuse across datasets.

    Respects args.disable_tracking to skip VGGSfM if not needed.
    When direct_inference is enabled, only loads the backbone model.
    """
    chosen_model = getattr(args, "chosen_model", "pi3")
    if getattr(args, "direct_inference", False):
        keys = {chosen_model}
    else:
        keys = {"salad", "dg", chosen_model}
        if not getattr(args, "disable_tracking", False):
            keys.add("vggsfm")
    return load_models(args, keys=keys)
