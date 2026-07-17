# This file is part of VIAME, and is distributed under an OSI-approved        #
# BSD 3-Clause License. See top-level LICENSE.txt or                          #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.           #
"""Standalone RF-DETR training entrypoint for multi-GPU (DDP) runs.

The embedded viame_train_detector interpreter cannot launch DDP (which needs a
re-executable script with a ``__main__`` guard), so rf_detr_trainer.py spawns
this module as a subprocess when more than one GPU is available. PTL then
re-execs it once per rank. Parameters arrive as a JSON file (argv[1])."""
import os
import sys
import json


def build_and_train(params):
    import torch

    from viame.pytorch.utilities import (
        ensure_fork_start_method, ensure_rfdetr_compatibility, parse_resolution,
        resolution_is_set)

    # Python 3.14 defaults Linux to the forkserver start method, which cannot
    # pickle rfdetr's ChannelSubset transform and kills every DataLoader worker.
    # PTL re-execs this script per rank, so each rank runs this before building
    # its own dataloaders. No-op on Python <= 3.13 and off Linux.
    ensure_fork_start_method()

    # rf_detr_trainer.py applies this shim before importing rfdetr, but that is a
    # monkey-patch on the transformers module in ITS process. We are a fresh
    # subprocess (and PTL re-execs us again per rank), so the patch is not
    # inherited and "import rfdetr" would die on BackboneConfigMixin. Must run
    # before rfdetr is first imported here.
    ensure_rfdetr_compatibility()

    import rfdetr

    if params.get("segmentation"):
        sizes = {"nano": "RFDETRSegNano", "small": "RFDETRSegSmall",
                 "medium": "RFDETRSegMedium", "large": "RFDETRSegLarge"}
    else:
        sizes = {"nano": "RFDETRNano", "small": "RFDETRSmall",
                 "medium": "RFDETRMedium", "base": "RFDETRBase",
                 "large": "RFDETRLarge"}
    model_cls = getattr(rfdetr, sizes[params["model_size"]])

    model_kwargs = dict(num_channels=params["num_channels"], device="cuda")
    # Arrives as a string ("1280" or "960x1728") so a non-square pair survives JSON.
    resolution = parse_resolution(params.get("resolution", 0))
    if resolution_is_set(resolution):
        model_kwargs["resolution"] = resolution
    if params.get("gradient_checkpointing"):
        model_kwargs["gradient_checkpointing"] = True
    if params.get("keypoints") and params.get("keypoint_names"):
        model_kwargs["keypoint_head"] = True
        model_kwargs["num_keypoints"] = len(params["keypoint_names"])

    # Seed from a prior checkpoint by routing it through pretrain_weights. train()
    # rebuilds the network inside RFDETRModelModule from model_config and loads
    # only model_config.pretrain_weights, so a post-construction load_state_dict on
    # the wrapper would be silently discarded. load_pretrain_weights aligns
    # num_classes from the checkpoint/dataset.
    seed = params.get("seed_model") or ""
    if seed and os.path.exists(seed):
        model = model_cls(pretrain_weights=seed, **model_kwargs)
    else:
        model = model_cls(**model_kwargs)

    model.train(**params["train_kwargs"])


if __name__ == "__main__":
    # This script lives in viame/pytorch/, which contains helper packages named
    # torchvision, netharn and srnn. Python puts the script's own directory at
    # sys.path[0], so those would SHADOW the real PyPI packages (the local
    # torchvision has no .transforms) and break "import rfdetr". Drop the script
    # directory so torch/torchvision/rfdetr resolve to the installed packages.
    # This also runs on PyTorch-Lightning's per-rank re-execs of this script.
    _here = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [p for p in sys.path if p and os.path.abspath(p) != _here]
    build_and_train(json.load(open(sys.argv[1])))
