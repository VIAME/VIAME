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
    if params.get("resolution", 0) > 0:
        model_kwargs["resolution"] = params["resolution"]
    if params.get("gradient_checkpointing"):
        model_kwargs["gradient_checkpointing"] = True

    seed = params.get("seed_model") or ""
    if seed and os.path.exists(seed):
        ckpt = torch.load(seed, map_location="cpu", weights_only=False)
        args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
        args = args if isinstance(args, dict) else vars(args)
        num_classes = args.get("num_classes",
                               len(params.get("class_names") or []) or 90)
        model = model_cls(pretrain_weights=None, num_classes=num_classes,
                          **model_kwargs)
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.model.model.load_state_dict(ckpt["model"])
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
