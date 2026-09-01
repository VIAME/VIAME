# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Convert a netharn ``ClfModel`` classifier (``*.zip`` deploy file) to ONNX.

Companion to :mod:`viame.pytorch.netharn_mmdet_to_onnx`, which handles the
netharn *detectors*. These are the whole-frame classifiers behind the kwiver
``netharn_classifier`` algorithm and the chip reclassifiers behind
``refine_detections :refiner:type netharn`` -- in the HabCam add-on, the 22
substrate classifiers and the two scallop EfficientNet reclassifiers.

A ``ClfModel`` is far simpler than the R-CNNs: an ``InputNorm`` layer followed
by a stock torchvision backbone (resnet50 / resnext101 / efficientnetv2s /
efficientnetv2m), returning raw class logits. There is no NMS, no RoIAlign and
no custom autograd Function, so the ordinary export path works -- none of the
workarounds ``netharn_mmdet_to_onnx`` needs apply here.

Because the ``InputNorm`` is a layer *inside* the model, the exported graph
performs the mean/std normalization itself and expects RGB in ``[0, 1]``. The
sidecar therefore records ``scale = 1/255`` with an identity mean/std.

Output contract
---------------
Input ``input`` of shape ``(N, 3, H, W)`` with N dynamic; output ``class_probs``
of shape ``(N, C)``, softmax already applied, with column order matching
``meta.category_names``.

Softmax is baked in deliberately. netharn decodes with
``CategoryTree.hierarchical_softmax``, which for a flat (single ``idx_groups``)
tree -- every classifier shipped to date -- is exactly a plain softmax over all
classes. Models with a real class hierarchy are refused rather than silently
given the wrong normalization.

CLI:
    python -m viame.pytorch.netharn_clf_to_onnx \
        --deployed=models/scallop_efficientnet_four_class.zip \
        --output=models/scallop_efficientnet_four_class.onnx
"""
from __future__ import annotations

import json
import os
import re
import zipfile

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
def load_netharn_clf_model(deployed_fpath):
    """Load a netharn ClfModel deploy zip, tolerating the retired depr-mmdet
    packages (netharn's package import pulls mmdet in even for classifiers)."""
    from viame.pytorch.netharn.detect_predict import (
        patch_numpy, setup_module_aliases)
    from torch_liberator.deployer import DeployedModel

    setup_module_aliases()
    patch_numpy()
    model = DeployedModel(deployed_fpath).load_model()
    model.eval()
    return model


def read_input_dims(deployed_fpath):
    """(height, width) the classifier was trained at, from its train_info.json.

    Mirrors ``ClfPredictor._infer_native``: prefer the structured ``other``
    block, fall back to the repr'd config string, else None.
    """
    if not str(deployed_fpath).endswith('.zip'):
        return None
    with zipfile.ZipFile(deployed_fpath) as zf:
        names = [n for n in zf.namelist() if n.endswith('train_info.json')]
        if len(names) != 1:
            return None
        info = json.loads(zf.read(names[0]).decode('utf-8'))

    dims = info.get('other', {}).get('input_dims')
    if isinstance(dims, (list, tuple)) and len(dims) == 2:
        return (int(dims[0]), int(dims[1]))

    text = info.get('extra', {}).get('config', '')
    match = re.search(r"'input_dims':\s*\[\s*(\d+),\s*(\d+)", text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# ---------------------------------------------------------------------------
class _ClfOnnxWrapper(nn.Module):
    """Tensor in, probability tensor out.

    ``ClfModel.forward`` returns ``{'class_energy': logits}``; a dict output
    cannot be traced, so unwrap it and apply the softmax that netharn's
    ``ClfCoder`` would have applied host-side.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image):
        outputs = self.model(image)
        logits = outputs['class_energy'] if isinstance(outputs, dict) else outputs
        return torch.softmax(logits, dim=1)


def _check_flat_hierarchy(model):
    """Refuse models whose class tree would make plain softmax wrong."""
    groups = getattr(model.classes, 'idx_groups', None)
    if groups is not None and len(groups) > 1:
        raise NotImplementedError(
            'this classifier has a hierarchical class tree '
            f'(idx_groups={groups}); a flat softmax would not reproduce '
            'CategoryTree.hierarchical_softmax, so the graph would be wrong')


def build_modelspec(model, input_dims, source):
    # The InputNorm layer is inside the graph, so the caller only rescales to
    # [0, 1]; folding mean/std into the sidecar too would apply them twice.
    return {
        'input': {
            'shape_hw': [int(input_dims[0]), int(input_dims[1])],
            'layout': 'NCHW',
            'dynamic_batch': True,
        },
        'preprocess': {
            'scale': 1.0 / 255.0,
            'normalize_mean': [0.0, 0.0, 0.0],
            'normalize_std': [1.0, 1.0, 1.0],
            # netharn's dataset resizes with kwimage.imresize(letterbox=True),
            # i.e. aspect preserved and padded -- not a squash resize.
            'resize_mode': 'letterbox',
            'interpolation': 'linear',
            'channel_order': 'rgb',
        },
        'postprocess': {
            'decoder': 'classifier',
            'softmax_applied': True,
        },
        'meta': {
            'category_names': list(model.classes),
            'arch': getattr(model, 'arch', None),
            'task': 'classification',
            'source_model': os.path.basename(str(source)),
        },
    }


# ---------------------------------------------------------------------------
def netharn_clf_to_onnx(deployed_fpath,
                        output_fpath,
                        input_dims=None,
                        opset_version=17,
                        dynamic_batch=True,
                        write_modelspec=True):
    """Export a netharn ClfModel zip to ONNX + ``.modelspec.json``.

    Args:
        deployed_fpath (str): the netharn ``*.zip`` deploy file.
        output_fpath (str): where to write the ``.onnx``.
        input_dims (tuple): (height, width). Defaults to the training input
            size recorded in the zip, else 224x224.
        opset_version (int): 17 is plenty for these backbones; unlike the
            detectors there is no standard-op floor to respect.
        dynamic_batch (bool): leave the batch axis dynamic so callers can batch
            chips the way netharn's predictor does. Turn off for a fixed-batch
            graph if a backend needs static shapes.

    Returns:
        str: path to the written .onnx file.
    """
    model = load_netharn_clf_model(deployed_fpath)
    _check_flat_hierarchy(model)

    if input_dims is None:
        input_dims = read_input_dims(deployed_fpath) or (224, 224)
    height, width = int(input_dims[0]), int(input_dims[1])

    wrapper = _ClfOnnxWrapper(model).eval()
    dummy = torch.rand(1, 3, height, width)

    out_dpath = os.path.dirname(os.path.abspath(output_fpath))
    if out_dpath:
        os.makedirs(out_dpath, exist_ok=True)

    dynamic_axes = ({'input': {0: 'batch'}, 'class_probs': {0: 'batch'}}
                    if dynamic_batch else None)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), output_fpath,
            input_names=['input'], output_names=['class_probs'],
            export_params=True, do_constant_folding=True, verbose=False,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            # TorchScript exporter: it honors dynamic_axes, which the dynamo
            # exporter does not for these graphs (same reason as the stereo
            # models in plugins/onnx).
            dynamo=False)

    if write_modelspec:
        spec = build_modelspec(model, (height, width), deployed_fpath)
        spec_fpath = os.path.splitext(output_fpath)[0] + '.modelspec.json'
        with open(spec_fpath, 'w') as file:
            json.dump(spec, file, indent=2)

    return output_fpath


# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert a netharn ClfModel classifier to ONNX')
    parser.add_argument('--deployed', required=True,
                        help='input netharn .zip deploy file')
    parser.add_argument('--output', required=True, help='output .onnx path')
    parser.add_argument('--input-dims', default=None,
                        help='HxW (or a single N for NxN) network input size; '
                             'defaults to the training input size')
    parser.add_argument('--opset', type=int, default=17)
    parser.add_argument('--static-batch', action='store_true',
                        help='export a fixed batch-1 graph instead of leaving '
                             'the batch axis dynamic')
    args = parser.parse_args(argv)

    input_dims = None
    if args.input_dims:
        parts = [int(v) for v in args.input_dims.replace('x', ',').split(',')]
        input_dims = (parts[0], parts[-1])

    out = netharn_clf_to_onnx(
        args.deployed, args.output, input_dims=input_dims,
        opset_version=args.opset, dynamic_batch=not args.static_batch)
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
