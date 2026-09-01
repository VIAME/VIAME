# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Convert a netharn/bioharn mmdet detector (``*.zip`` deploy file) to plain ONNX.

Motivation
----------
The HabCam add-on ships three detectors that were trained with bioharn on top
of mmdet 2.x: ``scallop_cfrnn_one_class`` and ``multi_category_cfrnn``
(Cascade R-CNN) and ``scallop_hrnet_four_class`` (HRNetV2-w18 Mask R-CNN). The
last of those was serialized against the renamed ``mmcv_depr`` / ``mmdet_depr``
packages from https://github.com/VIAME/depr-mmdet-plugin, an external plugin
that is no longer built (and can no longer be built: its prebuilt ``_ext`` is a
cp310 binary against an ancient torch). Running these models through this
converter produces a self-contained ``.onnx`` graph plus a
``.modelspec.json`` sidecar that :mod:`viame.onnx.onnx_detector` executes with
nothing but onnxruntime -- no torch, no mmdet, no depr plugin.

The exported graph uses only standard ONNX operators (``RoiAlign`` and
``NonMaxSuppression`` included), so no onnxruntime custom-op library is needed.
That is a deliberate consequence of exporting through mmdet's own
``onnx_export`` methods; this module replaces the earlier mmdeploy-based
``crcnn_to_onnx``, which needed mmdeploy plus its deploy configs and broke once
mmdeploy moved from ``mmcv.Config`` to ``mmengine.Config``.

Output contract
---------------
One input ``input`` of shape ``(1, 3, H, W)`` and one output ``detections`` of
shape ``(1, N, 6)`` holding ``[x1, y1, x2, y2, score, label]`` in model-input
pixel coordinates, already NMS'd by the graph. That is the ``mmdet`` decoder of
:class:`viame.onnx.onnx_predictor.OnnxPredictor`. Boxes and labels are fused
into a single tensor on purpose: a multi-output graph trips a tuple-lowering
assert in the TorchScript ONNX exporter.

CLI:
    python -m viame.pytorch.netharn_mmdet_to_onnx \
        --deployed=models/scallop_cfrnn_one_class.zip \
        --output=models/scallop_cfrnn_one_class.onnx
"""
from __future__ import annotations

import json
import os
import types
import warnings
import zipfile

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------
def load_netharn_model(deployed_fpath):
    """Load a bioharn deploy zip, tolerating the retired depr-mmdet packages.

    Returns the outer netharn wrapper (``MM_Detector``-like), whose
    ``.detector`` is the mmdet model and whose ``.classes`` is the category
    tree.
    """
    from viame.pytorch.netharn.detect_predict import (
        patch_legacy_mm_test_cfg, setup_module_aliases)
    from torch_liberator.deployer import DeployedModel

    setup_module_aliases()
    model = DeployedModel(deployed_fpath).load_model()
    model.eval()
    patch_legacy_mm_test_cfg(model)
    return model


def read_window_dims(deployed_fpath):
    """(height, width) the model was trained to see, from its train_info.json.

    Returns None when the file does not record a window size.
    """
    if not str(deployed_fpath).endswith('.zip'):
        return None
    with zipfile.ZipFile(deployed_fpath) as zf:
        names = [n for n in zf.namelist() if n.endswith('train_info.json')]
        if len(names) != 1:
            return None
        info = json.loads(zf.read(names[0]).decode('utf-8'))
    # The config is a repr() of a python dict, not JSON, so pull the one field
    # we need out textually rather than trying to eval it.
    import re
    text = info.get('extra', {}).get('config', '')
    match = re.search(r"'window_dims':\s*\[\s*(\d+),\s*(\d+)", text)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return None


# ---------------------------------------------------------------------------
# onnx_export grafting
# ---------------------------------------------------------------------------
def _bind(obj, name, func):
    """Attach ``func`` to ``obj`` as a bound method under ``name``."""
    setattr(obj, name, types.MethodType(func, obj))


def graft_onnx_export(detector):
    """Give an old bioharn detector the mmdet ONNX-export entry points.

    Models whose topology was frozen by torch-liberator carry *copies* of the
    mmdet 2.6-era classes, which predate mmdet's ``onnx_export`` methods. The
    exported copies are otherwise the same code the installed mmdet still runs,
    so borrowing the current implementations is safe -- they only touch
    attributes (``bbox_coder``, ``loss_cls``, ``num_classes``,
    ``reg_class_agnostic``, ``_bbox_forward`` ...) that the old copies define
    identically. Native mmdet modules already have the methods and are left
    alone.
    """
    from mmdet.models.detectors.two_stage import TwoStageDetector
    from mmdet.models.roi_heads.bbox_heads.bbox_head import BBoxHead
    from mmdet.models.roi_heads.cascade_roi_head import CascadeRoIHead
    from mmdet.models.roi_heads.standard_roi_head import StandardRoIHead

    if not hasattr(detector, 'onnx_export'):
        _bind(detector, 'onnx_export', TwoStageDetector.onnx_export)

    roi_head = getattr(detector, 'roi_head', None)
    if roi_head is None:
        raise NotImplementedError(
            'only two-stage (R-CNN family) netharn models can be exported; '
            f'{type(detector).__name__} has no roi_head')

    if not hasattr(roi_head, 'onnx_export'):
        is_cascade = getattr(roi_head, 'num_stages', None) is not None
        template = CascadeRoIHead if is_cascade else StandardRoIHead
        _bind(roi_head, 'onnx_export', template.onnx_export)
        if not is_cascade:
            # StandardRoIHead.onnx_export delegates to this helper.
            _bind(roi_head, 'bbox_onnx_export', StandardRoIHead.bbox_onnx_export)

    bbox_heads = roi_head.bbox_head
    if isinstance(bbox_heads, nn.Module):
        bbox_heads = [bbox_heads]
    for head in bbox_heads:
        # mmdet gates on a property the old copies never had; the old copies
        # also only ever use plain (non-custom) classification channels.
        if not hasattr(head, 'custom_cls_channels'):
            head.custom_cls_channels = False
        if not hasattr(head, 'onnx_export'):
            _bind(head, 'onnx_export', BBoxHead.onnx_export)


def unwrap_single_stream_backbone(detector):
    """Make a late-fusion backbone accept a plain tensor.

    ``MM_Detector_V3`` models wrap their backbone in a
    ``LateFusionPyramidBackbone`` that takes a ``{channel_key: tensor}`` dict
    and sums the per-stream pyramids. Every HabCam model is single-stream
    ('rgb'), where that sum is the identity, so we can hoist the one stream up
    and expose an ordinary tensor input. Multi-stream models cannot be an ONNX
    graph with a single image input at all.
    """
    backbone = getattr(detector, 'backbone', None)
    streams = getattr(backbone, 'chan_backbones', None)
    if streams is None:
        return
    keys = list(streams.keys())
    if len(keys) != 1:
        raise NotImplementedError(
            'cannot export a multi-stream (late fusion) model to a '
            f'single-input ONNX graph; streams={keys}')
    detector.backbone = streams[keys[0]]


class _OnnxWrapper(nn.Module):
    """Adapt ``detector.onnx_export`` to a single-tensor-in/out signature.

    Fusing ``dets`` and ``labels`` into one ``(1, N, 6)`` tensor sidesteps
    ``_jit_pass_lower_all_tuples`` (which asserts on these graphs), and keeps
    the decoder in the predictor trivial. This is also why the mask branch is
    dropped before export: a second output would reintroduce the tuple.
    """

    def __init__(self, detector, meta):
        super().__init__()
        self.detector = detector
        self.meta = meta

    def forward(self, image):
        result = self.detector.onnx_export(image, [self.meta])
        dets, labels = result[0], result[1]
        return torch.cat([dets, labels.unsqueeze(-1).to(dets.dtype)], dim=-1)


# ---------------------------------------------------------------------------
# preprocessing description
# ---------------------------------------------------------------------------
def _input_normalization(model):
    """(mean, std) the ONNX graph expects the caller to apply, on 0-1 pixels.

    ``MM_Detector`` normalizes outside ``self.detector`` (so the exported graph
    starts after the normalization and the sidecar has to reproduce it), while
    ``MM_Detector_V3`` folds an ``InputNorm`` into the backbone (so the graph
    already does it and the sidecar must not do it twice).
    """
    input_norm = getattr(model, 'input_norm', None)
    if input_norm is None:
        return [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]
    mean = np.asarray(input_norm.mean.detach().cpu()).reshape(-1)
    std = np.asarray(input_norm.std.detach().cpu()).reshape(-1)
    if mean.size == 1:
        mean = np.repeat(mean, 3)
    if std.size == 1:
        std = np.repeat(std, 3)
    return [float(v) for v in mean[:3]], [float(v) for v in std[:3]]


def build_modelspec(model, net_shape, score_thresh, nms_thresh, topk, source):
    mean, std = _input_normalization(model)
    return {
        'input': {
            'shape_hw': [int(net_shape[0]), int(net_shape[1])],
            'layout': 'NCHW',
        },
        'preprocess': {
            'scale': 1.0 / 255.0,
            'normalize_mean': mean,
            'normalize_std': std,
            'interpolation': 'bilinear',
            'channel_order': 'rgb',
        },
        'postprocess': {
            'decoder': 'mmdet',
            'score_thresh': float(score_thresh),
            'nms_iou_thresh': float(nms_thresh),
            'topk': int(topk),
        },
        'meta': {
            'category_names': list(model.classes),
            'segmentation': False,
            'source_model': os.path.basename(str(source)),
            'window_dims': [int(net_shape[0]), int(net_shape[1])],
        },
    }


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
def netharn_mmdet_to_onnx(deployed_fpath,
                          output_fpath,
                          net_shape=None,
                          opset_version=11,
                          score_thresh=None,
                          nms_thresh=None,
                          max_per_img=None,
                          write_modelspec=True):
    """Export a bioharn mmdet detector zip to ONNX + ``.modelspec.json``.

    Args:
        deployed_fpath (str): the netharn ``*.zip`` deploy file.
        output_fpath (str): where to write the ``.onnx``.
        net_shape (tuple): (height, width) to fix the graph at. Defaults to the
            training window size recorded in the zip, else 640x640.
        opset_version (int): 11 is the floor for the standard ``RoiAlign`` /
            ``NonMaxSuppression`` ops these graphs emit.
        score_thresh, nms_thresh, max_per_img: override the model's own
            ``test_cfg.rcnn`` values, which are otherwise baked into the graph.

    Returns:
        str: path to the written .onnx file.
    """
    model = load_netharn_model(deployed_fpath)
    detector = model.detector

    if net_shape is None:
        net_shape = read_window_dims(deployed_fpath) or (640, 640)
    height, width = int(net_shape[0]), int(net_shape[1])

    rcnn_cfg = detector.test_cfg.rcnn
    if score_thresh is not None:
        rcnn_cfg.score_thr = float(score_thresh)
    if nms_thresh is not None:
        rcnn_cfg.nms.iou_threshold = float(nms_thresh)
    if max_per_img is not None:
        rcnn_cfg.max_per_img = int(max_per_img)

    if getattr(detector, 'with_mask', False):
        # Boxes only. A mask branch would emit a dense (1, N, H, W) float
        # tensor -- hundreds of MB per frame at N=100 -- that none of the
        # HabCam pipelines consume, and the second graph output would trip the
        # same tuple-lowering assert _OnnxWrapper exists to avoid. ``with_mask``
        # is a property over ``roi_head.mask_head``, so dropping the head is
        # what tells onnx_export to stop after the box branch.
        warnings.warn('[netharn_mmdet_to_onnx] dropping the mask branch; '
                      'the exported graph emits boxes only')
        detector.roi_head.mask_head = None
        detector.roi_head.mask_roi_extractor = None

    unwrap_single_stream_backbone(detector)
    graft_onnx_export(detector)

    meta = {
        'img_shape': (height, width, 3),
        'ori_shape': (height, width, 3),
        'pad_shape': (height, width, 3),
        'scale_factor': [1.0, 1.0, 1.0, 1.0],
        'flip': False,
        'flip_direction': None,
    }
    wrapper = _OnnxWrapper(detector, meta).eval()
    dummy = torch.rand(1, 3, height, width)

    out_dpath = os.path.dirname(os.path.abspath(output_fpath))
    if out_dpath:
        os.makedirs(out_dpath, exist_ok=True)

    with torch.no_grad():
        torch.onnx.export(
            wrapper, (dummy,), output_fpath,
            input_names=['input'], output_names=['detections'],
            export_params=True, keep_initializers_as_inputs=False,
            do_constant_folding=True, verbose=False,
            opset_version=opset_version,
            # The TorchScript exporter, not the dynamo one: mmdet's
            # onnx_export methods are written against torch.onnx symbolics.
            dynamo=False,
            # Inlining the autograd Functions (RoIAlign, the dummy NMS) throws
            # away their symbolics and then trips an internal tuple-lowering
            # assert; keeping them opaque is what routes them to the standard
            # RoiAlign / NonMaxSuppression ops.
            autograd_inlining=False)

    if write_modelspec:
        spec = build_modelspec(
            model, (height, width),
            score_thresh=rcnn_cfg.score_thr,
            nms_thresh=rcnn_cfg.nms.get('iou_threshold', 0.5),
            topk=rcnn_cfg.max_per_img,
            source=deployed_fpath)
        spec_fpath = os.path.splitext(output_fpath)[0] + '.modelspec.json'
        with open(spec_fpath, 'w') as file:
            json.dump(spec, file, indent=2)

    return output_fpath


# ---------------------------------------------------------------------------
def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert a netharn/bioharn mmdet model to ONNX')
    parser.add_argument('--deployed', required=True,
                        help='input netharn .zip deploy file')
    parser.add_argument('--output', required=True, help='output .onnx path')
    parser.add_argument('--window', default=None,
                        help='HxW (or a single N for NxN) network input size; '
                             'defaults to the training window size')
    parser.add_argument('--opset', type=int, default=11)
    parser.add_argument('--score-thresh', type=float, default=None)
    parser.add_argument('--nms-thresh', type=float, default=None)
    parser.add_argument('--max-per-img', type=int, default=None)
    args = parser.parse_args(argv)

    net_shape = None
    if args.window:
        parts = [int(v) for v in args.window.replace('x', ',').split(',')]
        net_shape = (parts[0], parts[-1])

    out = netharn_mmdet_to_onnx(
        args.deployed, args.output, net_shape=net_shape,
        opset_version=args.opset,
        score_thresh=args.score_thresh, nms_thresh=args.nms_thresh,
        max_per_img=args.max_per_img)
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
