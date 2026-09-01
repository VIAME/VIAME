# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Convert a darknet YOLO model (``.cfg`` + ``.weights``) to ONNX.

Third of the netharn/darknet exporters, alongside
:mod:`viame.pytorch.netharn_mmdet_to_onnx` (detectors) and
:mod:`viame.pytorch.netharn_clf_to_onnx` (classifiers). It uses the bundled
``darknet2onnx`` package to rebuild the darknet graph in PyTorch, then traces
that itself (see the note in the export below) and adds the class names, the
``.modelspec.json`` sidecar and the CLI, so the result drops into
:mod:`viame.onnx.onnx_detector`.

Two upstream gaps are fixed by VIAME's patch of that package
(``packages/patches/darknet-to-pytorch-onnx``), without which neither HabCam
model converts: the YOLOv2 ``[region]`` head was unimplemented, and the
``[route]`` builder asserted that a multi-way concat lists the preceding layer
first (YOLOv7-tiny does not).

Output contract
---------------
Input ``input`` of shape ``(1, 3, H, W)``; three outputs -- ``boxes``
``(1, N, 4)`` as **normalized** ``cxcywh``, ``probs`` ``(1, N, C)`` per-class
scores, and ``confs`` ``(1, N, 1)`` objectness. That is the ``darknet`` decoder
of :class:`viame.onnx.onnx_predictor.OnnxPredictor`, which multiplies
objectness by class score, thresholds and runs NMS host-side -- darknet models
carry no NMS in the graph.

CLI:
    python -m viame.pytorch.darknet_to_onnx \
        --cfg=models/scallop_yolo_v7_one_class.cfg \
        --weights=models/scallop_yolo_v7_one_class.weights \
        --labels=models/scallop_yolo_v7_one_class.lbl \
        --output=models/scallop_yolo_v7_one_class.onnx
"""
from __future__ import annotations

import json
import os


def read_labels(labels_fpath):
    """One class name per line, in darknet's class-index order."""
    if not labels_fpath or not os.path.exists(labels_fpath):
        return []
    with open(labels_fpath) as file:
        return [line.strip() for line in file if line.strip()]


def build_modelspec(net_hw, category_names, score_thresh, nms_thresh, source,
                    resize_mode='squash'):
    return {
        'input': {
            'shape_hw': [int(net_hw[0]), int(net_hw[1])],
            'layout': 'NCHW',
        },
        'preprocess': {
            'scale': 1.0 / 255.0,
            'normalize_mean': [0.0, 0.0, 0.0],
            'normalize_std': [1.0, 1.0, 1.0],
            'interpolation': 'linear',
            # 'squash' matches darknet's resize_option=chip (the chip already
            # is the network size); 'letterbox' matches resize_option=
            # maintain_ar, where a whole frame is fitted preserving aspect.
            'resize_mode': resize_mode,
            # BGR, unlike the netharn-derived models: VIAME's C++ darknet
            # detector hands OpenCV's native BGR straight to the network, so
            # that is what these weights were trained and are used against.
            # Feeding RGB silently degrades it -- the boxes stay plausible but
            # stop matching the darknet detector at all.
            'channel_order': 'bgr',
        },
        'postprocess': {
            'decoder': 'darknet',
            'score_thresh': float(score_thresh),
            'nms_iou_thresh': float(nms_thresh),
        },
        'meta': {
            'category_names': list(category_names),
            'task': 'detection',
            'source_model': os.path.basename(str(source)),
        },
    }


def darknet_to_onnx(cfg_fpath,
                    weights_fpath,
                    output_fpath,
                    labels_fpath=None,
                    batch_size=1,
                    opset_version=17,
                    score_thresh=0.01,
                    nms_thresh=0.45,
                    resize_mode='squash',
                    write_modelspec=True):
    """Export a darknet ``.cfg``/``.weights`` pair to ONNX + ``.modelspec.json``.

    Args:
        cfg_fpath, weights_fpath: the darknet model pair.
        output_fpath (str): where to write the ``.onnx``.
        labels_fpath (str): the ``.lbl`` class-name list. Defaults to the
            ``.lbl`` sitting beside the cfg.
        batch_size (int): 1 for a static graph; pass 0 or less for a dynamic
            batch axis.
        opset_version (int): 17; these graphs are plain conv/activation stacks
            with no exotic ops.
        score_thresh, nms_thresh: recorded in the sidecar for the host-side
            decode; darknet graphs bake in neither.

    Returns:
        str: path to the written .onnx file.
    """
    import torch
    from darknet2onnx.darknet2pytorch.model import Darknet

    if labels_fpath is None:
        candidate = os.path.splitext(cfg_fpath)[0] + '.lbl'
        labels_fpath = candidate if os.path.exists(candidate) else None

    model = Darknet(cfg_fpath)
    model.load_weights(weights_fpath)
    model.eval()

    category_names = read_labels(labels_fpath)
    num_classes = getattr(model, 'num_classes', len(category_names))
    if category_names and num_classes and len(category_names) != num_classes:
        raise ValueError(
            f'{labels_fpath} lists {len(category_names)} classes but '
            f'{cfg_fpath} declares {num_classes}')
    if not category_names:
        category_names = [str(i) for i in range(num_classes or 0)]

    out_dpath = os.path.dirname(os.path.abspath(output_fpath))
    if out_dpath:
        os.makedirs(out_dpath, exist_ok=True)

    # Not darknet2onnx's own export_darknet_to_onnx(): it hardcodes opset 11
    # and leaves `dynamo` at torch's default, which on torch>=2.9 selects the
    # dynamo exporter, whose downconvert to 11 dies in onnx's version converter
    # ("No Adapter From Version 16 for Identity") and silently writes a graph
    # with no weights. Same TorchScript path and opset as the other exporters.
    dynamic_axes = None
    n_batch = batch_size if batch_size > 0 else 1
    if batch_size <= 0:
        dynamic_axes = {'input': {0: 'batch'}, 'boxes': {0: 'batch'},
                        'probs': {0: 'batch'}, 'confs': {0: 'batch'}}
    dummy = torch.randn(n_batch, 3, model.height, model.width)
    with torch.no_grad():
        torch.onnx.export(
            model, (dummy,), output_fpath,
            input_names=['input'], output_names=['boxes', 'probs', 'confs'],
            export_params=True, do_constant_folding=True, verbose=False,
            opset_version=opset_version, dynamic_axes=dynamic_axes,
            dynamo=False)

    if write_modelspec:
        spec = build_modelspec((model.height, model.width), category_names,
                               score_thresh, nms_thresh, weights_fpath,
                               resize_mode=resize_mode)
        spec_fpath = os.path.splitext(output_fpath)[0] + '.modelspec.json'
        with open(spec_fpath, 'w') as file:
            json.dump(spec, file, indent=2)

    return output_fpath


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description='Convert a darknet YOLO model to ONNX')
    parser.add_argument('--cfg', required=True, help='darknet .cfg')
    parser.add_argument('--weights', required=True, help='darknet .weights')
    parser.add_argument('--output', required=True, help='output .onnx path')
    parser.add_argument('--labels', default=None,
                        help='.lbl class-name list (defaults to <cfg>.lbl)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='0 or less exports a dynamic batch axis')
    parser.add_argument('--score-thresh', type=float, default=0.01)
    parser.add_argument('--nms-thresh', type=float, default=0.45)
    parser.add_argument('--resize-mode', choices=['squash', 'letterbox'],
                        default='squash',
                        help='letterbox reproduces darknet resize_option=maintain_ar')
    args = parser.parse_args(argv)

    out = darknet_to_onnx(
        args.cfg, args.weights, args.output, labels_fpath=args.labels,
        batch_size=args.batch_size, score_thresh=args.score_thresh,
        nms_thresh=args.nms_thresh, resize_mode=args.resize_mode)
    print(f'wrote {out} ({os.path.getsize(out) / 1e6:.1f} MB)')


if __name__ == '__main__':
    main()
