#!/usr/bin/env python3
"""Export a trained RF-DETR checkpoint (.pth / .ckpt) to an ONNX package for the
generic ``viame.onnx`` detector plugin.

The checkpoint is loaded with the exact architecture-recovery logic the runtime
RFDETRDetector uses (:meth:`RFDETRDetector._infer_architecture` etc.), the model
is rebuilt and its weights loaded, then ``rfdetr``'s own ``model.export()`` traces
the graph and names the file. We write ``<name>.onnx`` plus a
``<name>.modelspec.json`` sidecar that tells the plugin how to preprocess
(ImageNet norm, scale 1/255) and which decoder to use (``rfdetr`` -- raw
``dets``/``labels`` outputs, per-class sigmoid + top-k, no baked NMS).

Segmentation checkpoints add a ``masks`` output (per-query mask logits) and
keypoint checkpoints a raw pose-head output, which rfdetr names ``keypoints``,
or ``outputs_keypoints`` beside masks; ``meta.outputs`` lists the graph's names
and ``meta.keypoints`` the slot names. The ``rfdetr`` decoder reads boxes and
masks only.

Usage:
  python -m viame.pytorch.onnx_exporters.rf_detr_to_onnx \
      --weights models/sealion_detector_8cls_rf_detr_l_1008.pth \
      --output-dir models/sealion_detector_8cls_rf_detr_l_1008_onnx \
      [--resolution 1008 | 960x1728] [--score-thresh 0.10] [--opset 17]
"""
import argparse
import json
import os

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

_DET = {'nano': 'RFDETRNano', 'small': 'RFDETRSmall', 'medium': 'RFDETRMedium',
        'base': 'RFDETRBase', 'large': 'RFDETRLarge'}
_SEG = {'nano': 'RFDETRSegNano', 'small': 'RFDETRSegSmall',
        'medium': 'RFDETRSegMedium', 'large': 'RFDETRSegLarge'}


# State-dict entries rfdetr rebuilds from the model config rather than trains.
_REBUILT_BUFFERS = ('_kp_active_mask',)


def _keypoint_names(args, count):
    """Checkpoint keypoint names, padded or replaced so there are ``count``."""
    names = args.get('keypoint_names') or []
    if isinstance(names, str):
        names = names.split(',')
    names = [n for n in (str(n).strip(" []'\"") for n in names) if n]
    if len(names) != count:
        names = (['head', 'tail'] + ['kp_%d' % i for i in range(2, count)])[:count]
    return names


def build_rfdetr_model(weight_fpath, device='cpu', resolution_override=0):
    """Rebuild the trained RF-DETR wrapper (an rfdetr.RFDETR* object with an
    ``.export()`` method) from a checkpoint, mirroring RFDETRDetector._build_model."""
    import torch
    from viame.pytorch.rf_detr_detector import RFDETRDetector
    from viame.pytorch.utilities import (
        ensure_rfdetr_compatibility, parse_resolution, resolution_is_set)

    ensure_rfdetr_compatibility()

    ckpt = torch.load(weight_fpath, map_location=device, weights_only=False)
    sd = RFDETRDetector._extract_state_dict(ckpt)
    args = RFDETRDetector._checkpoint_args(ckpt)
    arch = RFDETRDetector._infer_architecture(
        sd, ckpt_resolution=args.get('resolution'))

    if any(k.startswith('keypoint_head.') for k in sd):
        raise RuntimeError(
            f"{weight_fpath} was trained with VIAME's earlier RF-DETR keypoint "
            "head, which this RF-DETR release replaces with its own pose head. "
            "Retrain the model, or export it with the rf-detr branch "
            "viame/main-v1.7-backup.")

    model_size = (arch['model_size'] or str(args.get('model_size', 'large'))).lower()
    segmentation = (arch['segmentation'] if arch['segmentation'] is not None
                    else bool(args.get('segmentation')))
    num_channels = arch['num_channels'] or int(args.get('num_channels', 3))
    temporal_frames = int(arch.get('temporal_frames') or 1)
    if resolution_override:
        resolution = parse_resolution(resolution_override)
    elif arch['resolution'] is not None:
        resolution = arch['resolution']
    else:
        resolution = parse_resolution(args.get('resolution', 0))

    import rfdetr
    table = _SEG if segmentation else _DET
    if model_size not in table:
        raise ValueError(f"unknown model_size {model_size!r}")
    RFDETRModel = getattr(rfdetr, table[model_size])

    if 'class_embed.weight' in sd:
        n_classes = int(sd['class_embed.weight'].shape[0])
    else:
        n_classes = int(args.get('num_classes', 90))

    kw = dict(pretrain_weights=None, num_classes=n_classes,
              num_channels=num_channels, device=device)
    if resolution_is_set(resolution):
        kw['resolution'] = resolution
    if temporal_frames > 1:
        kw['temporal_frames'] = temporal_frames

    # Pose head: rebuilt from the keypoint count of each class.
    keypoints = None
    if any(k.startswith('keypoint_embed.') for k in sd):
        per_class = args.get('num_keypoints_per_class')
        mask = sd.get('_kp_active_mask')
        if not per_class and getattr(mask, 'ndim', 0) == 2:
            per_class = mask.sum(dim=1).tolist()
        if not per_class:
            per_class = [len(args.get('keypoint_names') or ['head', 'tail'])] * n_classes
        per_class = [int(n) for n in per_class]
        kw['use_grouppose_keypoints'] = True
        kw['num_keypoints_per_class'] = per_class
        keypoints = dict(names=_keypoint_names(args, max(per_class)),
                         num_keypoints_per_class=per_class)

    model = RFDETRModel(**kw)
    model.model.reinitialize_detection_head(n_classes)
    result = model.model.model.load_state_dict(sd, strict=False)
    missing = [k for k in result.missing_keys if k not in _REBUILT_BUFFERS]
    if missing or result.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint {weight_fpath} does not match the model built for it: "
            f"missing keys {missing[:8]}, unexpected keys "
            f"{list(result.unexpected_keys)[:8]}")

    class_names = args.get('class_names') or list(model.class_names)
    res = resolution if resolution_is_set(resolution) else 0

    # num_select drives PostProcess's top-k and VARIES by variant (large det
    # 300, seg large 200, seg nano 100, ...). Read it off the built model's
    # postprocessor so the ONNX decode top-k matches the torch detector -- a
    # wrong value only diverges on dense frames (>num_select candidates), so it
    # must come from the model, not a constant.
    num_select = 300
    pp = getattr(getattr(model, 'model', None), 'postprocess', None)
    if pp is not None and getattr(pp, 'num_select', None):
        num_select = int(pp.num_select)

    info = dict(model_size=model_size, segmentation=bool(segmentation),
                num_channels=int(num_channels), num_classes=n_classes,
                class_names=list(class_names), resolution=res,
                num_select=num_select, temporal_frames=temporal_frames,
                keypoints=keypoints)
    return model, info


def _shape_hw(resolution):
    if isinstance(resolution, (tuple, list)):
        return int(resolution[0]), int(resolution[1])
    r = int(resolution) if resolution else 0
    return r, r


def export(weights, output_dir, resolution_override=0, score_thresh=0.10,
           topk=None, opset=17, device='cpu'):
    model, info = build_rfdetr_model(weights, device=device,
                                     resolution_override=resolution_override)
    if topk is None:                     # match the model's PostProcess num_select
        topk = info['num_select']
    H, W = _shape_hw(info['resolution'] or model.model.resolution)
    if not (H and W):
        raise ValueError("could not determine model resolution; pass --resolution")

    os.makedirs(output_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(weights))[0]

    # rfdetr names the file <output_name>.onnx and returns its path.
    onnx_path = str(model.export(output_dir=output_dir, shape=(H, W),
                                 batch_size=1, opset_version=opset,
                                 output_name=name))
    if not os.path.isfile(onnx_path):
        raise RuntimeError(f"rfdetr export produced no .onnx in {output_dir}")

    # Output names as rfdetr wrote them, in graph order.
    import onnx
    graph = onnx.load(onnx_path, load_external_data=False).graph
    outputs = [output.name for output in graph.output]
    meta = {"variant": f"rf_detr_{info['model_size']}"
                       + ("_seg" if info['segmentation'] else ""),
            "category_names": info['class_names'],
            "segmentation": info['segmentation'], "opset": int(opset),
            "outputs": outputs}
    if info['keypoints']:
        # Raw pose-head rows: (queries, classes * max keypoints, 8) holding
        # normalized x, y, findable and visible logits, 3 precision terms and a class logit.
        meta["keypoints"] = info['keypoints']
    if info['temporal_frames'] > 1:
        meta["temporal_frames"] = info['temporal_frames']
    spec = {
        "modelId": name,
        "input": {"shape_hw": [H, W],
                  "channels": info['num_channels'] * info['temporal_frames'],
                  "dtype": "float32", "layout": "NCHW"},
        "preprocess": {"scale": 1.0 / 255.0,
                       "normalize_mean": IMAGENET_MEAN,
                       "normalize_std": IMAGENET_STD,
                       "interpolation": "bilinear",
                       "channel_order": "rgb"},
        "postprocess": {"decoder": "rfdetr", "score_thresh": float(score_thresh),
                        "topk": int(topk)},
        "meta": meta,
    }
    spec_path = os.path.join(output_dir, f"{name}.modelspec.json")
    with open(spec_path, "w") as f:
        json.dump(spec, f, indent=1)
    print(f"[rf_detr_to_onnx] wrote {onnx_path}")
    print(f"[rf_detr_to_onnx] wrote {spec_path}")
    print(f"[rf_detr_to_onnx] {info['model_size']} seg={info['segmentation']} "
          f"keypoints={bool(info['keypoints'])} "
          f"res={H}x{W} classes={len(info['class_names'])}")
    return onnx_path, spec_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True, help="RF-DETR .pth / .ckpt")
    ap.add_argument("--output-dir", required=True, help="ONNX package output dir")
    ap.add_argument("--resolution", default="0",
                    help="override input size, N or HEIGHTxWIDTH "
                         "(else recovered from ckpt)")
    ap.add_argument("--score-thresh", type=float, default=0.10)
    ap.add_argument("--topk", type=int, default=0,
                    help="override num_select (0 = read from the model)")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    from viame.pytorch.utilities import parse_resolution
    export(a.weights, a.output_dir,
           resolution_override=parse_resolution(a.resolution),
           score_thresh=a.score_thresh, topk=(a.topk or None),
           opset=a.opset, device=a.device)


if __name__ == "__main__":
    main()
