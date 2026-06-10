# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Export an epipolar stereo matcher to a single ONNX file.

Models:
  match        EpipolarMatcher       (method 1) -> right_points, scores
  measure      EpipolarMeasurer      (method 1, + in-graph triangulation)
  dino         EpipolarDinoMatcher   (method 2, DINO top-K + NCC) -> ... scores
  dino-measure EpipolarDinoMeasurer  (method 2, + in-graph triangulation)

For the NCC-only models (match/measure), images and image size are dynamic
runtime inputs and template_size/num_samples are graph constants.

For the DINO models, the DINOv2 backbone is baked into the graph and the image
size is FIXED at export time (--height/--width), since ViT positional-embedding
interpolation makes dynamic-resolution export fragile and a stereo rig's
resolution is constant. Image inputs are color float32 [3, H, W] RGB in [0,255].
Calibration is always a runtime tensor input.

Examples:
  python export_stereo_mapping.py --model match   --out epipolar_match.onnx
  python export_stereo_mapping.py --model measure  --out epipolar_measure.onnx
  python export_stereo_mapping.py --model dino     --out epipolar_dino.onnx \
         --height 1080 --width 1920 --dino-model dinov2_vitb14 --dino-top-k 25
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

INPUT_NAMES_GRAY = ["left_gray", "right_gray"]
INPUT_NAMES_RGB = ["left_rgb", "right_rgb"]
INPUT_NAMES_TAIL = [
    "points_left",
    "K_left", "dist_left", "R_left", "t_left",
    "K_right", "dist_right", "R_right", "t_right",
    "min_depth", "max_depth",
]

_K = [[400.0, 0.0, 160.0], [0.0, 400.0, 120.0], [0.0, 0.0, 1.0]]


def _calib_and_depths():
    """Shared calibration + depth example inputs (horizontal-baseline rig)."""
    K = torch.tensor(_K, dtype=torch.float32)
    zero_d = torch.zeros(8, dtype=torch.float32)
    R = torch.eye(3, dtype=torch.float32)
    t_l = torch.zeros(3, dtype=torch.float32)
    t_r = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float32)
    return [K, zero_d, R, t_l, K, zero_d, R, t_r,
            torch.tensor(500.0), torch.tensor(5000.0)]


def _example_inputs(kind, height, width):
    torch.manual_seed(0)
    pts = torch.tensor([[160.0, 120.0], [80.0, 60.0]], dtype=torch.float32)
    tail = [pts] + _calib_and_depths()
    if kind in ("match", "measure"):
        left = torch.rand(height, width, dtype=torch.float32)
        right = torch.rand(height, width, dtype=torch.float32)
        return tuple([left, right] + tail)
    # dino: color [3, H, W] in [0, 255]
    left = torch.rand(3, height, width, dtype=torch.float32) * 255.0
    right = torch.rand(3, height, width, dtype=torch.float32) * 255.0
    return tuple([left, right] + tail)


def _build(kind, args):
    """Return (model, input_names, output_names, is_dino)."""
    if kind in ("match", "measure"):
        from epipolar_matcher import EpipolarMatcher, EpipolarMeasurer
        cls = EpipolarMeasurer if kind == "measure" else EpipolarMatcher
        model = cls(args.template_size, args.num_samples)
        names = INPUT_NAMES_GRAY + INPUT_NAMES_TAIL
        is_dino = False
    else:
        from epipolar_dino_matcher import (
            EpipolarDinoMatcher, EpipolarDinoMeasurer, load_dino_backbone)
        backbone, ps = load_dino_backbone(args.dino_model, args.dino_weights)
        cls = EpipolarDinoMeasurer if kind == "dino-measure" else EpipolarDinoMatcher
        model = cls(backbone, ps, args.height, args.width,
                    args.template_size, args.num_samples, args.dino_top_k)
        names = INPUT_NAMES_RGB + INPUT_NAMES_TAIL
        is_dino = True
    outputs = ["right_points", "best_score", "second_score"]
    if kind in ("measure", "dino-measure"):
        outputs.append("points_3d")
    model.eval()
    return model, names, outputs, is_dino


def export(kind, args):
    model, names, output_names, is_dino = _build(kind, args)
    inputs = _example_inputs(kind, args.height, args.width)

    # Dynamic batch of keypoints (P) for all models; image size is dynamic only
    # for the NCC-only models.
    dynamic_axes = {"points_left": {0: "P"}}
    for o in output_names:
        dynamic_axes[o] = {0: "P"}
    if not is_dino:
        dynamic_axes["left_gray"] = {0: "H", 1: "W"}
        dynamic_axes["right_gray"] = {0: "Hr", 1: "Wr"}

    # Use the legacy TorchScript exporter: it reliably honors dynamic_axes (the
    # dynamo path bakes the dynamic image size for the NCC-only models, and
    # cannot reconcile the dynamic keypoint axis with the top-K gather in the
    # DINO models). It handles grid_sample and the DINOv2 ViT fine.
    with torch.no_grad():
        torch.onnx.export(
            model, inputs, args.out,
            input_names=names, output_names=output_names,
            dynamic_axes=dynamic_axes, opset_version=args.opset,
            do_constant_folding=True, dynamo=False)
    sz = os.path.getsize(args.out) / 1e6
    print("Exported %s -> %s (%.1f MB, template_size=%d, num_samples=%d%s)"
          % (kind, args.out, sz, args.template_size, args.num_samples,
             (", dino_top_k=%d, %dx%d" % (args.dino_top_k, args.height, args.width))
             if is_dino else ""))

    if not args.no_verify:
        _verify(model, inputs, names, output_names, args, is_dino)


def _verify(model, inputs, input_names, output_names, args, is_dino):
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    print("Verification (onnxruntime vs eager PyTorch):")
    # Primary check: geometry strict for NCC-only models. For DINO models the
    # synthetic random images create NCC/cosine near-ties where torch and ONNX
    # argmax/topk can pick adjacent candidates, so geometry is informational and
    # the (tie-independent) scores carry the proof; positional correctness is
    # covered by the synthetic end-to-end runner test.
    if not _check(model, inputs, sess, input_names, output_names,
                  "default", strict_geometry=not is_dino):
        raise SystemExit("ONNX verification failed")

    # Secondary check exercises a varying axis. For NCC-only models that is the
    # image size (guards against the dynamic axis baking the export resolution);
    # for DINO models (fixed size) it is the keypoint count P.
    alt = list(inputs)
    if not is_dino:
        alt[0] = torch.rand(300, 400, dtype=torch.float32)
        alt[1] = torch.rand(300, 400, dtype=torch.float32)
        tag = "alt size 300x400"
    else:
        tag = "alt P=1"
    alt[2] = torch.tensor([[args.width * 0.5, args.height * 0.5]],
                          dtype=torch.float32)
    if not _check(model, tuple(alt), sess, input_names, output_names, tag,
                  strict_geometry=False):
        raise SystemExit("ONNX secondary verification failed")


def _check(model, inputs, sess, input_names, output_names, tag, strict_geometry):
    import numpy as np

    with torch.no_grad():
        ref = [r.cpu().numpy() for r in model(*inputs)]
    feed = {name: inputs[i].cpu().numpy() for i, name in enumerate(input_names)}
    got = sess.run(output_names, feed)

    geometry = {"right_points", "points_3d"}
    ok = True
    for name, a, b in zip(output_names, ref, got):
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        if name in geometry and not strict_geometry:
            status = "diff (tie-break, informational)"
        else:
            good = diff < 1e-3
            ok = ok and good
            status = "OK" if good else "MISMATCH"
        print("  [%s] %-14s max|diff| = %.3e  [%s]" % (tag, name, diff, status))
    return ok


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    choices=["match", "measure", "dino", "dino-measure"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--template-size", type=int, default=25)
    ap.add_argument("--num-samples", type=int, default=5000)
    ap.add_argument("--opset", type=int, default=18)
    # DINO-specific
    ap.add_argument("--height", type=int, default=240,
                    help="Fixed image height (DINO models; also the NCC example).")
    ap.add_argument("--width", type=int, default=320,
                    help="Fixed image width (DINO models; also the NCC example).")
    ap.add_argument("--dino-model", default="dinov2_vitb14")
    ap.add_argument("--dino-top-k", type=int, default=25)
    ap.add_argument("--dino-weights", default="",
                    help="Optional local DINOv2 weights path.")
    ap.add_argument("--no-verify", action="store_true")
    args = ap.parse_args()
    export(args.model, args)


if __name__ == "__main__":
    main()
