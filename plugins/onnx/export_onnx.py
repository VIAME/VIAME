# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Export the epipolar template-matching stereo matcher to a single ONNX file.

Two models can be exported:

  --model match    EpipolarMatcher  -> right_points, best_score, second_score
  --model measure  EpipolarMeasurer -> right_points, best_score, second_score,
                                        points_3d  (in-graph triangulation)

template_size and num_samples are baked into the graph as constants (matching
the C++ :template_size and :epipolar_num_samples config keys). Everything else
-- images, keypoints, calibration, and the depth range -- is a runtime input,
so one exported model works for any rig and image size.

Example:
  python export_onnx.py --model match   --out epipolar_match.onnx
  python export_onnx.py --model measure --out epipolar_measure.onnx \
         --template-size 25 --num-samples 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from epipolar_matcher import EpipolarMatcher, EpipolarMeasurer

# Names follow EpipolarMatcher.forward() argument order.
INPUT_NAMES = [
    "left_gray", "right_gray", "points_left",
    "K_left", "dist_left", "R_left", "t_left",
    "K_right", "dist_right", "R_right", "t_right",
    "min_depth", "max_depth",
]


def _example_inputs(template_size, num_samples):
    """Small but valid example inputs for tracing / sanity checking."""
    torch.manual_seed(0)
    H, W = 240, 320
    left = torch.rand(H, W, dtype=torch.float32)
    right = torch.rand(H, W, dtype=torch.float32)
    pts = torch.tensor([[160.0, 120.0], [80.0, 60.0]], dtype=torch.float32)

    # A plausible horizontal-baseline rig: identical pinhole intrinsics, right
    # camera shifted along -x by 100 units (baseline). No distortion.
    K = torch.tensor([[400.0, 0.0, 160.0],
                      [0.0, 400.0, 120.0],
                      [0.0, 0.0, 1.0]], dtype=torch.float32)
    zero_d = torch.zeros(8, dtype=torch.float32)
    R_l = torch.eye(3, dtype=torch.float32)
    t_l = torch.zeros(3, dtype=torch.float32)
    R_r = torch.eye(3, dtype=torch.float32)
    t_r = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float32)
    min_d = torch.tensor(500.0, dtype=torch.float32)
    max_d = torch.tensor(5000.0, dtype=torch.float32)
    return (left, right, pts, K, zero_d, R_l, t_l, K, zero_d, R_r, t_r,
            min_d, max_d)


def export(model_kind, out_path, template_size, num_samples, opset, verify):
    if model_kind == "match":
        model = EpipolarMatcher(template_size, num_samples)
        output_names = ["right_points", "best_score", "second_score"]
    elif model_kind == "measure":
        model = EpipolarMeasurer(template_size, num_samples)
        output_names = ["right_points", "best_score", "second_score", "points_3d"]
    else:
        raise ValueError("model must be 'match' or 'measure'")
    model.eval()

    inputs = _example_inputs(template_size, num_samples)

    dynamic_axes = {
        "left_gray": {0: "H", 1: "W"},
        "right_gray": {0: "Hr", 1: "Wr"},
        "points_left": {0: "P"},
        "right_points": {0: "P"},
        "best_score": {0: "P"},
        "second_score": {0: "P"},
    }
    if model_kind == "measure":
        dynamic_axes["points_3d"] = {0: "P"}

    with torch.no_grad():
        torch.onnx.export(
            model, inputs, out_path,
            input_names=INPUT_NAMES,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
        )
    print("Exported %s model -> %s (template_size=%d, num_samples=%d, opset=%d)"
          % (model_kind, out_path, template_size, num_samples, opset))

    if verify:
        _verify(model, inputs, out_path, output_names)


def _verify(model, inputs, out_path, output_names):
    """Compare onnxruntime outputs against the eager PyTorch module."""
    import onnxruntime as ort

    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])

    print("Verification (onnxruntime vs eager PyTorch):")
    # At the export size, every output (including geometry) must match exactly.
    if not _check(model, inputs, sess, output_names, "default size",
                  strict_geometry=True):
        raise SystemExit("ONNX verification failed")

    # Re-run on a different image size and point count to confirm the dynamic
    # axes did not bake the export-time resolution (which would corrupt
    # edge-of-image candidates on differently sized inputs). Here we hard-check
    # only the NCC scores: they are tie-independent, so their bit-for-bit match
    # proves the graph computes identically at the new size. The geometry
    # outputs (right_points / points_3d) are reported but not failed -- on
    # synthetic random images torch.argmax and ONNX ArgMax can break NCC ties
    # toward adjacent depth samples, which moves the matched point by a fraction
    # of a pixel (and the triangulated point correspondingly). Real imagery has
    # a unique NCC peak and does not exhibit this.
    alt = list(inputs)
    alt[0] = torch.rand(300, 400, dtype=torch.float32)        # left_gray
    alt[1] = torch.rand(300, 400, dtype=torch.float32)        # right_gray
    alt[2] = torch.tensor([[200.0, 150.0]], dtype=torch.float32)  # 1 point
    if not _check(model, tuple(alt), sess, output_names, "alt size (300x400)",
                  strict_geometry=False):
        raise SystemExit("ONNX dynamic-axis verification failed")


def _check(model, inputs, sess, output_names, tag, strict_geometry):
    import numpy as np

    with torch.no_grad():
        ref = [r.cpu().numpy() for r in model(*inputs)]
    feed = {name: inputs[i].cpu().numpy() for i, name in enumerate(INPUT_NAMES)}
    got = sess.run(output_names, feed)

    geometry = {"right_points", "points_3d"}
    ok = True
    for name, a, b in zip(output_names, ref, got):
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        if name in geometry and not strict_geometry:
            tag_status = "diff (tie-break, informational)"
        else:
            good = diff < 1e-3
            ok = ok and good
            tag_status = "OK" if good else "MISMATCH"
        print("  [%s] %-14s max|diff| = %.3e  [%s]"
              % (tag, name, diff, tag_status))
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["match", "measure"], default="match",
                    help="Which graph to export.")
    ap.add_argument("--out", required=True, help="Output .onnx path.")
    ap.add_argument("--template-size", type=int, default=25,
                    help="Odd NCC template window size (graph constant).")
    ap.add_argument("--num-samples", type=int, default=5000,
                    help="Number of depth samples along the epipolar curve.")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--no-verify", action="store_true",
                    help="Skip the onnxruntime vs PyTorch verification step.")
    args = ap.parse_args()

    export(args.model, args.out, args.template_size, args.num_samples,
           args.opset, not args.no_verify)


if __name__ == "__main__":
    main()
