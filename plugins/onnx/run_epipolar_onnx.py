# This file is part of VIAME, and is distributed under an OSI-approved #
# BSD 3-Clause License. See either the root top-level LICENSE file or  #
# https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    #

"""
Run the exported epipolar template-matching ONNX model end to end.

This is the host driver around the single .onnx graph: it loads a calibration
file (any format read_stereo_rig supports), loads the left/right images and a
set of left-image keypoints, feeds them to onnxruntime, applies the score
threshold and uniqueness-ratio test, and triangulates the matched pairs.

It reproduces the relevant config of measurement_from_annotations_template.pipe
(:template_matching_threshold, :uniqueness_ratio, :epipolar_min/max_disparity
or depth, :template_size baked into the model at export).

Triangulation:
  * "match"  model  -> matched points only; this script triangulates host-side
    with triangulate_fast_numpy (bit-exact port of triangulate_fast_two_view).
  * "measure" model -> uses the graph's own points_3d output (close port; see
    triangulate.py for the single documented deviation).

Examples:
  # explicit head/tail pairs (x,y), length per pair
  python run_epipolar_onnx.py \
      --onnx epipolar_match.onnx \
      --calibration calibration_matrices.npz \
      --left-image left.png --right-image right.png \
      --points 812,455 1033,470 --pairs \
      --min-disparity 7 --max-disparity 724

  # points from a file (one "x,y" per line)
  python run_epipolar_onnx.py --onnx epipolar_measure.onnx \
      --calibration calib.json --left-image l.png --right-image r.png \
      --points-file keypoints.txt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from calibration_io import load_calibration
import geometry_numpy as geom
from triangulate import triangulate_fast_numpy


def _parse_points(args):
    pts = []
    if args.points:
        for tok in args.points:
            x, y = tok.split(",")
            pts.append([float(x), float(y)])
    if args.points_file:
        with open(args.points_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                x, y = line.replace(",", " ").split()[:2]
                pts.append([float(x), float(y)])
    if not pts:
        raise SystemExit("No keypoints given (use --points or --points-file)")
    return np.asarray(pts, dtype=np.float32)


def _depth_range(args, calib):
    """Resolve the depth-sampling range, from depth or disparity bounds.

    Mirrors the C++: disparity bounds take precedence and convert via
    depth = focal * baseline / disparity (min disparity -> max depth).
    """
    if args.min_disparity > 0 and args.max_disparity > 0:
        fx = float(calib["K_left"][0, 0])
        baseline = float(np.linalg.norm(calib["t_right"] - calib["t_left"]))
        min_depth = fx * baseline / args.max_disparity
        max_depth = fx * baseline / args.min_disparity
        return min_depth, max_depth
    if args.min_depth > 0 and args.max_depth > 0:
        return args.min_depth, args.max_depth
    raise SystemExit("Specify a search range: --min/--max-disparity or "
                     "--min/--max-depth")


def _load_gray(path):
    """Load an image as a float32 grayscale array. Prefers OpenCV (matching the
    C++ BGR2GRAY conversion); falls back to Pillow if cv2 is unavailable."""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise SystemExit("Could not read image: " + path)
        return img.astype(np.float32)
    except ImportError:
        from PIL import Image
        return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def _load_rgb(path, exp_h, exp_w):
    """Load an image as float32 RGB [3, H, W] in [0, 255] for the DINO models.
    The DINO graph has a fixed input resolution; the image must match it."""
    from PIL import Image
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32)            # [H, W, 3] RGB
    h, w = arr.shape[:2]
    if (exp_h and h != exp_h) or (exp_w and w != exp_w):
        raise SystemExit(
            "DINO model expects %sx%s images but %s is %dx%d. Re-export with "
            "--height %d --width %d, or resize the inputs."
            % (exp_h, exp_w, os.path.basename(path), h, w, h, w))
    return np.transpose(arr, (2, 0, 1)).copy()         # [3, H, W]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True, help="Exported .onnx model.")
    ap.add_argument("--calibration", required=True,
                    help="Calibration file/dir (.npz/.json/.yml/.mat/dir).")
    ap.add_argument("--left-image", required=True)
    ap.add_argument("--right-image", required=True)
    ap.add_argument("--points", nargs="+", help='Left points as "x,y" tokens.')
    ap.add_argument("--points-file", help='File of "x,y" per line.')
    ap.add_argument("--pairs", action="store_true",
                    help="Treat points as consecutive head/tail pairs and "
                         "report a length per pair.")
    ap.add_argument("--min-disparity", type=float, default=0.0)
    ap.add_argument("--max-disparity", type=float, default=0.0)
    ap.add_argument("--min-depth", type=float, default=0.0)
    ap.add_argument("--max-depth", type=float, default=0.0)
    ap.add_argument("--template-threshold", type=float, default=0.2,
                    help="Minimum NCC score to accept a match.")
    ap.add_argument("--uniqueness-ratio", type=float, default=0.85,
                    help="Reject if second_score/best_score exceeds this "
                         "(0 disables).")
    ap.add_argument("--output", help="Optional CSV output path.")
    args = ap.parse_args()

    import onnxruntime as ort

    calib = load_calibration(args.calibration)
    points = _parse_points(args)
    min_depth, max_depth = _depth_range(args, calib)

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    in_names = [i.name for i in sess.get_inputs()]

    # DINO models take color [3, H, W] RGB inputs (named left_rgb/right_rgb) at a
    # fixed export resolution; NCC-only models take grayscale [H, W].
    if "left_rgb" in in_names:
        shape = next(i.shape for i in sess.get_inputs() if i.name == "left_rgb")
        exp_h = shape[1] if isinstance(shape[1], int) else None
        exp_w = shape[2] if isinstance(shape[2], int) else None
        left = _load_rgb(args.left_image, exp_h, exp_w)
        right = _load_rgb(args.right_image, exp_h, exp_w)
        img_keys = {"left_rgb": left, "right_rgb": right}
    else:
        img_keys = {"left_gray": _load_gray(args.left_image),
                    "right_gray": _load_gray(args.right_image)}

    feed = {
        **img_keys,
        "points_left": points,
        "K_left": calib["K_left"].astype(np.float32),
        "dist_left": calib["dist_left"].astype(np.float32),
        "R_left": calib["R_left"].astype(np.float32),
        "t_left": calib["t_left"].astype(np.float32),
        "K_right": calib["K_right"].astype(np.float32),
        "dist_right": calib["dist_right"].astype(np.float32),
        "R_right": calib["R_right"].astype(np.float32),
        "t_right": calib["t_right"].astype(np.float32),
        "min_depth": np.asarray(min_depth, dtype=np.float32),
        "max_depth": np.asarray(max_depth, dtype=np.float32),
    }

    out_names = [o.name for o in sess.get_outputs()]
    outs = dict(zip(out_names, sess.run(out_names, feed)))

    right_points = outs["right_points"]
    best = outs["best_score"]
    second = outs["second_score"]
    has_3d = "points_3d" in outs

    print("Search depth range: [%.2f, %.2f]" % (min_depth, max_depth))
    print("%-4s %-19s %-19s %-7s %-7s %-6s" %
          ("idx", "left(x,y)", "right(x,y)", "score", "2nd", "ok"))

    accepted = []
    pts3d = []
    for i in range(len(points)):
        lx, ly = points[i]
        rx, ry = right_points[i]
        ok = best[i] >= args.template_threshold
        if ok and args.uniqueness_ratio > 0 and second[i] > 0 and best[i] > 0:
            ok = (second[i] / best[i]) <= args.uniqueness_ratio
        accepted.append(bool(ok))

        if has_3d:
            p3 = outs["points_3d"][i]
        else:
            ln = geom.unmap(float(lx), float(ly), calib["K_left"], calib["dist_left"])
            rn = geom.unmap(float(rx), float(ry), calib["K_right"], calib["dist_right"])
            p3 = triangulate_fast_numpy(
                np.array(ln), np.array(rn),
                calib["R_left"], calib["t_left"],
                calib["R_right"], calib["t_right"])
        pts3d.append(p3)

        print("%-4d (%8.2f,%8.2f) (%8.2f,%8.2f) %6.3f %6.3f %-6s" %
              (i, lx, ly, rx, ry, best[i], second[i], "yes" if ok else "NO"))

    if args.pairs:
        _report_pairs(points, right_points, pts3d, accepted, calib)

    if args.output:
        _write_csv(args.output, points, right_points, best, second,
                   accepted, pts3d)
        print("Wrote", args.output)


def _report_pairs(points, right_points, pts3d, accepted, calib):
    print("\nPair measurements (head/tail):")
    for p in range(0, len(points) - 1, 2):
        h, t = p, p + 1
        if not (accepted[h] and accepted[t]):
            print("  pair %d-%d: incomplete (a keypoint was rejected)" % (h, t))
            continue
        ph, pt = pts3d[h], pts3d[t]
        length = float(np.linalg.norm(pt - ph))
        mid = (ph + pt) / 2.0
        left_center = -calib["R_left"].T @ calib["t_left"]  # = 0 for left at origin
        rng = float(np.linalg.norm(mid - left_center))

        # RMS reprojection error over the 4 measurements, as in
        # compute_stereo_measurement.
        def err2(p3, lpt, rpt):
            lr = geom.project(p3, calib["K_left"], calib["dist_left"],
                              calib["R_left"], calib["t_left"])
            rr = geom.project(p3, calib["K_right"], calib["dist_right"],
                              calib["R_right"], calib["t_right"])
            return np.sum((lr - lpt) ** 2) + np.sum((rr - rpt) ** 2)

        e = (err2(ph, points[h], right_points[h]) +
             err2(pt, points[t], right_points[t]))
        rms = float(np.sqrt(e / 4.0))
        print("  pair %d-%d: length=%.3f  midpoint=(%.2f,%.2f,%.2f)  "
              "range=%.3f  rms=%.3fpx" %
              (h, t, length, mid[0], mid[1], mid[2], rng, rms))


def _write_csv(path, points, right_points, best, second, accepted, pts3d):
    with open(path, "w") as f:
        f.write("idx,left_x,left_y,right_x,right_y,score,second,accepted,"
                "X,Y,Z\n")
        for i in range(len(points)):
            p3 = pts3d[i]
            f.write("%d,%.4f,%.4f,%.4f,%.4f,%.5f,%.5f,%d,%.5f,%.5f,%.5f\n" %
                    (i, points[i, 0], points[i, 1],
                     right_points[i, 0], right_points[i, 1],
                     best[i], second[i], int(accepted[i]),
                     p3[0], p3[1], p3[2]))


if __name__ == "__main__":
    main()
