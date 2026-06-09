# Epipolar template matching as a single ONNX graph

This plugin reimplements VIAME's "regular computer-vision" stereo correspondence
method — the one selected by `epipolar_template_matching` in
[`measurement_from_annotations_template.pipe`](../../configs/pipelines/measurement_from_annotations_template.pipe)
— as a single, self-contained `.onnx` file.

It is the first of the three stereo-measurement methods (the others being the
DINO-augmented matcher and the depth-map matcher); only this CV method is
covered here.

## Is it possible? Yes — with one host/graph boundary

ONNX is a static tensor-graph format. The expensive, per-keypoint image work of
method (1) is pure tensor math and maps cleanly into one graph:

| C++ (`plugins/core/measurement_utilities.cxx`) | ONNX graph |
| --- | --- |
| `compute_epipolar_points` (unproject ray → sample depths → reproject to target, full lens distortion) | `compute_epipolar_points` in `epipolar_matcher.py` |
| `find_corresponding_point_epipolar_template_matching` / `score_template_at_point` (`TM_CCOEFF_NORMED`) | patch gather + `_ncc` + ArgMax |
| score threshold + uniqueness ratio | output `best_score`/`second_score`; host applies thresholds |
| `triangulate_fast_two_view` | host (`triangulate_fast_numpy`, **bit-exact**) or in-graph (`triangulate_fast_torch`, close port) |

Two things **cannot** live inside ONNX and stay on the host:

1. **Parsing calibration files.** ONNX has no file I/O. `calibration_io.py`
   mirrors `read_stereo_rig` (`.npz/.json/.yml/.mat/`dir`) and feeds the graph
   `K/dist/R/t` as **runtime tensor inputs**, so one exported model works for
   any rig and any calibration format.
2. **The SVD in homogeneous DLT triangulation.** ONNX has no stable SVD/Inverse
   operator. The `match` model therefore leaves triangulation to the host
   (bit-exact). The `measure` model triangulates in-graph but replaces the final
   4×4 homogeneous null-space solve with the equivalent **inhomogeneous**
   normal-equations solve (analytic 3×3 inverse); the closed-form Lindstrom
   optimal correction is reproduced exactly. For finite points the two agree to
   floating-point precision.

The chosen NCC variant is the point-by-point one (`ncc`), not the FFT strip
(`ncc_strip`), because the strip needs a dynamic crop; the scores are the same
normalized cross-correlation, so results match.

## Files

| File | Role |
| --- | --- |
| `epipolar_matcher.py` | Torch modules `EpipolarMatcher` / `EpipolarMeasurer` (the graph). |
| `triangulate.py` | `triangulate_fast_numpy` (host, exact) and `triangulate_fast_torch` (graph). |
| `calibration_io.py` | Load `.npz/.json/.yml/.mat/`dir` → calibration tensors. |
| `geometry_numpy.py` | NumPy intrinsics map/unmap/project for host normalization + RMS. |
| `export_onnx.py` | Export either model to one `.onnx` (+ onnxruntime verification). |
| `run_epipolar_onnx.py` | End-to-end driver: calib + images + keypoints → matches + measurements. |

## Usage

```bash
# Export (template_size / num_samples become graph constants)
python export_onnx.py --model match   --out epipolar_match.onnx
python export_onnx.py --model measure  --out epipolar_measure.onnx \
       --template-size 25 --num-samples 5000

# Run: match head/tail pairs and measure length
python run_epipolar_onnx.py \
    --onnx epipolar_match.onnx \
    --calibration calibration_matrices.npz \
    --left-image left.png --right-image right.png \
    --points 812,455 1033,470 --pairs \
    --min-disparity 7 --max-disparity 724 \
    --template-threshold 0.2 --uniqueness-ratio 0.85
```

The graph inputs are `left_gray, right_gray, points_left, K_left, dist_left,
R_left, t_left, K_right, dist_right, R_right, t_right, min_depth, max_depth`;
outputs are `right_points, best_score, second_score` (+ `points_3d` for the
`measure` model). Point count and image size are dynamic axes.

## Conventions

World frame = left camera (`P_cam = R·P_world + t`), so the left camera is
`R = I, t = 0` and the right camera carries the rig's relative `R, T` — matching
`camera_rig_io.cxx`. Distortion is the vital radial-tangential model
`[k1,k2,p1,p2,k3,k4,k5,k6]`, undistorted with the same 5-iteration Gauss-Newton
scheme as `simple_camera_intrinsics`.

## Build

Built only when `VIAME_ENABLE_ONNX` (and Python) are on; installed as the
`onnx_stereo` python package. Runtime deps: `onnxruntime`, `numpy`, and either
`opencv` or `Pillow` for image loading (the runner falls back to Pillow if cv2
is unavailable), optional `scipy` (.mat). Export additionally needs `torch` and
`onnxscript` (torch 2.x routes `torch.onnx.export` through the dynamo exporter).
Models are exported at opset 18.

Validated with torch 2.12 / onnxruntime 1.23 / numpy 2.0: both models export and
onnxruntime reproduces the eager PyTorch outputs (NCC scores to ~1e-8, matched
points and triangulation exact); a synthetic end-to-end run matches both
keypoints to ~0.25 px and the `match` (host) and `measure` (in-graph)
triangulations agree.
