# VIAME ONNX plugins

Two independent groups of pure-Python, onnxruntime-based tools live here (no
torch at inference time):

1. **A generic ONNX object detector** (`onnx_predictor.py` + `onnx_detector.py`,
   registered as the kwiver `onnx` algorithm) — see below.
2. **Epipolar stereo matching as a single ONNX graph** — the rest of this file.

## Generic ONNX object detector

`onnx_detector.py` runs any detection graph described by a
`<model>.modelspec.json` sidecar sitting next to the `.onnx`. The sidecar names
the input size, the preprocessing (scale, per-channel mean/std, interpolation,
RGB vs BGR), the postprocessing thresholds, the class names, and a *decoder*
that says how to read the graph's outputs:

| `postprocess.decoder` | Graph contract |
| --- | --- |
| `detr` / `baked` (default) | DEIMv2 / RT-DETR: `images` + `orig_target_sizes` in, `(labels, boxes_xyxy, scores)` out, NMS baked in. |
| `rfdetr` | RF-DETR: raw `dets` (normalized `cxcywh`) + `labels` logits; top-k over the query x class grid host-side, optional per-query masks. |
| `yolo` | One `(1, 4+C, N)` / `(1, N, 4+C)` output in `cxcywh`; decode, threshold and NMS host-side. |
| `mmdet` | mmdet 2.x R-CNN family: one `(1, N, 6)` output of `[x1, y1, x2, y2, score, label]` in model-input pixels, NMS baked in. |
| `darknet` | darknet YOLO (v2 `[region]` through v7 `[yolo]`): `boxes` `(1, N, 4)` normalized `cxcywh`, `probs` `(1, N, C)`, `confs` `(1, N, 1)`; score is objectness x class, thresholded and NMS'd host-side. |

The detector is whole-frame only; wrap it in `ocv_windowed` to tile large
imagery, exactly as the other VIAME detectors are wrapped.

### Converting netharn/bioharn mmdet models (the `mmdet` decoder)

The HabCam add-on's detectors are bioharn Cascade R-CNN / HRNet Mask R-CNN zips
built against mmdet 2.x, and one of them was serialized against the renamed
`mmcv_depr` / `mmdet_depr` packages from the external
[depr-mmdet-plugin](https://github.com/VIAME/depr-mmdet-plugin), which is no
longer buildable. `viame.pytorch.netharn_mmdet_to_onnx` converts such a zip into
a graph the `onnx` detector runs with nothing but onnxruntime:

```bash
python -m viame.pytorch.netharn_mmdet_to_onnx \
    --deployed=models/scallop_cfrnn_one_class.zip \
    --output=models/scallop_cfrnn_one_class.onnx
```

That writes both the `.onnx` and its `.modelspec.json`. The network input size
defaults to the training window recorded in the zip (`--window` overrides it),
and the graph emits only standard ONNX operators — `RoiAlign` and
`NonMaxSuppression` included — so no onnxruntime custom-op library is required.
Conversion needs torch + mmdet (VIAME's own mmdet 2.27 / mmcv 1.7.1); *running*
the result needs neither. The same conversion is reachable from a pipeline via
the `convert_to_onnx` process.

Exported graphs reproduce the torch detectors: on HabCam imagery all three
models return the same detection counts with box differences under ~0.2 px and
score differences under 1e-4, and `detector_habcam_test_cfrnn_only.pipe` yields
the same CSV through the `onnx` detector as it did through `netharn`.

### Converting netharn classifiers

`viame.pytorch.netharn_clf_to_onnx` does the same for netharn `ClfModel`
classifiers -- the models behind the kwiver `netharn_classifier` algorithm and
the `refine_detections :refiner:type netharn` reclassifier (in HabCam: 22
substrate classifiers and 2 scallop EfficientNets):

```bash
python -m viame.pytorch.netharn_clf_to_onnx \
    --deployed=models/scallop_efficientnet_four_class.zip \
    --output=models/scallop_efficientnet_four_class.onnx
```

These are plain `InputNorm` + torchvision backbone, so the export needs none of
the detector workarounds. The batch axis is left dynamic so callers can batch
chips the way netharn's predictor does, the `InputNorm` stays inside the graph
(the sidecar therefore asks only for a `1/255` rescale), and the softmax is
baked in -- for the flat class trees these models use, netharn's
`hierarchical_softmax` is exactly a plain softmax. A model with a real class
hierarchy is refused rather than silently mis-normalized.

Note the sidecar records `resize_mode: letterbox`: netharn's classifier dataset
uses `kwimage.imresize(..., letterbox=True)`, which preserves aspect ratio and
pads, unlike the squash resize the detectors use.

Measured on an RTX 5000 with torch 2.12 / onnxruntime-gpu 1.23.2, EfficientNetV2-S
at 224x224, versus the eager PyTorch model (ms per batch):

| batch | torch CPU | onnx CPU | torch GPU | onnx GPU |
| ---: | ---: | ---: | ---: | ---: |
| 1  |  44.5 |  16.8 | 13.8 |  4.3 |
| 4  |  85.8 |  45.0 | 14.0 | 11.1 |
| 8  | 161.8 |  94.5 | 14.9 | 20.3 |
| 16 | 363.3 | 204.9 | 26.6 | 38.0 |
| 32 | 959.6 | 470.0 | 51.2 | 75.9 |

ONNX wins on CPU everywhere (1.7-2.6x). On GPU it wins at small batches (3.2x at
batch 1, 1.3x at batch 4) and loses beyond that (0.7x at batch 16-32): torch is
launch-bound at small batch, and better at saturating the GPU at large batch.
Since `netharn_classifier` runs one frame per call and the netharn refiner
defaults to `batch_size` 2-4, both HabCam uses sit on the winning side.

### Running classifiers: `onnx_classifier` and the `onnx` refiner

Two kwiver algorithms consume these graphs, replacing `netharn_classifier` and
the `netharn` refiner respectively:

* **`onnx_classifier`** (an `image_object_detector`, like the netharn one it
  replaces) classifies a whole frame and emits a single detection covering it,
  carrying every class probability. `negative_class` renaming is preserved.
* **`onnx`** (a `refine_detections`) crops a chip per input detection,
  classifies it, and rewrites the detection's type. The chip geometry, area and
  border filters, target-scale normalization and prior/taxonomy blending are
  ports of `viame.pytorch.netharn_refiner` and behave identically.

Both share `onnx_clf_predictor`, whose `letterbox_resize` is a pixel-exact port
of `kwimage.imresize(..., letterbox=True)`. Two details there are load-bearing:
kwimage picks **`INTER_AREA` when shrinking** (`INTER_LANCZOS4` when growing),
and it **rounds** the pad offset rather than flooring. Getting either wrong is
survivable on chips but not on whole-frame classification, where a 1360x1024
frame is shrunk ~6x -- an early version using `INTER_LINEAR` throughout changed
the top class on 5 of 66 substrate classifications.

Frame scores match netharn exactly because for the flat class trees these
models use, `CategoryTree.decision(..., criterion='entropy')` reduces to plain
argmax / max-probability.

Verified end to end on the HabCam example imagery, against the netharn path:

| pipeline | rows | top-class mismatches | max prob diff |
| --- | ---: | ---: | ---: |
| `detector_habcam_substrate.pipe` (22 classifiers) | 66 vs 66 | 0 | 1e-6 |
| `detector_habcam_scallop_four_class.pipe` (reclassifier) | 37 vs 37 | 0 | 1e-6 |

(1e-6 is the CSV's print precision, not a real difference.)

### Converting darknet YOLO models

`viame.pytorch.darknet_to_onnx` converts a darknet `.cfg`/`.weights` pair,
using the bundled `darknet2onnx` to rebuild the graph in PyTorch:

```bash
python -m viame.pytorch.darknet_to_onnx \
    --cfg=models/fish_yolo_v2.cfg --weights=models/fish_yolo_v2.weights \
    --labels=models/fish_yolo_v2.lbl --output=models/fish_yolo_v2.onnx \
    --resize-mode=letterbox
```

Three things here are easy to get wrong and cost real accuracy:

* **These models are BGR.** VIAME's C++ darknet detector hands OpenCV's native
  BGR straight to the network, unlike the netharn-derived models. Feeding RGB
  produces plausible-looking boxes that match the darknet detector *not at all*
  (0 of 5 on a test chip, versus 5 of 5 with BGR).
* **`--resize-mode` must mirror the pipe's `resize_option`.** `squash` (the
  default) matches `resize_option=chip`, where the chip already is the network
  size; `letterbox` matches `maintain_ar`, where a whole frame is fitted
  preserving aspect. The `darknet` decoder maps boxes back through whichever.
* The vendored exporter's own `export_darknet_to_onnx` is not used: it pins
  opset 11 and leaves `dynamo` at torch's default, which on torch>=2.9 fails
  the opset downconvert and silently writes a graph with no weights inline.

VIAME patches three gaps in `darknet2onnx` (`packages/patches/darknet-to-pytorch-onnx`),
without which neither HabCam model converts: the YOLOv2 `[region]` head was
unimplemented, `[reorg]` used Python-2 float division in a `view()`, and the
`[route]` builder asserted that a multi-way concat lists the preceding layer
first (YOLOv7-tiny does not).

Agreement with the C++ `darknet` detector is close but, unlike the netharn
conversions, not exact. On the HabCam example imagery, counting detections
above 0.05:

| model | pipeline | darknet | onnx | strong dets matched IoU>0.5 |
| --- | --- | ---: | ---: | ---: |
| `scallop_yolo_v7_one_class` | `detector_habcam_test_yolo_only` | 7 | 8 | 7/7 |
| `fish_yolo_v2` | `detector_habcam_scallop_and_flatfish` | 20 | 21 | 17/20 |

On a single chip with no tiling, v7 matches 10/10 with scores equal to four
decimals. The YOLOv2 region head reproduces confident detections closely (a
0.9866 skate becomes 0.9855, box within ~5 px) but diverges more in the low
score range; treat `fish_yolo_v2` as a close-but-not-bit-exact port.

# Epipolar stereo matching as a single ONNX graph

This plugin reimplements VIAME's stereo correspondence methods as single,
self-contained `.onnx` files. Two of the three stereo-measurement methods are
covered (the third, the depth-map matcher, is not):

1. **Method 1 — NCC** (`epipolar_template_matching`, the "regular computer
   vision" method in
   [`measurement_from_annotations_template.pipe`](../../configs/pipelines/measurement_from_annotations_template.pipe)):
   epipolar candidate generation + NCC template matching.
2. **Method 2 — DINO + NCC** (`epipolar_descriptor_type=dino` in
   [`add-ons/dino/measurement_from_annotations_ncc_dino.pipe`](../../configs/add-ons/dino/measurement_from_annotations_ncc_dino.pipe)):
   the same epipolar matching, but DINOv2 dense features first pick the top-K
   semantically similar candidates and NCC refines among them. The DINOv2 ViT is
   baked into the graph.

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
| `epipolar_matcher.py` | Method 1: `EpipolarMatcher` / `EpipolarMeasurer` (the graph). |
| `epipolar_dino_matcher.py` | Method 2: `EpipolarDinoMatcher` / `EpipolarDinoMeasurer` (DINO top-K + NCC). |
| `triangulate.py` | `triangulate_fast_numpy` (host, exact) and `triangulate_fast_torch` (graph). |
| `calibration_io.py` | Load `.npz/.json/.yml/.mat/`dir` → calibration tensors. |
| `geometry_numpy.py` | NumPy intrinsics map/unmap/project for host normalization + RMS. |
| `export_stereo_mapping.py` | Export either model to one `.onnx` (+ onnxruntime verification). |
| `run_epipolar_onnx.py` | End-to-end driver: calib + images + keypoints → matches + measurements. |

## Usage

```bash
# Method 1 (NCC). template_size / num_samples become graph constants.
python export_stereo_mapping.py --model match   --out epipolar_match.onnx
python export_stereo_mapping.py --model measure  --out epipolar_measure.onnx \
       --template-size 25 --num-samples 5000

# Method 2 (DINO + NCC). The DINOv2 ViT is baked in and the image size is FIXED
# at export (--height/--width must match your camera resolution).
python export_stereo_mapping.py --model dino         --out epipolar_dino.onnx \
       --height 1080 --width 1920 --dino-model dinov2_vitb14 --dino-top-k 25
python export_stereo_mapping.py --model dino-measure  --out epipolar_dino_measure.onnx \
       --height 1080 --width 1920

# Run: match head/tail pairs and measure length (same driver for all models)
python run_epipolar_onnx.py \
    --onnx epipolar_match.onnx \
    --calibration calibration_matrices.npz \
    --left-image left.png --right-image right.png \
    --points 812,455 1033,470 --pairs \
    --min-disparity 7 --max-disparity 724 \
    --template-threshold 0.2 --uniqueness-ratio 0.85
```

Method-1 graph inputs are `left_gray, right_gray, points_left, K_left,
dist_left, R_left, t_left, K_right, dist_right, R_right, t_right, min_depth,
max_depth`; method 2 replaces the grayscale images with color `left_rgb,
right_rgb` (`[3, H, W]` RGB in `[0, 255]`). Outputs are `right_points,
best_score, second_score` (+ `points_3d` for the `*measure` models). Point count
is a dynamic axis; image size is dynamic for method 1 and fixed for method 2.
The driver auto-detects color vs grayscale inputs.

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
is unavailable), optional `scipy` (.mat). Export needs `torch`; method 2
additionally downloads the DINOv2 backbone via `torch.hub` (or `--dino-weights`
for a local checkpoint). Models export with the legacy TorchScript exporter
(`dynamo=False`) at opset 18 — it honors dynamic axes, which the dynamo exporter
does not for these graphs.

Validated with torch 2.12 / onnxruntime 1.23 / numpy 2.0:
- Methods 1 & 2 export and onnxruntime reproduces the eager PyTorch outputs
  (NCC/DINO scores to ~1e-8; method-1 matched points and triangulation exact).
- Method 1 synthetic end-to-end matches keypoints to ~0.25 px and the host vs
  in-graph triangulations agree.
- Method 2's DINO stage matches the production `dino_matcher.py` exactly (top-25
  candidate overlap 25/25, identical cosine scores). Positional end-to-end
  validation of method 2 needs real imagery: DINOv2 features are uninformative
  on synthetic patterns (so is the reference).
