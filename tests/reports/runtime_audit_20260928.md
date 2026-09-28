# Runtime and cv2-port review — 2026-09-28

## Scope and versions

Reviewed `main-lite` at `9343e5123`, with main source at `b243e5d4d`.
Other ongoing work and existing dirty submodules were left untouched.
This review adds reports only; it does not fix production code.

Tested three separate runtime installations:

* Main native install: `~/Dev/viame/build/install`.
* Current Lite native install: `~/Dev/viame-lite/build/install`.
* An isolated install of the locally retained Linux CPython 3.10 wheel
  `viame-0.23.3-cp310-cp310-manylinux_2_35_x86_64.whl`.

PyPI requests failed with DNS resolution errors. The local 0.23.3 artifact
cannot be certified as the latest published wheel. Its code predates several
recent changes. Dependencies were supplied from the existing installation;
this tests the wheel's own VIAME code, scripts, binaries and configurations,
but does not establish clean-machine dependency completeness or Windows support.
The existing native installations likewise are not fresh builds of their HEADs.

[Machine-readable results](runtime_audit_20260928.json) include wheel and
executable hashes, per-command results, comparison measurements and failures.
Scripts, inputs, logs and outputs are in `/tmp/viame-audit-20260928/`.

## Fixes to prioritize

### 1. Cached calibration cannot finish exporting JSON — main and Lite

`viame calibrate image.png -c corners.npz ...` computes calibration, then
fails with `Object of type int64 is not JSON serializable`. Reproduced in
all three runtimes with eight synthetic stereo checkerboard observations.

`tools/calibrate.py:2463` converts cached arrays to tuples, retaining NumPy
integer elements. Lines 2717–2720 put those elements into the JSON document
without conversion. The ordinary cache writer produces the same NumPy types,
so this also affects reusing a cache written by the tool itself. Convert the
dimensions and grid sizes to Python integers. Write JSON atomically so an
export error cannot leave a truncated calibration file. This fix belongs on
main first.

### 2. The depth port reverses point-cloud colors — current Lite

`tools/depth.py:73` still treats `imageops.read_image()` output as BGR.
That reader now returns RGB. `red = bgr[:, 2]` and `blue = bgr[:, 0]` swap
the colors of every exported point.

On the same synthetic stereo pair, main and Lite produced identical 44,288
3D positions. All red/blue values were reversed: the first point was
`135 35 11` on main and `11 35 135` on Lite. The older 0.23.3 wheel matched
main, confirming a regression in the newer port. Preserve RGB order when
building the PLY color fields and test a deliberately asymmetric color.

### 3. The query pipeline still does not construct on Lite

`query_retrieval_and_iqr.pipe` constructs on the main build but fails on both
Lite runtimes. Its connections at `configs/pipelines/query_retrieval_and_iqr.pipe:29`
feed `track_descriptor_set` and `image_set` into `perform_query`.
`library/descriptors/perform_query_process.cxx:432` declares neither input.
Main's implementation declares and consumes both.

Reconcile the process and pipeline contracts, including `query_from_track.pipe`.
Removing connections without checking how exemplar descriptors and images are
used would risk changing query behavior. This is an existing branch divergence,
not evidence that the latest color-conversion commit introduced it.

### 4. Normal MMCV video iteration indexes the entire video first

`library/video_io/frames.py:235` only takes the cheap forward-seek path when
`_frames` already exists. A new reader's `seek(0)` therefore builds the full
timestamp index. The patched MMCV `VideoReader.__iter__` calls precisely that
operation before yielding its first frame.

Measured on the 242-frame Tules excerpt: `seek(0)` indexed all 242 frames
and took about 0.87 seconds. Long-video startup scales with video length.
Handle initial/zero seeks without indexing; build the timestamp table only
when a backward seek actually needs it. Retain the existing exact frame tests.

### 5. The exact distance-transform helper changes the no-background case

`library/utilities/imageops.py:337` delegates to SciPy without handling an
all-nonzero mask. On a 5×7 all-one mask it returns distances from 1 to 7.81;
the OpenCV precise transform returns a constant approximately `1.84e19`.
SciPy's implicit outside background is not the old operator's behavior.

The SAM3 CPU fallback in `packages/patches/sam3/sam3/model/edt.py:46` reaches
this case for an all-zero input tensor. Define the no-background result
explicitly and test it against the intended CPU/GPU contract. SAM3 model
inference itself was not run in this audit.

### 6. Successful training help/list commands return failure

All three runtimes print valid `train --help` and `train --list` output but
exit with code 1. `tools/train.cxx:1458` and `:1596` explicitly return
`EXIT_FAILURE`. Use success for successful help/list requests. This is also
present on main.

## Remaining output differences

Ten pipelines were run on identical lossless input images in all three
runtimes. Eight produced exactly matching outputs:

* `filter_enhance`, `filter_default`, `filter_split_left_side`,
  `filter_split_right_side`, `filter_debayer`, `filter_normalize_16bit`;
* `detector_simple_hough`: three identical CSV detections;
* `detector_calibration_target`: 54 identical CSV detections.

Two differed:

* **`filter_debayer_and_enhance`:** current Lite differed from main in
  1,198 of 96,000 channel values, each by one. The old wheel matched main.
  A stage-by-stage reproduction found exact white balance, RGB-to-Lab, CLAHE,
  Lab-to-RGB, RGB-to-HSV and saturation scaling. The difference appears in
  `image_kernels.from_hsv` (`library/image_kernels/color.h:633`). The source
  already documents residual rounding differences; this pipeline demonstrates
  that they remain observable. Resolve them before claiming exact image parity,
  particularly for downstream thresholding.
* **`filter_draw_dets`:** main's installed runtime swaps the background's
  red/blue planes; both Lite runtimes preserve the source colors. This was
  reproduced with an empty detection match and rerun with an actual matching
  annotation. Establish the intended main behavior and rebuild the reference
  before changing Lite to reproduce this color swap. Current main source
  explicitly tags its draw result BGR, so the runtime difference needs tracing
  through the installed image-container/writer path.

The `rectify` tool produced identical decoded pixels on an identity calibration.
The `disparity` tool produced identical numeric arrays on a textured shifted
pair. Depth geometry matched exactly; its current Lite colors did not.

## GFIT video result

Current Lite completed a CPU run of `tracker_gfit_groups_v3.pipe` on the
retained Tules video excerpt at 5 Hz, including motion warmup. A fresh CPU
run with the retained `0.23.3+parity.main` test wheel matched every exported
CSV data field: **18 detections and eight track states**. Export comments
were excluded from comparison. Both runs used the same video, pipelines,
models, fixed seeds and explicit CPU classifier settings.

The normal main install and the old local release wheel lack their respective
`gfit_motion` Python modules; they could not serve as GFIT references.
The main comparison therefore uses the previously assembled test wheel,
whose portability limitations are described in
[the earlier GFIT report](gfit_video_parity_20260927.md).

The GPU driver was unavailable (`nvidia-smi` could not communicate with it).
This audit does not extend the previous CUDA matrix, test every GFIT variant,
or establish parity with the latest packages published on PyPI.

## Command-line coverage

Invoked `--help` for every discovered applet: **32 in each Lite runtime**
and **39 in main**. All printed help successfully; `train` returned the
incorrect failure status described above. These counts exclude the dispatcher
help pseudo-command and example command lines.

The existing 124-case applet suite produced:

| Runtime | Passed | Failed | Skipped |
|---|---:|---:|---:|
| Current Lite | 124 | 0 | 0 |
| Local 0.23.3 wheel | 113 | 2 | 9 |
| Main install | 122 | 2 | 0 |

The wheel's failures are its missing add-on catalog and missing default
training configuration. The catalog is already included by the current
`cmake/wheel/contents.txt:98`; regenerate and verify the wheel. The default
training config needs a deliberate packaging/fallback decision, including
its model, template and include dependencies.

Main's two suite failures are pipeline-check differences: it accepts the
test's nonexistent process and lacks Lite's `--no-resolve` option.

An additional **10 CLI configuration/conversion tests passed per runtime**.
One additional test imports current source directly; running it in the older
wheel interpreter failed because that package lacks the new `projection`
module. Rerunning that source test under current Lite passed. It is excluded
from the CLI pass/fail counts.

| Applet | Functional exercise/result |
|---|---|
| `add-ons` | List/install-from-file tests; release-wheel catalog failure |
| `configs`, `convert` | Extraction, strict errors and annotation conversions passed |
| `csv`, `json`, `resample`, `score` | Data operations and validation tests passed |
| `inspect`, `run`, `runner`, `pipeline` | Image/video/list/archive/pipeline tests; suite differences above |
| `explore-config` | Read a real configuration successfully |
| `extract` | Extracted video frames successfully using FFmpeg |
| `plot` | Generated detection plots successfully |
| `metadata` | Exported an image survey CSV successfully |
| `ensemble` | Completed a small two-input fusion optimization |
| `mosaic` | Wrote an identity-homography image mosaic |
| `rectify`, `disparity`, `depth` | Real stereo inputs; output comparisons above |
| `calibrate` | Real synthetic observations; cached export failed in all runtimes |
| `segment` | Inventoried a valid annotated image dataset; model inference untested |
| `register` | Metadata mode completed on a small image survey |
| `index` | List command completed; database ingestion/search untested |
| `monitor` | Status command completed; no monitoring daemon/email started |
| `gpu` | Ran and reported the unavailable GPU |
| `train` | Argument/config/list paths exercised; no training job run |
| `registry-dump`, `pipe-check` | Generated complete registry/configuration reports |
| `3d` | Blocked by missing Open3D, including with `--no-dense` |
| `view` | Fails because `vpView` is absent; no GUI validation |
| `search` | Rejects the absent search database; GUI/query execution unverified |

Main-only KWIVER applets were checked for help/startup, not full reconstruction
or KLV workflows.

## Pipeline packaging checks

The local release wheel's `pipe-check --all` reports **100 constructible
entries and six errors** out of 106 entries. This is construction coverage,
not proof of successful model configuration or output correctness.

| Failing entry | Main build comparison |
|---|---|
| `query_retrieval_and_iqr.pipe` | Constructs on main; Lite input-port regression above |
| `register_multimodal_unsync_ocv.pipe` | Missing process on main too |
| `train_aug_add_motion_and_color_freq.pipe` | Invalid third merge input on main too |
| `train_aug_intensity_hue_motion.pipe` | Invalid third merge input on main too |
| `train_aug_warp_ir_to_eo.pipe` | Unfinished `TODO` syntax on main too |
| `utility_link_detections_default.pipe` | Missing `common_seamap_tracker_v2.5.pipe` on main too |

Current Lite reports 253 constructible entries out of 292. Its other errors
include standalone include fragments, unexpanded templates and unavailable
optional processes, so the 39 error entries are not 39 runnable-pipeline
regressions. Before publishing, validate the selected entry points and their
include closure; omit unfinished templates from the runnable selection.

## Further optimization

`library/utilities/clustering.py` materializes a float64 `N×K×D` distance
array, recomputes assignments/distances, and recomputes all previous-center
distances during k-means++ initialization. A 65,536-sample RGB, 16-center,
10-iteration check took **1.07 s**, versus **0.080 s** for installed OpenCV
with one thread. This is one local microbenchmark, not a general throughput
claim; seeds/initial centers differ between implementations.

Keep the running nearest-center distance during initialization and compute
assignments in bounded-size blocks. That avoids quadratic work in the number
of seeded centers and large temporary allocations. Preserve the explicit seed
contract and validate both labels/compactness and augmentation behavior.

Prioritize the initial video seek and color/port-contract bugs before adding
more GPU kernels. The measured defects affect ordinary CPU workflows and
package usability regardless of CUDA availability.
