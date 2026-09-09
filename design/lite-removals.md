# VIAME lite: dependency removal designs

Each section: what the dependency is used for today, what replaces each
use, what is registered under which names, and how fidelity is checked.
Order is the execution order.

## 1. VXL (Phase 3)

Used by `arrows/vxl` (kwiver, ~16k lines) and `plugins/vxl` (~2k lines).
Only the following registrations are referenced by pipelines:

| Name (pipelines) | Kind | Replacement | New name / alias |
|---|---|---|---|
| `vxl_convert_image` (158) | image_filter | `image_ops`: format conversion (byte/uint16/float), channel select/merge, scale factor, percentile normalisation, force 1 or 3 channels | `convert_image`, alias `vxl_convert_image` |
| `vxl` image_io (139) | image_io | Until P7: alias to `ocv` image_io. After P7: `core` codecs impl | alias `vxl` |
| `vxl_average` (35) | image_filter | `image_ops`: window / cumulative / exponential temporal averaging, float accumulator, output cast | `average_frames`, alias `vxl_average` |
| `vxl_color_commonality` (5) | image_filter | `image_ops`: colour histogram commonality map, configurable bins/smoothing | `color_commonality`, alias `vxl_color_commonality` |
| `vxl_morphology` (2) | image_filter | `image_ops`: erode/dilate/open/close, rectangular and elliptical elements | `morphology`, alias `vxl_morphology` |
| `vxl_threshold` (1) | image_filter | `image_ops`: absolute / percentile threshold | `threshold`, alias `vxl_threshold` |
| `vxl_enhancer` | image_filter | Same as `ocv_enhancer` (P7 makes that in-house too) | alias of `ocv_enhancer` |
| `vxl_white_balancing` | image_filter | `image_ops`: grey-world / percentile white balance | `white_balance`, alias `vxl_white_balancing` |
| `format_images_srm` | process | Rewritten on `image_ops` | same name |
| `kw_archive_writer` (6) | process | Open decision 3 | removed.json or same name |
| `vxl` bundle_adjust / estimate_* / triangulate_landmarks | not in any pipeline | not replaced | removed.json |

Config keys of each `vxl_*` impl are copied verbatim into the replacement.
Golden tests: every `filter_*.pipe` and `common_default_input*.pipe` variant
that uses a `vxl_*` filter runs on `pipelines_test_data` and compares output
images (max abs diff, mean abs diff thresholds per filter, recorded in the
test).

`image_ops` v1 scope (C++, on `vital::image`, templated over pixel type):
type conversion, channel ops, per-pixel scale/offset, percentiles,
temporal averaging, box/gaussian blur, erode/dilate, colour histogram,
threshold, white balance. Golden data recorded from the VXL build before
it is switched off.

## 2. OpenCV (Phase 7)

Used by `arrows/ocv` (16k lines), most of `plugins/opencv`, `plugins/core`
(9 files), `plugins/darknet`, `plugins/svm`. Grouped by API area:

### 2.1 Image container bridge

`arrows/ocv/image_container.h` (28 include sites). Replacement: none
needed; `vital::image` is the only C++ image type, and the python bridge is
the numpy view already exposed by the bindings. Every `cv::Mat`-typed
helper signature becomes `vital::image_of<T>`.

### 2.2 Codecs (`imread`/`imwrite`, `ocv` image_io, `image_writer`)

`video_io/codecs/`: vendored `stb_image.h` + `stb_image_write.h` (png, jpg,
bmp), in-house baseline TIFF reader/writer (strips, 8/16-bit, 1/3/4
channels, uncompressed / LZW / PackBits). Registered as image_io `core`
with aliases `ocv`, `vxl`. Anything the codecs do not handle falls back to
image_io `pil` (python). Golden: byte-identical decode against OpenCV for
png/bmp/tiff; jpg within decoder tolerance.

### 2.3 imgproc primitives -> `image_ops` v2

| OpenCV calls | `image_ops` function | Used by |
|---|---|---|
| `cvtColor` (BGR/RGB/GRAY/Lab/HSV, Bayer) | `color::convert`, `color::demosaic` | debayer, enhancer, colour correction, hue shift, convert_color |
| `resize` (linear/area/nearest) | `resample::resize` | windowed detector/refiner/trainer chipping, darknet resize, descriptors |
| `warpPerspective`, `warpAffine`, `remap` | `warp::perspective`, `warp::affine`, `warp::remap` | warp_image, stabilisation, rectification |
| `GaussianBlur`, `blur`, `Sobel`, `filter2D`, `addWeighted` | `filter::*` | enhancer, DoG, canny, sharpening |
| `threshold`, `erode`, `dilate`, `getStructuringElement`, `findContours`, `contourArea`, `convexHull`, `minAreaRect`, `boundingRect` | `binary::*`, `contours::*` | masks, keypoints from mask, calibration target detector, ellipse proposal |
| `calcHist`, `normalize`, `minMaxLoc`, CLAHE | `hist::*` | percentile eq, enhancer, colour commonality |
| `split`, `merge`, `hconcat`, `vconcat` | `channel::*`, `layout::*` | split/merge images |
| `matchTemplate` (CCOEFF_NORMED) | `match::template_ncc` | epipolar template matching, stereo pairing |
| `rectangle`, `line`, `circle`, `putText` | `draw::*` with an embedded 8x13 bitmap font | draw_detected_object_set, ocv_write, debug output |

### 2.4 calib3d and geometry -> python (`viame.measurement`)

`calibrateCamera`, `stereoCalibrate`, `stereoRectify`, `initUndistortRectifyMap`,
`undistortPoints`, `projectPoints`, `findChessboardCorners`, `solvePnP`,
`StereoSGBM`, `findHomography` (RANSAC), `estimateAffine*`, SIFT/ORB
matching. The C++ algorithms and processes that wrap these
(`calibrate_stereo_cameras`, `optimize_stereo_cameras`,
`compute_stereo_disparity`, `filter_stereo_feature_tracks`,
`calibrate_single_camera*`, `detect_calibration_targets`,
`stabilize_image`, `ocv_SIFT`, `estimate_homography`) are re-registered as
python implementations under the same names with the same config keys.
The C++ measurement code that only needs projection/triangulation keeps
working on `core_types/math` (P6).

### 2.5 Calibration files

`cv::FileStorage` reads/writes the OpenCV YAML/XML matrix format for
intrinsics, extrinsics, rig files. `file_io/opencv_yaml.{h,cxx}`: reader and
writer for the `%YAML:1.0` subset (`!!opencv-matrix` with rows/cols/dt/data,
scalars, sequences, nested maps). Tested against every calibration file in
`tests/data`, `examples/`, and the stereo add-ons. XML variant supported
read-only.

### 2.6 ml / features / segmentation -> python or removal

| Use | Today | Replacement |
|---|---|---|
| `cv::ml::Boost` in `adaboost_classifier`, `iqr_session_adaboost`, `process_query_adaboost` | 1 pipeline | Open decision 6: python sklearn port under the same name, or removed.json |
| `classify_fish_hierarchical_svm` | not in any pipeline | Open decision 6 |
| `gabor_features`, `hog_features`, `kmedians` | descriptors, not in pipelines | Python (`skimage`/numpy) or removed.json |
| `grabCut`, watershed | `ocv_grabcut`, `ocv_watershed` | python (`ocv_watershed` already exists in python; grabcut ported to python cv2) |
| `hough_circle`, `detect_heat_map`, `ocv_3frame_differencing`, GMM motion | 1-4 pipelines each | python implementations, same names |
| `plot_metrics` (evaluation plots) | C++ OpenCV drawing | python matplotlib in `evaluation/python`; `viame score` shells to it |
| `image_viewer` process | 1 pipeline | python (`cv2.imshow`) or removed.json |

If open decision 1 becomes "no cv2 in python either", 2.4 and 2.6 are
re-done on numpy/scipy/torch: calibration via a small bundle adjuster in
scipy, SGBM via a numpy/torch implementation, SIFT via `kornia`. That is a
separate, larger effort and is not scheduled here.

### 2.7 Final flip

`find_package(OpenCV)` removed; `VIAME_ENABLE_OPENCV` deleted; `ldd` of
every plugin shows no `libopencv_*`. `opencv-python-headless` remains in
the python lock files.

## 3. FFmpeg (Phase 4)

Used only by `arrows/ffmpeg`: `video_input` impl `ffmpeg` (aliases
`vidl_ffmpeg` in older configs), `video_output` impl `ffmpeg`, and the
`ffmpeg_init` helper. KLV/metadata paths are unused by VIAME.

### 3.1 Reader

`video_io/pyav_video_input.py`, class registered as `video_input`
`ffmpeg` (aliases `vidl_ffmpeg`, `pyav`). Config keys copied from the
kwiver impl: `filter_desc`, `start_at_frame`, `stop_after_frame`,
`frame_rate` override, `use_misp_timestamps` (ignored, logged), `sync`
options. Semantics that must match: frame numbering from 1, timestamp from
pts converted to microseconds, `seek_frame` accuracy (decode-from-keyframe),
`end_of_video`, `good_frame`, `num_frames` when the container provides it.
Frames decoded to RGB8 (or RGB16 for >8-bit sources) into `vital::image`
via numpy without copy.

### 3.2 Writer

`video_io/pyav_video_output.py` registered as `video_output`
`ffmpeg`. Keys: `codec` (default `h264`), `bit_rate`, `pixel_format`
(default `yuv420p`), `frame_rate`, container from extension. `_finalize`
flushes and closes (see the existing writer-finalize note in the tree).

### 3.3 Fallback

`ffmpeg_cli` video_input: subprocess `ffmpeg -i ... -f rawvideo -pix_fmt gbrp -`
using the binary from the `imageio-ffmpeg` wheel. Used if PyAV import fails,
which logs a warning, or when `use_cli` is set on the PyAV reader. `gbrp`
rather than `rgb24` because vital stores images planar: handing `Image` an
interleaved array costs a per-pixel walk (see the finding in STATUS.md).
The filter chain and its swscale flags are the PyAV reader's, and
presentation times come from `showinfo`, so the pixels and the times are the
same through either path.

### 3.4 If python `av` is not allowed (open decision 1)

The remaining option is a C++ demuxer/decoder, which means re-adding a
codec library. Out of scope; would need a decision.

### 3.5 Final flip

`KWIVER_ENABLE_FFMPEG=OFF` (P4, kwiver still a submodule), `arrows/ffmpeg`
not copied in P5, `find_package(FFMPEG)` gone. Benchmark recorded in
STATUS.md.

## 4. Eigen (Phase 6)

Used by 35 of 151 `vital/types` files, `arrows/mvg`, `arrows/core`
(association matrix, transforms), 9 `plugins/` files, and the python
bindings (numpy <-> Eigen casters).

### 4.1 `core_types/math`

```
vector.h      vector<N,T>: +,-,*,dot,cross(3),norm,normalized,homogeneous
matrix.h      matrix<R,C,T>: +,-,*,transpose,identity,determinant(2,3,4),inverse(2,3,4),
              block, row/col access; dynamic matrix<T> for DLT systems
rotation.h    quaternion rotation_<T>: from/to matrix, axis-angle, yaw-pitch-roll,
              compose, inverse, rotate vector
covariance.h  symmetric NxN storage
decomp.h      jacobi_svd (small dense), jacobi_eigen_symmetric, cholesky, solve_least_squares
similarity.h  scale + rotation + translation
```

Typedefs keep the vital names (`vector_2d`, `vector_3d`, `matrix_3x3d`,
`rotation_d`, `homography` = `matrix_3x3d`), so most consumers change only
includes and the few expression-template idioms (`.array()`, `.cast<>()`,
`Eigen::Map`). Unit tests compare against values recorded from the Eigen
build (`tests/golden/math/*.json`).

### 4.2 Consumers

| Consumer | Change |
|---|---|
| `vital/types/{camera*,landmark*,feature*,homography*,transform_2d,similarity,rotation,covariance,point,vector,matrix,polygon,bounding_box,geo_*}` | Port to `core_types/math`; `geo_*` are not copied in P5 |
| `arrows/mvg/triangulate` | DLT with `jacobi_svd`; inhomogeneous variant with `solve_least_squares` |
| `arrows/mvg` bundle adjust, camera optimisation, necker reverse, initialise cameras | Not copied; `calibrate_cameras_from_tracks` and `optimize_stereo_cameras` move to python (scipy least_squares) under the same names |
| `arrows/core/compute_association_matrix_from_features`, `transform_detected_object_set`, `interpolate_track` | Port to `matrix<T>` |
| `plugins` 9 files (`measurement_utilities`, `pair_stereo_*`, `camera_rig_io`, `calibrate_cameras_from_tracks_process`, `read_transform_homography_json`, ...) | Port |
| python bindings | numpy casters for `vector<N,T>` / `matrix<R,C,T>` written by hand (buffer protocol) |

### 4.3 Final flip

`third_party/eigen` deleted; `git grep Eigen::` empty.

## 5. Kwiver-derived infrastructure (Phases 5 and 8)

P5 copies; P8 replaces. After P8 every file under `library/` is code we
own and understand, and the remaining vendored sources are those in
`lite-dependencies.md` §4.

| Component | P5 (copy) | P8 (replace) |
|---|---|---|
| `vital/types` | -> `core_types/` pruned to the ~40 used headers | Already re-based on own math in P6; P8 removes dead members and the `any`/`attribute_set` generics nothing uses |
| `vital/config` | -> `algorithm_framework/config/` | Kept (it is the `.conf`/`.pipe` semantics); `kwiversys` calls replaced by std |
| `vital/logger` + `logger_plugins` | -> `algorithm_framework/logger/` | Rewritten: ~300 lines, level from env, stderr/file sinks, same `LOG_*` macros |
| `vital/plugin_management` | -> `algorithm_framework/plugin/` | Replaced by `algorithm_framework/registry/`: static registration from `register_builtins()`, factory map keyed by interface + name, alias table, lazy python factories by import path, `registry-dump` support. No directory scanning. Optional ~150-line dlopen hook for external plugins per open decision 10 (`lite-build-system.md` §4) |
| `vital/algo` | -> `algorithm_framework/algo/` | Interfaces kept; `algorithm.txx` / `pluggable_macro_magic.h` simplified while keeping the `PLUGGABLE_IMPL`/`PARAM` macro surface VIAME plugins use |
| `vital/util`, `vital/io`, `vital/range`, `vital/exceptions` | -> `algorithm_framework/{util,io,range,exceptions}` | Pruned to used helpers; `kwiversys` replaced by `std::filesystem`, `std::regex`, own `dynamic_library` |
| `vital/kwiversys` | -> `third_party/kwiversys` | Deleted |
| `vital/internal/cereal` | -> `third_party/cereal` | Deleted; `file_io/json.{h,cxx}` thin rapidjson helper used by `camera_rig_io`, `read_transform_homography_json`, `tools/json.cxx` |
| `vital/applets` | -> `algorithm_framework/applets/` | Kept; `cxxopts` stays vendored |
| `sprokit/pipeline`, `pipeline_util`, schedulers, adapters, runner | -> `pipeline_framework/` | Trimmed: remove unused process/port flag paths, `process_cluster` only if no `.pipe` uses `cluster` (checked by `pipe-check`), remove `kwiversys`; add pipeline-level alias resolution for process types |
| `sprokit/processes/core` (37 used) | -> functional dirs | Kept |
| `python/kwiver/*` bindings | -> `algorithm_framework/`, `pipeline_framework/`, `core_types/` (alongside the C++) | Rewritten by hand with pybind11 in one extension `viame._core` linking `libviame`: types (numpy buffer views), config, algorithm trampolines for the 18 interfaces python implements, process/datum/port API, `pythread_per_process` scheduler, declarative lazy registration replacing `__vital_algorithm_register__`/`__sprokit_register__` eager imports. castxml/pygccxml gone |
| kwiver CMake macros | -> `cmake/kwiver_compat/` | Deleted; `viame_*` helpers only |
