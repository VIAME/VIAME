# Phase 7: drop OpenCV from the C++ build

Goal: no `#include <opencv2/...>` under `library/` or `tools/`;
`find_package(OpenCV)` gone. Python keeps `cv2` (open decision 1).
References: lite-removals.md §2. Independent of P6; if done before P6,
P7-T04 ports to Eigen-free math anyway where it touches geometry.

### P7-T01 Inventory and golden recording
Depends: P5-T07
Do:
- `git grep -ln "opencv2/" -- library tools > design/lite-opencv-uses.txt` with, per file, the OpenCV API areas used (2.1 bridge, 2.2 codecs, 2.3 imgproc, 2.4 calib3d, 2.5 FileStorage, 2.6 ml/features/segmentation, drawing).
- Record goldens: every pipeline using an `ocv_*` impl or OpenCV-backed process on `pipelines_test_data`; codec fixtures (png 8/16-bit gray/rgb/rgba, jpg, bmp, tiff 8/16-bit strips uncompressed/LZW/PackBits, one tiled tiff expected to fall back); calibration YAML/XML files from `tests/data`, `examples`, add-ons with their parsed matrices as JSON.
Done when:
- Inventory and `tests/golden/{opencv,codecs,calib}` committed.

### P7-T02 Codecs
Depends: P7-T01
Do:
- `third_party/stb` (stb_image, stb_image_write, pinned versions). `library/video_io/codecs/tiff.{h,cxx}` per lite-removals.md §2.2 and open decision 7 scope. `image_io` impl `core` (aliases `ocv`, `vxl`) using them; `pil` python image_io fallback registered too. Writer: png/jpg/bmp via stb, tiff via own writer (uncompressed).
- Golden: decode byte-identical to OpenCV for png/bmp/tiff; jpg within decoder tolerance (max abs 8, mean < 1).
Done when:
- Golden passes; `image_writer` pipelines produce readable files.

### P7-T03 `image_ops` v2
Depends: P7-T01
Do:
- Add the functions in lite-removals.md §2.3 table: colour conversions (RGB/BGR/GRAY/Lab/HSV, Bayer demosaic bilinear + VNG-lite), resize (nearest/bilinear/area), warp affine/perspective/remap with border modes, filter2D/Sobel/gaussian, binary morphology with structuring elements, connected components + contour tracing, contour area/convex hull/min-area-rect/bounding rect, histogram/normalize/minmax/CLAHE, split/merge/concat, template matching NCC, drawing (rect/line/circle/filled poly/text with embedded bitmap font).
- Unit tests vs hand values; golden vs OpenCV on fixtures with per-function tolerances (resize bilinear <= 1, CLAHE <= 2, demosaic <= 3, warps <= 1 away from borders).
Done when:
- Tests pass; `library/image_ops` has no OpenCV include.

### P7-T04 Port `image_processing` and `object_detectors` C++
Depends: P7-T02, P7-T03
Do:
- `debayer`, `enhancer` (CLAHE in Lab), `color_correction`, `hue_shift`, `convert_color`, `split_image_*`, `merge_images` (ocv impl -> `core`, alias `ocv`), `warp_image_ocv` (alias), `draw_detected_object_set` (alias `ocv`), `refine_detections_write_to_disk` (`ocv_write`), `windowed_*` chipping, `darknet_custom_resize`, `detect_calibration_targets` (chessboard detection -> python impl, same name), `diff_of_gauss`, `canny_edge`, `detect_in_subregions_process`, `hough_circle`, `detect_heat_map`, `ocv_3frame_differencing`, `stabilize_image` + `estimate_homography` + SIFT (python impls, same names). Remove `arrows/ocv/image_container.h` uses.
- Golden per pipeline from P7-T01.
Done when:
- `git grep opencv2/ -- library/{image_processing,object_detectors}` empty; GOLDEN passes.

### P7-T05 Calibration file IO
Depends: P7-T01
Do:
- `library/file_io/opencv_yaml.{h,cxx}`: reader/writer for the OpenCV YAML subset, XML read-only, per lite-removals.md §2.5. Replace every `cv::FileStorage` use (`calibrate_single_camera`, `calibrate_stereo_cameras`, `camera_rig_io`, measurement processes, `stereo_utils` if C++).
- Golden: parse every fixture to JSON equal to the recorded one; write-then-read round trip.
Done when:
- Golden passes; no `FileStorage` in `library`.

### P7-T06 Port `measurement` to python where calib3d is needed
Depends: P7-T05
Do:
- Python implementations registered under the existing names: `calibrate_stereo_cameras`, `ocv_optimize_stereo_cameras`, `ocv_stereo_disparity` (SGBM via cv2), `filter_stereo_feature_tracks`, `ocv_calibrate_single_camera` process, `measure_using_stereo`/`compute_measurements` keep C++ but use `core_types/math` projection + `image_ops` template matching; `pair_stereo_*` C++ loses cv2 rectification by consuming precomputed rectification maps produced by the python calibration step (stored in the rig file). Delete `arrows/ocv/camera_intrinsics.h` uses.
- Golden: `measurement_*` pipelines.
Done when:
- `git grep opencv2/ -- library/measurement` empty; GOLDEN passes.

### P7-T07 Port `classifiers`, `segmentation`, `descriptors`, `training`, `evaluation`
Depends: P7-T04
Do:
- `add_keypoints_from_mask` (contours from `image_ops`), `refine_detections_grabcut` -> python, `refine_detections_watershed` -> alias of python `ocv_watershed`, `classify_fish_hierarchical_svm` / `adaboost` / `gabor` / `hog` / `kmedians` per open decision 6 (python or `removed.json`), `windowed_refiner`/`windowed_trainer` chipping, `train_detector_svm` image handling, `plot_metrics` -> python matplotlib called from `viame score`, `evaluate_models` if it touches OpenCV, `image_viewer` process -> python or removed.
Done when:
- `git grep opencv2/ -- library` returns nothing outside `object_detectors/darknet*` (handled next); GOLDEN passes.

### P7-T08 Darknet without OpenCV
Depends: P7-T07
Do:
- `third_party/darknet` built with `ENABLE_OPENCV=OFF`; `darknet_detector` feeds `image_ops`-resized buffers; `darknet_trainer` unchanged (shells to the binary). If the fork's `-DOPENCV` paths are required for training image loading, add stb-based loading to the fork on branch `viame/lite`.
Done when:
- `detector_darknet*` goldens pass.

### P7-T09 Switch OpenCV off
Depends: P7-T06, P7-T08
Do:
- Remove `find_package(OpenCV)`, `VIAME_ENABLE_OPENCV`, `ZLIB` (replace NPZ reading in `camera_rig_io` with `third_party/miniz` or move to python; decide and note). `ldd` of every plugin: no `libopencv_*`, `libz` only if miniz not chosen.
- DIVE smoke incl. calibration import and stereo measurement.
Done when:
- Build from clean on a machine without OpenCV dev packages; BASELINE, CRITICAL, GOLDEN pass; smoke passes.
