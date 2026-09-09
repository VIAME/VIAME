# VIAME lite: dependency disposition

Legend: **vendor** = source copied into `third_party/` and built here;
**pip** = python wheel from a lock file; **drop** = removed on `lite`;
**temp** = present during early phases, removed in the named phase.

## 1. fletch packages (all gone by P7)

| Package | Version today | Used by | Disposition |
|---|---|---|---|
| Eigen | 3.3.9 | vital types, sprokit type traits, mvg, 9 plugin files | **temp** vendored header in P1, **drop** in P6 (`core_types/math`) |
| OpenCV (+contrib ximgproc) | 4.9.0 | arrows/ocv, plugins/opencv, darknet, svm, core | **temp** `find_package` P1..P6, **drop** in P7 (`image_ops`, codecs, python ports); `opencv-python-headless` stays **pip** |
| FFmpeg + x264/x265 | 5.1.2 | arrows/ffmpeg | **temp** `find_package` P1..P3, **drop** in P4 (PyAV); `av` wheel **pip** |
| VXL | git | arrows/vxl, plugins/vxl | **drop** in P3 (`image_ops` v1 + aliases) |
| TinyXML1 | 2.6.2 | CVAT reader | **vendor** (4 files) or replace with a 200-line XML subset reader in P8; decision at P8 |
| pybind11 | 2.13.6 | python bindings | **vendor** header-only, permanent |
| libsvm | 3.11 (+HISTOGRAM kernel patch) | svm, IQR | **vendor** 2 files, permanent |
| CppDB, PostgreSQL client | 0.3.0 / 10.23 | cppdb plugin | **vendor** cppdb behind `VIAME_ENABLE_POSTGRESQL`; libpq from system when enabled (open decision 5) |
| ZLib | 1.2.11 | `camera_rig_io` NPZ, OpenCV, FFmpeg | **temp**; after P7 NPZ reading uses a vendored `miniz` (single file) or moves to python; decide in P7 |
| libjpeg-turbo, libtiff, libpng, libgeotiff | | OpenCV/VXL | **drop** with them; codecs are stb + own TIFF |
| PROJ, SQLite3, GDAL, openjpeg | | VXL/GDAL paths | **drop** (P1: not found, not needed) |
| CPython | 3.12.12 | desktop builds | not built here; system python, or python-build-standalone download on Windows/desktop (P10) |
| GTest | 1.8.1 | tests | **vendor** via FetchContent, tests only |
| OpenBLAS, Boost, log4cplus/log4cxx, Protobuf, Qt, qtExtensions, libkml, GeographicLib, VTK, libxml2, libjson | | pytorch-from-source, vivia, seal, tf | **drop** in P1 |
| CUDA / cuDNN | user-provided | darknet, torch wheels | `find_package(CUDAToolkit)` only while `third_party/darknet` exists; torch wheels carry their own |
| OpenMP, Threads | | evaluate_models, sprokit | system, permanent (no library to ship) |
| OpenSSL | | vertex-ai client | optional `find_package`, only with `VIAME_ENABLE_VERTEX_AI` |

## 2. kwiver components

Copied in P5, replaced in P8; see `lite-removals.md` §5 for the per-component
plan. Summary of what is never copied:

- Arrows: `klv`, `serialize`, `dbow2`, `vtk`, `kpf`, `ceres`, `qt`,
  `super3d`, `geocalc`, `gdal`, `cuda`, `zlib`, `pdal`, `proj`, `uuid`,
  `matlab`, `vxl` (gone in P3), `ffmpeg` (gone in P4).
- `arrows/core`: 35 of 40 registrations (`close_loops*`, `track_features*`,
  `keyframe_selection`, `compute_ref_homography` (unless `stabilize_image`
  is kept in C++ until P7; it moves to python there), `feature_descriptor_io`,
  `filter_tracks`, `texture_mesh`, `uv_unwrap_mesh`, csv readers/writers,
  `metadata_*`, `create_detection_grid`, `example_detector`, `simulator`,
  `filter_features_*`, `interpolate_track_spline`, `transfer_bbox_with_depth_map`,
  `transform_detected_object_set`, `tiled_multifile`, video_input decorators,
  applets `dump-klv`, `transcode`).
- `arrows/ocv`: 22 of 29 registrations; the 7 used ones are ported in P7.
- `arrows/mvg`: everything except `triangulate` and camera helpers.
- Sprokit processes never referenced by any pipeline (25 of 62; list in
  `tasks/phase-05-import-kwiver.md`), `sprokit/processes/{flow,transport,examples,matlab}`,
  `sprokit/src/processes`.
- All `tests/`, `examples/`, `extras/`, `doc/`, `docker/`,
  `tools/kwiver_tool_runner.cxx`, `plugin_explorer`, `pipe-to-dot`,
  `pipe-config`, `track_oracle`.

## 3. Other submodules and packages

| Path | Disposition |
|---|---|
| `packages/darknet` | `third_party/darknet` via `add_subdirectory`, optional (open decision 2) |
| `packages/dive` | not built; release download as today; submodule only for `VIAME_BUILD_DIVE_FROM_SOURCE` |
| `packages/vivia`, `seal-tk`, `itk-modules/*`, `tensorrt`, `tensorflow-libs` | **drop** in P1 |
| `packages/pytorch-libs/*`, `packages/python-utils/pyav` | **pip** wheels from the index built by the wheel CI (P9); submodules leave `lite` in P9 |
| `packages/patches/*` | applied in the wheel CI; `patches/fletch` obsolete in P1 |
| `plugins/pytorch/learn/{pydensecrf,tokencut,cutler,panopticapi}`, `remax/`, `siammask/` (library part), `mdnet/` | **pip** wheels (P9) |
| `plugins/pytorch/netharn`, `srnn`, `minima_loftr`, `torchvision/*_feature_extractor` | first-party under `library/` |
| onnxruntime C++ 1.12.1 download | **drop** in P1; python `onnxruntime` wheel only |

## 4. End state: everything that is not ours

| Vendored source | Size | Why it stays |
|---|---|---|
| pybind11 | header-only | python bindings |
| rapidjson | header-only | JSON read/write (`viame json`, camera rigs, DIVE transforms) |
| stb_image, stb_image_write | 2 headers | png/jpg/bmp codecs |
| baseline TIFF reader/writer | own code, ~1k lines | 16-bit TIFF input; listed here because it implements an external format |
| libsvm | `svm.h`, `svm.cpp` | SVM training / IQR |
| cxxopts | header-only | applet argument parsing |
| miniz (if chosen in P7) | 1 file | NPZ/zip reading |
| tinyxml (unless replaced in P8) | 4 files | CVAT XML |
| cppdb (optional) | small | Postgres descriptor store |
| cpp-httplib (optional) | header-only | vertex-ai client |
| darknet (optional, temporary) | ~30k C/CUDA | until models are ONNX |
| googletest (tests only, FetchContent) | | unit tests |

Python runtime wheels are governed by `python/requirements/*.lock`;
`numpy`, `torch`, `opencv-python-headless`, `av`, `scipy`, `pillow` are the
ones VIAME core code imports directly.
