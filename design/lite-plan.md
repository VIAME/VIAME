# VIAME lite: plan

## 1. Goals

1. One CMake project: `cmake -S . -B build && cmake --build build` builds
   all first-party C++ and python. No `ExternalProject`, no nested
   re-invocation, no `packages/fletch` or `packages/kwiver` submodules.
2. Zero third-party C++ library dependencies in the end state. Dropped, in
   this order: VXL, FFmpeg, Eigen, OpenCV, then the kwiver-derived
   infrastructure itself. What remains is a short list of vendored
   single-header or two-file sources (`lite-dependencies.md` §4) plus the
   Python interpreter.
3. Source organised by function under `library/`: `core_types`,
   `algorithm_framework`, `pipeline_framework`, `image_ops`, `video_io`,
   `file_io`, `image_processing`, `object_detectors`, `object_trackers`,
   `classifiers`, `segmentation`, `descriptors`, `measurement`, `training`,
   `evaluation`, `utilities`, `examples`. No directory named after a
   dependency; optional dependencies gate files, not directories.
4. Dependencies are removed one at a time, each behind its own phase, each
   leaving the build green and the compatibility contract intact.
5. Pipelines, add-on packages, DIVE, and `viame <applet>` keep working:
   every registered algorithm implementation name, process type, applet,
   and config key survives (aliases for renames, an explicit allow-list for
   removals).

## 2. End state

```
C++ build inputs:   C++17 compiler, CMake >= 3.21, Python 3 (headers + interpreter),
                    optionally CUDA toolkit (only while third_party/darknet exists)
Vendored sources:   pybind11, rapidjson, stb_image + stb_image_write, a baseline
                    TIFF reader, libsvm (2 files), cxxopts, optionally cppdb,
                    cpp-httplib, darknet
Python runtime:     wheels from lock files: numpy, torch, opencv-python-headless,
                    av (PyAV), scipy, and the VIAME model-backend forks
```

Assumption to confirm (open decision 1): "get rid of OpenCV/FFmpeg" means
the C++ build. Python code may keep using the `cv2` and `av` wheels for
algorithms that are inherently CV-library work (camera calibration, SGBM,
SIFT/RANSAC, video decode). If the answer is "python too", the OpenCV and
FFmpeg phases grow by the python ports listed in `lite-removals.md` §2.6
and §3.4 and should be re-estimated.

## 3. Guiding decisions

| Decision | Choice | Why |
|---|---|---|
| Order of removals | VXL, FFmpeg, (import kwiver subset), Eigen, OpenCV, kwiver remnants | VXL is required first. FFmpeg is next-easiest: 2 used registrations, replaced by python. Eigen and OpenCV both need the kwiver code in-tree first; Eigen is a mechanical swap behind a small math library, OpenCV needs kernels plus python ports. Kwiver infrastructure last because everything sits on it. Eigen and OpenCV phases are independent and may be swapped |
| Each removal is sub-stepped | Replacement lands and is golden-tested while the old dependency is still built; the dependency is switched off in the final task of the phase | Every task keeps the build green; a phase can pause mid-way |
| Image kernels | In-house `library/image_ops` (resize, warp, colour, demosaic, filters, morphology, histogram/CLAHE, drawing, simple contours) on `core_types::image` | Needed for the VXL replacements anyway; also removes most of OpenCV's imgproc use. Bounded scope (~5k lines), golden-tested |
| Codecs | Vendored `stb_image`/`stb_image_write` (png, jpg, bmp) + baseline TIFF reader (8/16-bit, uncompressed/LZW/PackBits) in `video_io/codecs` | Removes OpenCV imgcodecs; 16-bit TIFF is required by HabCam / 16-bit pipelines |
| Video | Python `video_input`/`video_output` on PyAV (aliases `ffmpeg`, `vidl_ffmpeg`), `image_list` stays C++ | No decoder in C++; most pipelines already run under the python scheduler |
| Linear algebra | `core_types/math`: fixed-size vector/matrix, quaternion rotation, 3x3/4x4 inverse, small-matrix Jacobi SVD/eigen, Cholesky. Heavier numerics (bundle adjust, camera optimisation, calibration) move to python numpy/scipy | Keeps C++ math small and testable; the heavy solvers are called rarely and are already partially python |
| CV-library algorithms | Calibration, stereo rectification/SGBM, feature matching, GrabCut, AdaBoost move to python implementations registered under the same names | Rewriting them in C++ without OpenCV is not worth it |
| Kwiver | Copy the used subset into `library/{core_types,algorithm_framework,pipeline_framework}` first (transitional), then replace piece by piece: types (during Eigen removal), kwiversys -> std, cereal -> rapidjson, plugin loader, logger, python bindings hand-written (no castxml), sprokit trimmed into `pipeline_framework` | A working in-tree copy makes every later replacement an isolated, golden-tested diff |
| Namespaces / module names | Kept (`kwiver::vital`, `sprokit`, `kwiver.vital.*`) until the final phase; then renamed to `viame` with a `kwiver` python shim | Keeps every earlier phase mechanical and diffable |
| Algorithm interfaces and registry | Kept: abstract interfaces, `:type name` selection, nested composition, python implementations behind the same interfaces. Simplified when we own it (P8): hand-written trampolines, smaller base class | 443 config files and every python algorithm depend on it |
| Plugin loading | **Static registration.** Every `library/<dir>/register.cxx` is compiled into one `libviame` shared library shared by the `viame` executable and the python extension; a generated `viame::register_builtins()` calls each. Python implementations register lazily by import path. No plugin directory scan, no `KWIVER_PLUGIN_PATH`. A ~150-line optional `VIAME_PLUGIN_PATH` dlopen hook for out-of-tree C++ plugins is open decision 10 | Dynamic loading was the source of the 15 s startup regressions, the four plugin directories, RPATH/DLL and `fixup_bundle` trouble, and late silent failures. After the port only darknet and cppdb are optional native code, so graceful runtime degradation buys little. Dynamic modules remain in P2 to P7 only because kwiver's loader is still in use |
| Python deps | Lock files + one pip step; forks are prebuilt wheels from a Kitware index built by a separate wheel CI | The forks rarely change relative to VIAME |
| Darknet | `third_party/darknet` via `add_subdirectory`, optional; removal after shipped models are ONNX | 39 pipelines still name `darknet` |

## 4. Phases

Detailed tasks are in `tasks/`. Each phase ends with a clean-configure
build and the full verification set green.

| Phase | Task file | Removes / delivers | Size |
|---|---|---|---|
| P0 | `phase-00-baseline.md` | Branch, `registry-dump` and `pipe-check` applets, JSON baselines, compare scripts, CRITICAL list | S |
| P1 | `phase-01-single-build.md` | Superbuild gone; kwiver via `add_subdirectory`; OpenCV/FFmpeg/Eigen still found; fletch gone; `third_party/` for eigen (temporary), tinyxml, libsvm, pybind11, cppdb; python lock files | M |
| P2 | `phase-02-library-layout.md` | `plugins/` -> `library/<functional>/`; CMake helpers; single `register.cxx` per library; python packages per library; kwiver still a submodule | L (mechanical) |
| P3 | `phase-03-drop-vxl.md` | `library/image_ops` v1; OpenCV-free replacements for every `vxl_*` impl under alias names; `KWIVER_ENABLE_VXL=OFF`; VXL out of the build | M |
| P4 | `phase-04-drop-ffmpeg.md` | Python `video_input`/`video_output` on PyAV with the same config keys; `KWIVER_ENABLE_FFMPEG=OFF`; FFmpeg out of the build | M |
| P5 | `phase-05-import-kwiver.md` | Used kwiver subset copied into `library/core_types`, `algorithm_framework`, `pipeline_framework`, and functional dirs; submodule removed; unused arrows/processes/tests not copied | L (mechanical) |
| P6 | `phase-06-drop-eigen.md` | `core_types/math`; all types and consumers ported; heavy numerics to python; Eigen out of `third_party` | L |
| P7 | `phase-07-drop-opencv.md` | Codecs; `image_ops` v2; drawing; calibration YAML parser; calib3d/ml/features ports to python; `find_package(OpenCV)` removed | XL |
| P8 | `phase-08-replace-kwiver-infra.md` | kwiversys -> std; cereal -> rapidjson helper; static registry replaces the plugin loader (single `libviame`, lazy python factories, startup benchmark); own logger; hand-written pybind11 bindings (castxml gone); sprokit trimmed to `pipeline_framework`; kwiver CMake macros gone | L |
| P9 | `phase-09-python-packaging.md` | Wheel CI, index, lock files final; vendored python moved out; patched wheels replace install-time patching | M |
| P10 | `phase-10-install-platforms.md` | Install layout, setup script, presets, Windows/macOS/docker, packaging, CI | M |
| P11 | `phase-11-rename.md` | `kwiver::vital` -> `viame`, `kwiver.*` -> `viame.*` with shim | L (mechanical) |

Dependency graph between phases: P0 -> P1 -> P2 -> P3 -> P4 -> P5 -> {P6, P7}
-> P8 -> P9 -> P10 -> P11. P6 and P7 are independent of each other; P9 and
P10 can start after P5 if needed.

## 5. Verification

| Check | Tool | Pass condition |
|---|---|---|
| Registered names, aliases, config keys | `viame registry-dump` vs `tests/baseline/registry.json` | Identical, minus `removed.json` |
| Every pipeline parses and resolves | `viame pipe-check --all` over configs/, examples/, add-on zips | Identical to baseline |
| Behaviour | ctest `CRITICAL`, `EXAMPLES`, `pipelines` | Pass on linux-gpu and linux-cpu |
| Replacement fidelity | ctest `GOLDEN`: one test per replaced implementation, tolerance in the test | Pass |
| Dependency actually gone | `git grep` hygiene lines in AGENT_GUIDE.md; `ldd` of every plugin module shows no lib from the dropped dependency | Empty / clean |
| DIVE | manual smoke at P1, P4, P7, P10 | Detector, tracker, training, scoring run from the GUI |

## 6. Risks

| Risk | Mitigation |
|---|---|
| In-house kernels differ numerically from VXL/OpenCV | Golden tests with per-filter tolerances written before the swap; VXL/OpenCV stay in the build until the golden tests pass |
| Python video reading too slow for high-rate pipelines | Benchmark task in P4 with a target (>= 150 fps 1080p decode on the reference machine); `ffmpeg_cli` rawvideo-pipe fallback; `image_list` unaffected |
| Eigen removal breaks numerics in measurement code | `core_types/math` unit tests against recorded Eigen results; measurement golden tests |
| Calibration file compatibility (OpenCV YAML) | Own reader/writer for the `%YAML:1.0` OpenCV matrix subset, tested on every calibration file in `tests/data` and add-ons |
| Add-ons reference old python module paths | Re-export packages kept until add-ons are rebuilt; `pipe-check` at configure time |
| External plugin authors | `include/viame/compat/` header shims for one release; documented in RELEASE_NOTES. If decision 10 drops dlopen, in-tree builds or python are the documented paths |
| Windows without fletch | Python from python-build-standalone; no other native deps remain by P7, so Windows becomes simpler, not harder |
| Scope creep into algorithm rewrites | Registry baseline must stay identical; anything beyond a task's scope becomes a new task |

## 7. Open decisions

1. Do python-side `cv2` and `av` wheels remain acceptable after OpenCV and
   FFmpeg are removed from the C++ build? (Assumed yes.)
2. Darknet: keep vendored, or convert shipped models to ONNX and drop.
3. `kw_archive_writer` / `filter_to_kwa.pipe`: drop or reimplement.
4. Matlab bridge: proposed drop.
5. Postgres/cppdb: keep as option (proposed) or drop.
6. AdaBoost IQR (`process_query_adaboost`, one pipeline) and
   `hierarchical_svm` refiner: port to python (sklearn) or drop.
7. TIFF scope: baseline + LZW + PackBits, 8/16-bit, strips; anything else
   (tiles, JPEG-in-TIFF, BigTIFF) falls back to the python `pil` image_io.
8. Phase 11 rename: do it, and whether the `kwiver` shim is permanent.
9. Hosting for the python wheel index.
10. Out-of-tree C++ plugins: keep a minimal `VIAME_PLUGIN_PATH` dlopen hook,
    or drop dynamic loading entirely and make python the only extension
    path. Depends on whether any external group ships compiled plugins.
