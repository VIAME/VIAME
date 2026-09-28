# VIAME lite: where the plan stands

Assessed 2026-09-28 against `main-lite` at `3f4b20863` plus the working tree,
on the reference machine. Every number here was measured for this document
rather than copied from the ledger; where the ledger and the tree disagreed,
the tree is what is written down and the disagreement is noted.

`lite-plan.md` says what was intended. `STATUS.md` says what each task did.
This says how much of the intent is met, what is known to be wrong, and what
is left that is worth doing.

## 1. Summary

**The dependency work is complete. The packaging and platform work is not.**

Of 90 tasks in the ledger, 81 are done, two were dropped or skipped on a
recorded decision, and seven remain: all four of phase 9, Windows and macOS
in phase 10, and the release notes, which are drafted and waiting on review.

| Goal (`lite-plan.md` §1) | State | Evidence |
|---|---|---|
| 1. One CMake project | **Met for VIAME's own code** | One configure, one build. No superbuild, no `packages/fletch`, no `packages/kwiver`. `ExternalProject_Add` survives in four places, all optional and none of them VIAME's code: CPython from source, DIVE from source, the python forks, and archive fetching |
| 2. Zero third-party C++ libraries | **Met** | `libviame.so` has eleven `NEEDED` entries: libpython, four CUDA libraries, libgomp, and the C and C++ runtimes. No OpenCV, VXL, Eigen, FFmpeg, zlib or kwiver. The hygiene greps in `AGENT_GUIDE.md` are empty for includes; what they still match is prose and one vendored python extension (§4.6) |
| 3. Source organised by function | **Met, with named exceptions** | 19 functional directories under `library/`. Eleven subdirectories and 37 files are still named after a dependency (§5.2) |
| 4. One dependency at a time, build green throughout | **Met** | Phases 3, 4, 6 and 7 each landed behind golden recordings. 520 unit, 14 baseline, 10 golden and critical, 27 tools: no failures |
| 5. Pipelines, add-ons, DIVE and applets keep working | **Met, as far as it is checked** | 258 of 289 shipped pipelines resolve, and the 31 that do not are accounted for (§3.2). 157 names removed, each with a reason. What "resolve" does not cover is §4.1 |

**Beyond the plan.** Open decision 1 assumed python would keep `cv2`. It did
not: VIAME's python imports no OpenCV, five vendored forks were taken off it
by patch, and no requirements file names an imaging library. That is a
second removal the plan never scheduled, and it is why `image_kernels` is
three times the size section 3 estimated.

## 2. What was delivered

### 2.1 By phase

| Phase | Delivered | State |
|---|---|---|
| P0 | `registry-dump`, `pipe-check`, the JSON baselines and their comparisons | done |
| P1 | Superbuild and fletch deleted; one configure | done |
| P2 | `plugins/` dissolved into `library/` | done |
| P3 | VXL out; `image_kernels` v1 | done |
| P4 | FFmpeg out; video is PyAV, with an `ffmpeg_cli` fallback | done |
| P5 | The used kwiver subset imported, the submodule removed | done |
| P6 | Eigen out; `core_types/math` | done |
| P7 | OpenCV out of the C++ build; codecs, kernels, calibration IO | done |
| P8 | Static registry, own logger, hand-written bindings, `std::filesystem` | done |
| P9 | Wheel CI, forks as wheels, submodules removed | **not started as written** (§5.1) |
| P10 | Install layout, presets, Docker | done; **Windows in progress, macOS not started** |
| P11 | `kwiver::vital` to `viame`, `kwiver.*` to `viame.*` | done |

### 2.2 What the plan did not ask for

| Work | Why it happened | Where |
|---|---|---|
| `cv2` out of VIAME's python | The user's instruction: every feature without OpenCV, not an optional extra | `library/utilities/{imageops,geometry,calibration,chessboard,clustering}.py`, the bindings in `image_kernels_python.cxx` |
| `cv2` out of five vendored forks | `import mmdet` needed it, so the requirement could not go until they did | `packages/patches/{mmcv,mmdetection,imgaug,mmdeploy,rf-detr}.patch`, `packages/patches/sam2/` |
| SURF, SIFT, ORB, FLANN, SGBM, StereoBM, WLS, GrabCut, Canny, Hough in C++ | Each was the last thing keeping a file on `cv2`; SURF additionally because no wheel ships it | `library/image_processing`, `library/image_kernels` |
| A `viame` wheel | The route to PyPI; replaces phase 9's index of fork wheels for the code VIAME owns | `cmake/wheel/`, `docs/wheels.md` |
| Two CUDA paths | GFIT preprocessing and the kernels, optional | `library/image_kernels/CUDA.md`, `GFIT_CUDA.md` |
| Reference tests gathered | `tests/golden` was recordings; the live OpenCV comparisons were scattered | `tests/reference/` |

## 3. Verification, as measured

### 3.1 Tests

| Group | Count | Result |
|---|---|---|
| `UNIT` | 520 | pass, one skip |
| `BASELINE` | 14 | pass |
| `CORE` | 11 | pass |
| `GOLDEN` | 3 suites, 312 cases | 276 pass, 36 skip |
| `CRITICAL` | 7 | pass |
| `tools:` | 27 | pass |

The 36 golden skips: 24 need a CUDA device for darknet, 9 are documented
divergences where the recorded code was wrong and the port is right, and 3
are `vxl_white_balancing`, removed on purpose.

Run at `-j2` on the reference machine. At `-j3`, with the machine shared,
seven unit tests fail and pass on a rerun; that is load, and it is recorded
here so that nobody mistakes it for a regression or, worse, the reverse.

### 3.2 Pipelines

289 files under `configs/` and `examples/`. 258 resolve. The 31 that do not:

| Count | Why | A defect? |
|---|---|---|
| 16 | Templates with `[placeholders]` that training fills in | no |
| 9 | Fragments that read `$CONFIG{global:scale}` from the pipeline that includes them | no |
| 2 | Fragments naming a process the including pipeline supplies | no |
| 2 | `seagis_measurement`; SeaGIS is a licensed option, off here | no |
| 1 | A template with a `TODO` token | no |
| 1 | `query_augment_image.pipe` names `pytorch_augmentation`, which neither this branch nor `main` registers | **yes, inherited** |

## 4. Known defects and gaps

Ordered by what they cost a user, not by how hard they are.

### 4.1 Nothing checks that a pipeline can be configured

`pipe-check` bakes a pipeline and resolves its names. It does not construct a
process or call `_configure`. Two defects of exactly this shape were found
on one day, both invisible to every baseline:

* `measure_objects_process` threw for any pipeline asking for segment
  disparity refinement -- five shipped configs -- because an `#ifndef` lost
  its compile definition when the file moved (fixed, `f5072a5c6`).
* `compute_curved_measurements` was never declared, so two add-on pipelines
  could not be built; `baseline:pipes` had recorded the failure as the
  baseline and agreed with itself (fixed; `baseline:declared_processes`
  guards the class).

**This is the largest gap in the verification.** The golden suite runs 16
pipelines end to end; 258 resolve. The other 242 are checked for spelling.
Task P12-T01 closes it.

### 4.2 The dependency install has never been run from the new locks

`opencv-contrib-python-headless` is out of `base.in` and all sixteen locks
were regenerated. The resolution is verified; the install is not. No tree has
yet had contrib replaced by the two distributions the locks now carry, and
`VIAME_INSTALL_PYTHON_DEPS` is still off by default for the reason the ledger
gives: the first run upgrades 56 packages.

Two consequences are known in advance:

* **Both OpenCV distributions are installed.** `ultralytics` requires
  `opencv-python`; `albucore`, `albumentations` and `kwimage[headless]`
  require `opencv-python-headless`. Same version, same `cv2/` directory, last
  one wins. The runtime image installs `libgl1` so that either works.
* **`kwimage` is pinned at 0.11.6 and its fix is in 0.12.0.** The
  post-install patch step was removed on the grounds that the `putText`
  compatibility fix is upstream, which it is -- in a version the lock does
  not select. No live VIAME path was found that passes a float image to it,
  so this is a guard removed rather than a break demonstrated.

### 4.3 Thirteen of nineteen forks are unverified for `cv2`

`baseline:fork_cv2` checks the submodules that are checked out. On the
reference machine that is six. `sam3` is known to have eight runtime files
that import `cv2` and no patch; the other twelve are unknown. With `cv2`
arriving transitively this breaks nothing today, but the claim "the forks are
off OpenCV" is true of five.

### 4.4 Platforms

* **Windows** has never been built from this branch. What can be checked on
  Linux is done: `VIAME_PYTHON_STANDALONE`, the presets, the setup script.
  The MSVC build, `install( RUNTIME_DEPENDENCY_SET )` and the MSI are not.
  The MSI script still has stages for VIVIA and SEAL, both removed in P1.
* **macOS** has not been started.
* **CI** builds with PyTorch, ONNX, darknet, COLMAP and DIVE all off, because
  `mmcv` alone exhausts the job's budget. It covers the C++ core and little
  of what a release ships.

### 4.5 Building torch from source is gone, and is wanted

P1-T02 removed `VIAME_BUILD_PYTORCH_FROM_SOURCE` and its torchvision twin;
torch comes from the PyTorch index. The index publishes `cu126` and `cu130`
and nothing else, so a build targeting another CUDA has no wheel to take.
`build_server_windows_msi.cmake` still sets both flags and nothing reads
them. What restoring it needs is in `cmake/viame_options.cmake` and P12-T06.

### 4.6 `VIAME_ENABLE_PYTORCH-LEARN` cannot import `cutler`

`library/object_detectors/learn/cutler` imports `pydensecrf` unconditionally.
`pydensecrf` is vendored in the tree, excluded from the install, absent from
`learn.lock`, and needs an `EIGEN_INCLUDE_DIR` that nothing has set since P6
deleted Eigen. The option is off by default and P9-T02 was to have turned
these packages into wheels.

### 4.7 Inherited, and still open

Found during the port, present on `main`, not fixed because fixing them
changes behaviour rather than removing a dependency. All are in
`lite-findings.md` 1.10:

* `-s global:key=value` does not reach `$CONFIG{global:key}`. Not
  re-verified on this branch.
* `optimize_stereo_cameras` refuses a board with an odd number of corners,
  which is its own default target.
* `ocv_calibrate_single_camera` guesses the image size from the corners.
* `compute_disparity` saturates its own disparity map when WLS is on.

### 4.8 Smaller

* **Eight build-server options are read by nothing**: the three
  `VIAME_BUILD_*_DIR`, the two from-source torch flags, and `VIAME_ENABLE_`
  `GDAL`, `SEAL` and `VIVIA`. Kept on the user's instruction; the first five
  name a build mode that may return.
* **The install is never cleaned.** The reference install carries 18
  libraries from before the fold, 280 MB, that nothing loads. The per-module
  python extensions were the dangerous kind and are now removed at configure
  time; these are only dead weight, but a wheel or an image built from an
  upgraded prefix would carry them.
* **The colour recording cannot see the SIMD tail.** `hsv_to_rgb` rounds the
  last `width % 32` pixels of a row, as OpenCV's AVX2 build does. The
  recording is 32 wide. Checked by hand against `cv2` across five widths and
  exact to one value in twenty thousand, but no test holds it there.
* **68 KB of generated audit is committed** under `tests/reports/` and read
  by nothing.

## 5. What is left

### 5.1 Phase 9 was overtaken, and needs restating

The plan was an index of fork wheels built by a separate CI, after which the
submodules and the install-time patching would go. What happened instead:

| Planned | What exists |
|---|---|
| Fork wheels from a Kitware index | Forks built from submodules by `viame_python_forks.cmake`, `--no-deps`, against `forks.lock` |
| `packages/patches` applied in wheel CI | Applied at build time by `apply_fork_patch.cmake`; two forks vendored into `viame.*` |
| Install-time patching removed by publishing patched wheels | Removed outright (`168a97c30`), on the rule that an index package is not modified |
| Lock files as the single source of truth | Done: sixteen locks, two pythons, three accelerators |
| Submodules leave the tree | 19 remain |

The first three rows reach the plan's goal by another road. The last does
not, and is the real remainder of the phase. P9-T01 to T03 as written should
not be started: they describe infrastructure the project no longer needs in
that form. `tasks/phase-09-python-packaging.md` carries the restatement.

### 5.2 Names

The user's instruction is that no code or comment refers to OpenCV or `cv2`,
with one exception: the `ocv_*` implementation names that pipelines select,
kept so that a pipeline written against `main` still runs.

| What | Count | State |
|---|---|---|
| Build flags | 2 | **done**: `VIAME_ENABLE_IMAGE_PROCESSING`, with the old spelling honoured for one release |
| Runtime imports of `cv2` outside `tests/reference` | 0 | **done**, guarded by `baseline:lazy_cv2` |
| `library/file_io/opencv_yaml.*`, `library/utilities/opencv_yaml.py` | 4 files, 23 referencing | todo |
| `library/utilities/compat/opencv/` | 9 aliases | todo: no pipeline, example or tool uses one |
| `tests/reference/opencv_{cases,fixtures}.py` | 2 | todo, with the group they serve |
| Prose in `library/` | about 750 lines | todo: mostly the specification each port was held to |
| `ocv_*` file names | 22 | **kept**: they match the registered names |
| `kwiver_*` file names, `library/compat/kwiver` | 8 | kept for one release by decision 8 |

The prose is the expensive row and the one to be careful with. A comment that
says a kernel reproduces a named function exactly, over how many inputs and
with what residual, is the only statement of what the code is for. Rewriting
it to describe the behaviour is right; deleting it is not.

### 5.3 Worth doing, in order of value

| | Task | Why |
|---|---|---|
| 1 | Configure every shipped pipeline (P12-T01) | §4.1: two defects in one day, and 242 pipelines checked only for spelling |
| 2 | Run the dependency install from the new locks (P12-T02) | §4.2: the one part of the requirements change nobody has seen work |
| 3 | Move `kwimage` to a version with its fix (P12-T03) | §4.2: a pin and a removed patch that disagree |
| 4 | Check out and guard every fork (P12-T04) | §4.3: thirteen unverified |
| 5 | Windows build (P10-T03) | §4.4: the largest platform, never built |
| 6 | Finish the names (P12-T05) | §5.2 |
| 7 | Torch from source (P12-T06) | §4.5 |
| 8 | Clean installs (P12-T07) | §4.8 |
| 9 | A wider colour recording (P12-T08) | §4.8 |
| 10 | macOS (P10-T04) | §4.4 |

## 6. Decisions

| # | Decision | State |
|---|---|---|
| 1 | Do python `cv2` and `av` wheels stay acceptable | **Superseded.** `av` stays. `cv2` does not: nothing VIAME owns imports it and no requirements file names it. It arrives because five packages VIAME depends on require it |
| 2 | Darknet | Vendored, inference only; the trainer is removed |
| 3 | `kw_archive_writer` | Dropped |
| 4 | Matlab bridge | Dropped |
| 5 | PostgreSQL | Removed entirely |
| 6 | AdaBoost IQR, hierarchical SVM | AdaBoost ported to scikit-learn; the SVM refiner removed |
| 7 | TIFF scope | As proposed |
| 8 | Phase 11 rename | Done; the shim lasts one release |
| 9 | Hosting for a wheel index | **Moot as asked.** VIAME's own code ships as a wheel on PyPI; whether the forks need an index is P9's restated question |
| 10 | Out-of-tree C++ plugins | Kept, as an explicit list of files |

**New, and the user's to make:**

| # | Decision | What turns on it |
|---|---|---|
| 11 | Do the eight unread build-server options stay | Whether a desktop superbuild returns. Recommended: keep the five that name a build mode, drop the three that name removed products |
| 12 | Do the MSI's VIVIA and SEAL stages stay | Both products left the tree in P1; the stages now build nothing and diff empty file lists |
| 13 | Is `tests/reports/` a record or an artefact | Whether generated audits are committed |
| 14 | Does the deprecated `VIAME_ENABLE_OPENCV` spelling get a release | It is the one OpenCV name left in the build, on purpose |
