# Lite branch status ledger

Update after every task (see AGENT_GUIDE.md). Status values: `todo`,
`in-progress`, `done`, `blocked (reason)`, `skipped (reason)`. Put the commit
hash in the Commit column and anything that deviated from the task text in
Notes.

## Current position

- Phase: P0
- Next task: P0-T01
- Last clean-configure build verified: never
- Reference machine: (fill in host, GPU, OS, CUDA)

## Tasks

| ID | Task | Depends | Status | Commit | Notes |
|---|---|---|---|---|---|
| **Phase 0: baseline and compatibility contract** | `phase-00-baseline.md` | | | | |
| P0-T01 | Create the `lite` branch | - | todo | | |
| P0-T02 | `registry-dump` applet | P0-T01 | todo | | |
| P0-T03 | `pipe-check` applet | P0-T01 | todo | | |
| P0-T04 | Baseline files and compare scripts | P0-T02, P0-T03 | todo | | |
| P0-T05 | Wire baseline checks into ctest | P0-T04 | todo | | |
| **Phase 1: single build with code in place** | `phase-01-single-build.md` | | | | |
| P1-T01 | Inventory the normal-build branch | P0-T05 | todo | | |
| P1-T02 | New top-level CMakeLists and options file | P1-T01 | todo | | |
| P1-T03 | `third_party/` for the small libraries | P1-T02 | todo | | |
| P1-T04 | `viame_dependencies.cmake` | P1-T03 | todo | | |
| P1-T05 | Kwiver as a subdirectory | P1-T04 | todo | | |
| P1-T06 | Plugins, tools, tests inside the same build | P1-T05 | todo | | |
| P1-T07 | Python dependency step | P1-T06 | todo | | |
| P1-T08 | Remove dead trees and submodules | P1-T07 | todo | | |
| P1-T09 | Docker and CI for the single build | P1-T08 | todo | | |
| P1-T10 | DIVE smoke test | P1-T08 | todo | | |
| **Phase 2: functional library layout** | `phase-02-library-layout.md` | | | | |
| P2-T01 | CMake helpers (kwiver-backed) | P1-T10 | todo | | |
| P2-T02 | Generate the file map | P2-T01 | todo | | |
| P2-T03 | Skeleton `library/` with empty targets | P2-T02 | todo | | |
| P2-T04 | Move `utilities`, `video_io`, `file_io` | P2-T03 | todo | | |
| P2-T05 | Move `image_processing`, `object_detectors`, `classifiers`, `segmentation` | P2-T04 | todo | | |
| P2-T06 | Move `object_trackers`, `descriptors`, `measurement` | P2-T05 | todo | | |
| P2-T07 | Move `training`, `evaluation`, `examples`, `tools` links | P2-T06 | todo | | |
| P2-T08 | Move remaining plugins and delete `plugins/` | P2-T07 | todo | | |
| P2-T09 | Relocate tests | P2-T08 | todo | | |
| P2-T10 | Docs and DIVE smoke | P2-T09 | todo | | |
| **Phase 3: drop VXL** | `phase-03-drop-vxl.md` | | | | |
| P3-T01 | Record VXL golden outputs | P2-T10 | todo | | |
| P3-T02 | `image_ops` v1 kernels | P2-T10 | todo | | |
| P3-T03 | `convert_image` (alias `vxl_convert_image`) | P3-T01, P3-T02 | todo | | |
| P3-T04 | `average_frames`, `threshold`, `morphology`, `color_commonality` | P3-T03 | todo | | |
| P3-T05 | `white_balance`, `vxl_enhancer`, `format_images_srm` | P3-T04 | todo | | |
| P3-T06 | `vxl` image_io alias and unreferenced vxl names | P3-T05 | todo | | |
| P3-T07 | Switch VXL off and delete its sources | P3-T06 | todo | | |
| P3-T08 | Migrate pipeline files to new names | P3-T07 | todo | | |
| P3-T09 | (conditional) `kw_archive_writer` without VXL | P3-T07; only if open decision 3 = keep | todo | | |
| **Phase 4: drop FFmpeg** | `phase-04-drop-ffmpeg.md` | | | | |
| P4-T01 | Record video golden data and benchmark baseline | P3-T08 | todo | | |
| P4-T02 | PyAV `video_input` | P4-T01 | todo | | |
| P4-T03 | PyAV `video_output` | P4-T02 | todo | | |
| P4-T04 | `ffmpeg_cli` fallback reader | P4-T02 | todo | | |
| P4-T05 | Switch FFmpeg off | P4-T03, P4-T04 | todo | | |
| **Phase 5: import the kwiver subset** | `phase-05-import-kwiver.md` | | | | |
| P5-T01 | Reachability list | P4-T05 | todo | | |
| P5-T02 | Copy `vital` into `core_types` and `algorithm_framework` | P5-T01 | todo | | |
| P5-T03 | Copy sprokit into `pipeline_framework` | P5-T02 | todo | | |
| P5-T04 | Copy used arrows and processes into functional dirs | P5-T03 | todo | | |
| P5-T05 | Remove the kwiver submodule | P5-T04 | todo | | |
| P5-T06 | Prune unused `vital` code by compile | P5-T05 | todo | | |
| P5-T07 | DIVE smoke and docs | P5-T06 | todo | | |
| **Phase 6: drop Eigen** | `phase-06-drop-eigen.md` | | | | |
| P6-T01 | Inventory Eigen usage | P5-T07 | todo | | |
| P6-T02 | `core_types/math` library | P6-T01 | todo | | |
| P6-T03 | Port `core_types` off Eigen | P6-T02 | todo | | |
| P6-T04 | Port `algorithm_framework`, `pipeline_framework`, `utilities`, `file_io` | P6-T03 | todo | | |
| P6-T05 | Port `measurement` and `object_trackers` | P6-T04 | todo | | |
| P6-T06 | Port the rest and delete Eigen | P6-T05 | todo | | |
| **Phase 7: drop OpenCV from the C++ build** | `phase-07-drop-opencv.md` | | | | |
| P7-T01 | Inventory and golden recording | P5-T07 | todo | | |
| P7-T02 | Codecs | P7-T01 | todo | | |
| P7-T03 | `image_ops` v2 | P7-T01 | todo | | |
| P7-T04 | Port `image_processing` and `object_detectors` C++ | P7-T02, P7-T03 | todo | | |
| P7-T05 | Calibration file IO | P7-T01 | todo | | |
| P7-T06 | Port `measurement` to python where calib3d is needed | P7-T05 | todo | | |
| P7-T07 | Port `classifiers`, `segmentation`, `descriptors`, `training`, `evaluation` | P7-T04 | todo | | |
| P7-T08 | Darknet without OpenCV | P7-T07 | todo | | |
| P7-T09 | Switch OpenCV off | P7-T06, P7-T08 | todo | | |
| **Phase 8: replace the kwiver-derived infrastructure** | `phase-08-replace-kwiver-infra.md` | | | | |
| P8-T01 | Hand-written python bindings for `core_types` | P6-T06, P7-T09 | todo | | |
| P8-T02 | Hand-written algorithm trampolines (castxml gone) | P8-T01 | todo | | |
| P8-T03 | Own plugin registry | P8-T02 | todo | | |
| P8-T04 | Own logger and exceptions cleanup | P8-T03 | todo | | |
| P8-T05 | kwiversys -> std | P8-T04 | todo | | |
| P8-T06 | cereal -> rapidjson helper | P8-T05 | todo | | |
| P8-T07 | Trim `pipeline_framework` | P8-T03 | todo | | |
| P8-T08 | Own CMake helper implementation | P8-T07 | todo | | |
| P8-T09 | Config and algorithm base simplification | P8-T08 | todo | | |
| P8-T10 | Lazy python registration and startup budget | P8-T03 | todo | | |
| **Phase 9: python packaging** | `phase-09-python-packaging.md` | | | | |
| P9-T01 | Wheel CI | P5-T07 (can run in parallel with P6-P8) | todo | | |
| P9-T02 | Vendored python -> wheels | P9-T01 | todo | | |
| P9-T03 | Remove submodules and install-time patches | P9-T02 | todo | | |
| P9-T04 | Lock file hygiene | P9-T03 | todo | | |
| **Phase 10: install layout, platforms, packaging** | `phase-10-install-platforms.md` | | | | |
| P10-T01 | Install layout and setup script | P8-T09 | todo | | |
| P10-T02 | CMakePresets | P10-T01 | todo | | |
| P10-T03 | Windows | P10-T02 | todo | | |
| P10-T04 | macOS | P10-T02 | todo | | |
| P10-T05 | Docker images and release packaging | P10-T02 | todo | | |
| P10-T06 | Release notes and migration guide | P10-T05 | todo | | |
| **Phase 11 (optional): rename to viame** | `phase-11-rename.md` | | | | |
| P11-T01 | C++ namespace rename | P10-T06 | todo | | |
| P11-T02 | Python module rename | P11-T01 | todo | | |
| P11-T03 | Env var and log-level rename | P11-T02 | todo | | |

## Decisions taken

| Date | Decision | Chosen by | Where recorded |
|---|---|---|---|
| | | | |

## Removed names log

Mirror of `tests/baseline/removed.json` with the task that removed each entry.

| Kind | Interface | Name | Task | Reason |
|---|---|---|---|---|
| | | | | |

## Measurements

| What | Value | Task | Date |
|---|---|---|---|
| C++ video decode throughput, 1080p h264 (baseline) | | P4-T01 | |
| PyAV decode throughput | | P4-T02 | |
| kwiver files copied (lines) | | P5-T01 | |
| core_types + algorithm_framework lines after prune | | P5-T06 | |
| pipeline_framework lines after trim | | P8-T07 | |
| `viame --version` / `runner --help` startup, before and after static registry | | P8-T10 | |
| Install size, `main` baseline (see lite-install-size.md) | 14 GB | - | 2026-09-08 |
| Install size after P1 lock files + blacklist | | P1-T08 | |
| Install size at end of P10 | | P10-T05 | |
| final core_types / algorithm_framework / pipeline_framework lines | | P8-T09 | |
