# Lite branch status ledger

Update after every task (see AGENT_GUIDE.md). Status values: `todo`,
`in-progress`, `done`, `blocked (reason)`, `skipped (reason)`. Put the commit
hash in the Commit column and anything that deviated from the task text in
Notes.

## Current position

- Phase: P4 (phases 1 and 2 deferred, see the decision below)
- Next task: P4-T01
- Last clean-configure build verified: 2026-09-09, P0-T05
- Reference machine: local workstation, CUDA 12.6, cuDNN 9.12, Ubuntu
  (kernel 6.8), python 3.10.12, gcc default, 16 cores

### Build arrangement during phases 0 to 1

Phase 1 is what replaces the superbuild, so until then the `lite` checkout
builds only the VIAME project itself and takes its dependencies from an
existing `main` superbuild of the same commit:

| What | Where |
|---|---|
| Source | `~/Dev/viame-lite/src` (this checkout, branch `lite`) |
| Build | `~/Dev/viame-lite/build/viame-build` |
| kwiver build | `~/Dev/viame-lite/build/kwiver-build`, cache `~/Dev/viame-lite/build/kwiver-cache.cmake` (P3-T10) |
| Install | `~/Dev/viame-lite/build/install` (seeded by copying the reference install) |
| Initial cache | `~/Dev/viame-lite/build/lite-cache.cmake`, mirroring every `VIAME_*` setting of the reference build |
| fletch, darknet | `~/Dev/viame/build` (reference superbuild, `main` @ 8edfd2f66) |
| kwiver | built here from the submodule pin, so `KWIVER_ENABLE_VXL` is ours to drive |

```
# kwiver first; KWIVER_ENABLE_VXL in kwiver-cache.cmake is the phase 3 flag
cmake -S src/packages/kwiver -B build/kwiver-build -C build/kwiver-cache.cmake
cmake --build build/kwiver-build -j16 && cmake --install build/kwiver-build

cmake -S src -B build/viame-build -C build/lite-cache.cmake   # PATH must have nvcc
cmake --build build/viame-build -j16 && cmake --install build/viame-build
ctest --test-dir build/viame-build -L BASELINE
```

`packages/downloads` holds symlinks to the reference checkout's downloads so
that model packs are not fetched twice. The `fletch` submodule is checked out
for `CMake/FindCUDNN.cmake`, and `kwiver` is checked out and built.

## Tasks

| ID | Task | Depends | Status | Commit | Notes |
|---|---|---|---|---|---|
| **Phase 0: baseline and compatibility contract** | `phase-00-baseline.md` | | | | |
| P0-T01 | Create the `lite` branch | - | done | 2547f6cb8 | `design/` committed on the new branch |
| P0-T02 | `registry-dump` applet | P0-T01 | done | 46669f626 | `plugin_map()` is protected, so the applet goes through `plugin_manager_internal`. 58 python-registered algorithms cannot report their config (the pybind trampoline returns the non-copyable `config_block` by copy) and are recorded with an `error` field instead, so their names are covered but not their defaults. Algorithm descriptions end in a registration-ordered option list, which python import order perturbs; the writer sorts it so two runs are byte identical |
| P0-T03 | `pipe-check` applet | P0-T01 | done | bb08e75a0 | Walks the install tree (`configs/pipelines`, `configs/add-ons`, `examples`) instead of extracting the add-on zips: the build already unpacks add-on pipelines into the install, so this covers them with no zip handling. A pipe names an implementation without its interface, so resolution means "registered under some interface"; that also matches nested non-algorithm keys such as `vxl_average:type = window`, which read as unresolved. Stable, so changes are still caught |
| P0-T04 | Baseline files and compare scripts | P0-T02, P0-T03 | done | 6711a97bc | Baseline is the CUDA build only, not a GPU/CPU union: see the new P0-T06. 293 files checked, 34 failing today, which is more than the two stale names the task text expected; the failures are recorded as the baseline so a fix is noticed too |
| P0-T05 | Wire baseline checks into ctest | P0-T04 | done | a6c4a6429 | `ctest -L BASELINE` passes, 17 s for both |
| P0-T06 | Merge a CPU-build registry dump into the baseline | P0-T05 | todo | | Added by P0-T04 |
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
| P3-T01 | Record VXL golden outputs | P0-T05 | done | 4b46563ba | Dependency changed from P2-T10 by the ordering decision. Recorded through the kwiver python bindings rather than through pipelines, which gives the exact array for every dtype instead of a re-encoded image; whole-pipeline recordings run the shipped `train_aug_*` pipelines, the only shipped pipelines that both use a vxl filter and write images |
| P3-T02 | `image_ops` v1 kernels | P0-T05 | done | 1205fc963 | Lands in `plugins/image_ops` for now; phase 2 moves it to `library/image_ops`. This commit carries the kernels the conversion, threshold and averaging filters need; the morphology, colour histogram, blur and white balance kernels land with P3-T04 and P3-T05, so each kernel arrives with the golden that checks it. Reproducing VXL exactly turned up two behaviours worth knowing: `vil_math_mean_over_planes` accumulates in the pixel type, so two uint8 planes of 100 and 200 average to 22, and the windowed averager subtracts the *newest* buffered frame rather than the oldest, so a full window is not a sliding mean. Both are pinned by unit tests |
| P3-T03 | `convert_image` (alias `vxl_convert_image`) | P3-T01, P3-T02 | done | 22eec3036 | Registered as `convert_image` only: the alias waits for P3-T10 to build kwiver with `KWIVER_ENABLE_VXL=OFF`, since `arrows/vxl` holds the name until then. Equivalence is checked instead by replaying every recorded `vxl_convert_image` case against `convert_image`, driven by the new `REPLACEMENTS` map in `tests/golden/cases.py`; all 14 variants match bit for bit on all three input types. `pending.json` does not exist yet, so nothing was removed from it |
| P3-T04 | `average_frames`, `threshold`, `morphology`, `color_commonality` | P3-T03 | done | a4069a51a | New names only, aliases still wait for VXL to be off. Two kernels added to `image_ops`: binary morphology and the colour commonality histogram. The structuring element and the morphology border rule were measured from the running implementation rather than read out of VXL sources, which is how the disk radius rule (strict `i*i + j*j < r*r`, so radius 1 is one pixel and 2 is a 3x3 square) and the border rule (the element is clipped, so eroding an all-true image leaves it all true) were pinned. `vil_threshold_above` turned out to be `>=`, not `>`: caught by the golden replay on the one input that has pixels exactly at the threshold, after a unit test had asserted the wrong rule. Two recorded paths are deliberate divergences, both uninitialised in VXL and unused by pipelines: `color_commonality` grid mode (tile regions built with corners in the wrong order) and `threshold` percentile on a multi-plane image |
| P3-T05 | `white_balance`, `vxl_enhancer`, `format_images_srm` | P3-T04 | done | 4ce488d49, 3e6c2bb0e | VIAME's own code is now VXL free and `plugins/vxl` is deleted. Deviations, all following the "used, nothing extra" rule: `vxl_white_balancing` was 682 lines of VXL-specific code with zero pipeline uses, so it went to `removed.json` rather than being ported. `vxl_enhancer` became an alias of `ocv_enhancer` after finding the two were already the same code at runtime: `plugins/vxl` and `plugins/opencv` each defined `viame::enhance_images` with identical mangled symbols, so the loader bound one definition for both factories and which one won depended on plugin load order. They matched on all 15 recorded cases. Two extra VXL users the plan does not mention were found and ported: `format_images_srm` (needed a bilinear resample kernel) and `convert_polys_to_mask`, which the VIAME CSV readers use for `poly_to_mask` and which used vgl polygon scan conversion; its `VIAME_ENABLE_VXL` guards are gone so the feature keeps working |
| P3-T06 | `core` image_io and unreferenced vxl names | P3-T05 | done | dfe480070 | Task text corrected first: aliasing `vxl` to `ocv` would have dropped all five config keys and changed `force_byte` and `auto_stretch`, which 56 pipelines depend on. `core` keeps the keys and reproduces all 18 recorded cases. Fifteen unreferenced registrations went to `removed.json` |
| P3-T07 | Switch VXL off and delete its sources | P3-T06 | done | 99ec1955b, 244755658 | VXL is off in kwiver, fletch and VIAME, and the `vxl_*` names are aliases of the in-house implementations. All 84 golden cases pass with VXL absent, including all 9 whole-pipeline recordings bit for bit. `kw_archive_writer` is removed on the user's decision on open decision 3; `filter_to_kwa.pipe` existed only to write one and is deleted, and five indexing and query pipelines dropped the side output, which is the whole of the deliberate `pipes.json` regeneration (the other 34 files differ only in exception message text, which embeds absolute source paths and which `compare_pipes.py` does not compare). `vidl_ffmpeg` is a second registration of kwiver's `ffmpeg` reader, as `lite-removals.md` section 3 says it should be; the six config keys the two do not share are in the new `pending.json`, the mechanism `phase-01` defines for a contract knowingly relaxed between phases, and section 3.1 folds them into the phase 4 PyAV reader. The flip exposed two defects, both fixed: the new algorithms had `PLUGGABLE_VARIABLES` and friends spelled out separately, copied from the `plugins/vxl` header that had the ODR clash, and that spelling omits `get_configuration`, so a partial config from a pipe file threw on the first unset key; and the pipeline golden runner inherited an environment that importing the kwiver python package had rewritten, adding the reference install to `LD_LIBRARY_PATH`, so it now sources the install itself |
| P3-T12 | `vidl_ffmpeg` video_input bridge | P3-T02 | dropped | | Added by P3-T07, then dropped: writing a C++ wrapper was inventing an implementation the plan does not have and phase 4 would delete. `lite-removals.md` section 3 already treats `vidl_ffmpeg` as an older name for the `ffmpeg` reader and section 3.1 folds its keys into the phase 4 PyAV reader, so P3 registers the alias and `pending.json` carries the gap. The recording of what the VXL reader did with each key, taken while it still existed, is committed as `tests/golden/video/manifest.json` for P4-T01 |
| P3-T08 | Migrate pipeline files to new names | P3-T07 | done | 8642ff6bc | `design/scripts/rename_impls.py`, 407 substitutions over 79 files in `configs/` and `examples/`; add-ons are left alone and keep resolving through the aliases, which is what the aliases are for. Every pipeline keeps its status; the only baseline changes are the renamed implementations and the config key paths that embed an implementation name. `pending.json` is not deleted as the task text says: it holds the `vidl_ffmpeg` keys phase 4 restores rather than the VXL names phase 1 would have put there |
| P3-T09 | (conditional) `kw_archive_writer` without VXL | P3-T07; only if open decision 3 = keep | skipped (decision 3 = drop) | | The user settled open decision 3: the KWA format is no longer needed |
| P3-T10 | Build kwiver locally so `VIAME_ENABLE_VXL` reaches it | P3-T02 | done | 0e97eb1a0 | Added by P3-T03; blocks P3-T06 and P3-T07. Only needed because phases 1 and 2 are deferred and this tree had been reusing the reference superbuild's kwiver; the flag itself was already wired. Built from the submodule pin, 64d8306. Building instead from the neighbouring checkout's working tree, which is 4 commits ahead, moved 13 `perform_query` and `handle_descriptor_request` entries and failed `baseline:registry` -- the check doing its job. The lite tree builds what its own submodule pins, and BASELINE and GOLDEN pass unchanged against the locally built kwiver |
| P3-T11 | `close_loops_homography_guided` and its polygon overlap | P3-T02 | done | 3f5201a2a | Added by P3-T10's usage scan: `lite-removals.md` §1 misses this name, which `common_image_stabilizer.pipe` really does select. Registered as `homography_guided`. Its only VXL use was `compute_homography_overlap`, reimplemented as convex polygon clipping and A/B'd against the VXL routine over 52 homographies to within 1e-9 |
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
| 2026-09-09 | Open decision 1: PyPI wheels stay acceptable, so the phase 4 PyAV reader and the phase 7 `opencv-python-headless` fallback are both in scope. The user's rule is that every dependency not on PyPI gets merged into `library/` | user | STATUS.md, lite-plan.md section 7 |
| 2026-09-09 | Open decision 3: drop `kw_archive_writer`. It was the last VXL user in kwiver's sprokit processes and keeping it meant reproducing VXL's `vsl` binary serialisation byte for byte | user | tests/baseline/removed.json, STATUS.md P3-T07 |
| 2026-09-09 | Converted code goes straight into `library/` per `lite-library-layout.md` rather than into new `plugins/` directories to be moved in phase 2 | user | library/, STATUS.md P3-T07 |
| 2026-09-09 | Follow the design documents rather than inventing implementations: the `vidl_ffmpeg` C++ wrapper was dropped in favour of the alias the plan specifies, with the gap recorded in `pending.json` | user | STATUS.md P3-T12 |
| 2026-09-09 | Phases 0 and 1 build only the VIAME project against the reference `main` superbuild's fletch/kwiver rather than rebuilding the superbuild; phase 1 replaces the arrangement | agent, build cost | STATUS.md "Build arrangement" |
| 2026-09-09 | `pipe-check --all` reads the install tree rather than extracting add-on zips | agent, P0-T03 | STATUS.md P0-T03 note |
| 2026-09-09 | Baseline is a CUDA build only; the CPU union is deferred to P0-T06 | agent, P0-T04 | STATUS.md P0-T04 note |
| 2026-09-09 | Dependency removal comes before the restructuring: phase 3 onward runs against the existing `plugins/` tree, and phases 1 and 2 move the resulting code afterwards. Neither phase 1 nor phase 2 removes a dependency, and both are large; going at the dependencies first gets the reduction sooner at the cost of the layout move carrying a little more | user | STATUS.md, this row |
| 2026-09-09 | `vidl_ffmpeg` is used, by VIAME's own tooling rather than by any shipped pipeline, and cannot be aliased to the `ffmpeg` reader: no shared config keys. P3-T12 writes a bridge | agent, P3-T07 | tasks/phase-03 P3-T12 |
| 2026-09-09 | `vxl` image_io is not aliasable to `ocv`, contrary to lite-removals.md §1: `ocv` declares no config keys at all, and 56 pipelines set `force_byte`, one of the two options whose behaviour actually differs. P3-T06 writes a `core` image_io in VIAME keeping all five keys instead | agent, P3-T06 | tasks/phase-03 P3-T06 |
| 2026-09-09 | `vxl_white_balancing` is removed rather than ported: zero pipeline uses, 682 lines of VXL | agent, P3-T05 | tests/baseline/removed.json |
| 2026-09-09 | `arrows/vxl` is what registers `vxl_convert_image` and the rest, so a replacement cannot claim the old name until kwiver is built with `KWIVER_ENABLE_VXL=OFF`. That flag is already wired to `VIAME_ENABLE_VXL` in `add_project_kwiver.cmake`, so nothing new is needed in the build; P3-T10 only exists because this tree had been reusing the reference superbuild's kwiver rather than building its own. Corrected after the first framing of P3-T10 wrongly called kwiver "prebuilt" and treated it as a blocker | user | STATUS.md P3-T03 note, tasks/phase-03 P3-T10 |
| 2026-09-09 | Golden coverage is the full set: per filter fixtures for every config variant the shipped pipelines use, plus whole pipeline recordings, committed under `tests/golden/` | user | tests/golden/README.md |

## Removed names log

Mirror of `tests/baseline/removed.json` with the task that removed each entry.

| Kind | Interface | Name | Task | Reason |
|---|---|---|---|---|
| | | | | |

## Usage scan: what arrows/vxl registers, and what the pipelines touch

From the 293 installed `.pipe` and `.conf` files, comments excluded (P3-T10).
Only the first group moves into VIAME; the second goes to `removed.json`.

| Registered name | Interface | Pipeline uses | Disposition |
|---|---|---|---:|
| `vxl_convert_image` | image_filter | 255 | ported, P3-T03 |
| `vxl` | image_io (reader and writer only) | 116 | alias to `ocv`, P3-T06 |
| `vxl_average` | image_filter | 66 | P3-T04 |
| `vxl_color_commonality` | image_filter | 6 | P3-T04 |
| `vxl_morphology` | image_filter | 8 | P3-T04 |
| `vxl_threshold` | image_filter | 3 | P3-T04 |
| `vxl_homography_guided` | close_loops | 2 | P3-T11 (plan gap) |
| `vxl_enhancer`, `vxl_white_balancing` | image_filter | 0 | VIAME's own, still port off VXL, P3-T05 |
| `format_images_srm` | process | 3 | VIAME's own, P3-T05 |
| `kw_archive_writer` | process | 6 | open decision 3 |
| `vxl` | bundle_adjust, estimate_essential_matrix, estimate_fundamental_matrix, estimate_homography, estimate_similarity_transform, optimize_cameras, split_image, triangulate_landmarks | 0 | `removed.json` |
| `vxl_plane` | estimate_canonical_transform | 0 | `removed.json` |
| `vxl_constrained` | match_features | 0 | `removed.json` |
| `vxl_kd_tree` | nearest_neighbors | 0 | `removed.json` |
| `vxl_aligned_edge_detection`, `vxl_high_pass_filter`, `vxl_hashed_image_classifier_filter`, `vxl_pixel_feature_extractor` | image_filter | 0 | `removed.json` |

The bare name `vxl` is registered for nine interfaces but only ever selected as
an `image_reader` or `image_writer` type, so only the image_io needs to survive.

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
