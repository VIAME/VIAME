# Phase 3: drop VXL

Goal: every `vxl_*` name resolves to an in-house implementation on
`library/image_ops`; VXL is not built. References: lite-removals.md §1.
Note: P1 built with `VIAME_ENABLE_VXL=OFF` and a `pending.json`. For golden
recording, P3-T01 needs one VXL-enabled build of `main` (or `lite` with
`VIAME_ENABLE_VXL=ON` and VXL installed on the reference machine).

### P3-T01 Record VXL golden outputs
Depends: P2-T10
Do:
- From a VXL-enabled install, run each pipeline in `configs/pipelines` and `examples` that uses a `vxl_*` filter or `vxl` image_io on `pipelines_test_data` inputs (script `tests/golden/record.py PIPELINE INPUT OUT`). Commit outputs under `tests/golden/vxl/<pipeline>/` (images as PNG; 16-bit kept 16-bit) with a `manifest.json` (inputs, config overrides, VXL git hash).
- Also record per-filter unit fixtures: for each impl, 3 input images (8-bit RGB, 8-bit gray, 16-bit gray) x each config variant found in pipelines -> output image.
Done when:
- `tests/golden/vxl/` committed; `tests/golden/README.md` documents how to re-record.

### P3-T02 `image_ops` v1 kernels
Depends: P2-T10
Do:
- `library/image_ops/`: header-templated over `vital::image_of<T>`; functions listed in lite-removals.md §1 (convert dtype with scaling rules identical to `vil_convert_stretch_range`/cast semantics, channel select/merge, scale/offset, percentile computation with histogram, temporal average accumulator (window, cumulative, exponential), box blur, gaussian blur, erode/dilate rect+ellipse, threshold abs/percentile, colour histogram + commonality map, grey-world/percentile white balance).
- Unit tests in `library/image_ops/tests/` against hand-computed values; no golden yet.
Done when:
- `ctest -R image_ops` passes; no OpenCV or VXL include in `library/image_ops`.

### P3-T03 `convert_image` (alias `vxl_convert_image`)
Depends: P3-T01, P3-T02
Do:
- `library/image_processing/convert_image.{h,cxx}`, config keys copied from `arrows/vxl/convert_image.h` (`format`, `single_channel`, `scale_factor`, `random_grayscale`, `percentile_norm`, `force_three_channel`, `force_8bit` and the rest; diff against `registry.json` to make sure none is missed). Register `convert_image` + alias `vxl_convert_image`.
- Golden test `viame_add_golden_test(convert_image ...)` over the P3-T01 fixtures; tolerance: max abs diff <= 1 for 8-bit outputs, <= 4 for 16->8 percentile paths; record the actual numbers in the test.
Done when:
- Golden passes; `registry-dump` shows the alias; `pending.json` no longer lists `vxl_convert_image`.

### P3-T04 `average_frames`, `threshold`, `morphology`, `color_commonality`
Depends: P3-T03
Do:
- Same pattern for `vxl_average`, `vxl_threshold`, `vxl_morphology`, `vxl_color_commonality` with their config keys. Golden per filter.
Done when:
- Golden passes; four more names leave `pending.json`.

### P3-T05 `white_balance`, `vxl_enhancer`, `format_images_srm`
Depends: P3-T04
Do:
- `perform_white_balancing` (templated header) rewritten on `image_ops`; `vxl_enhancer` registered as alias of `ocv_enhancer` (P7 makes that in-house); `format_images_srm_process` rewritten on `image_ops`.
Done when:
- Golden passes for the srm and white-balance pipelines; names leave `pending.json`.

### P3-T06 `core` image_io and unreferenced vxl names
Depends: P3-T05
Correction: this task used to say "register `vxl` as an alias of `ocv`
image_io". That is wrong and would break most pipelines. Measured against the
recordings:
- `ocv` image_io declares **no config keys**; `vxl` declares five
  (`force_byte`, `auto_stretch`, `manual_stretch`, `intensity_range`,
  `split_channels`). Aliasing drops all five, so `baseline:registry` fails and
  every pipeline that sets one silently loses it.
- 56 shipped pipelines set `image_reader:vxl:force_byte true`, and that is one
  of the two behaviours that actually differ: on the 16-bit fixture `vxl` with
  `force_byte` returns uint8 while `ocv` returns uint16. `auto_stretch` differs
  too. 16 of the 18 recorded image_io cases match; those two do not.
Do:
- Write `image_io` in VIAME registered as `core`, keeping all five config keys
  and their defaults. Decode by delegating to a nested image_io (`ocv` until
  phase 7 brings codecs in-house), then apply the vxl semantics:
  - `force_byte`: convert to uint8, via `auto_stretch` (min and max mapped to
    0 and 255), `manual_stretch` (`intensity_range` mapped to 0 and 255), or a
    plain cast.
  - without `force_byte`: keep the native type, stretching to the type range
    when asked, where the destination maximum is extended by `1 - 1e-6` so the
    top value still truncates to the type maximum.
  - `split_channels` on load stacks sibling per-plane files as extra planes; on
    save it writes one file per plane.
- Add to `removed.json`: `vxl` bundle_adjust, estimate_canonical_transform,
  estimate_essential_matrix, estimate_fundamental_matrix, estimate_homography,
  estimate_similarity_transform, optimize_cameras, split_image,
  triangulate_landmarks, `vxl_plane`, `vxl_constrained`, `vxl_kd_tree`,
  `vxl_aligned_edge_detection`, `vxl_high_pass_filter`,
  `vxl_hashed_image_classifier_filter`, `vxl_pixel_feature_extractor`, and
  `kw_archive_writer` if open decision 3 is "drop" (else task P3-T09).
Done when:
- All 18 recorded `vxl` image_io cases replay against `core`.
- BASELINE passes with the removals recorded.

### P3-T07 Switch VXL off and delete its sources
Depends: P3-T06
Do:
- Delete `VIAME_ENABLE_VXL`, `library/image_processing/vxl_*` leftovers, kwiver `KWIVER_ENABLE_VXL` forced OFF, `arrows/vxl` excluded. `git grep -i vxl -- library tools cmake CMakeLists.txt configs` reviewed: only pipeline alias names remain.
Done when:
- `ldd build/install/lib/viame/plugins/*.so | grep -i vxl` empty; build from clean; BASELINE, CRITICAL, GOLDEN pass.

### P3-T08 Migrate pipeline files to new names
Depends: P3-T07
Do:
- Script `design/scripts/rename_impls.py` replaces `vxl_convert_image`->`convert_image`, `vxl_average`->`average_frames`, etc. in `configs/` and `examples/` only (not add-ons). Regenerate `tests/baseline/pipes.json` (resolved impl names change; statuses must not) and delete `pending.json`.
Done when:
- BASELINE passes with `pending.json` gone; add-on zips still resolve via aliases (`pipe-check --all`).

### P3-T09 (conditional) `kw_archive_writer` without VXL
Depends: P3-T07; only if open decision 3 = keep
Do:
- Reimplement KWA index/data writing with plain streams (format documented in the old `kw_archive_writer_process.cxx`); golden against a recorded archive.
Done when:
- `filter_to_kwa.pipe` output byte-identical to golden.

### P3-T10 Build kwiver locally so VIAME_ENABLE_VXL reaches it
Depends: P3-T02. Blocks P3-T06 and P3-T07.
Context: nothing about kwiver is fixed. `cmake/add_project_kwiver.cmake` already
passes `-DKWIVER_ENABLE_VXL:BOOL=${VIAME_ENABLE_VXL}` down to it, so the
superbuild turns `arrows/vxl` off with the flag we already own. The only reason
this is a task at all is the transitional arrangement: phases 1 and 2 are
deferred, so this tree has been reusing the reference superbuild's already-built
kwiver rather than building its own. Building fletch and pytorch again to change
one kwiver flag would cost hours for nothing, so kwiver alone is configured
directly against the reference fletch, with the same options the superbuild
passes.
Do:
- Check out the `kwiver` submodule and configure it into `build/kwiver-build`
  with `build/kwiver-cache.cmake`, which mirrors the reference kwiver build's
  options and exposes `KWIVER_ENABLE_VXL` as the flag phase 3 drives. Install
  into `build/install`.
- Point `kwiver_DIR`, `VIAME_BUILD_KWIVER_DIR` and `KWIVER_SOURCE_DIR` in
  `build/lite-cache.cmake` at this tree's kwiver.
- Rebuild kwiver with VXL still ON first and confirm nothing moved, so that the
  later flip is the only change under test.
Done when:
- `ctest -L BASELINE` and `ctest -L GOLDEN` pass against the locally built
  kwiver, unchanged from the reference-built one.
- `build/lite-cache.cmake` no longer names the reference kwiver build.

### P3-T11 `close_loops_homography_guided` and its polygon overlap
Depends: P3-T02. Blocks P3-T07.
Added by: P3-T10's usage scan, which found this name is used and the removal
design does not cover it.
Context: `lite-removals.md` §1 lists five `vxl_*` image filters and the `vxl`
image_io as the only referenced registrations. It misses a sixth:
`common_image_stabilizer.pipe` sets
`loop_closer:multi_method:method_2:type = vxl_homography_guided`, and that file
is included by `register_using_homographies.pipe` and
`common_stabilized_iou_tracker.pipe`. So `close_loops` `vxl_homography_guided`
is a real pipeline dependency, not a candidate for `removed.json`.
Do:
- Port `arrows/vxl/algo/close_loops_homography_guided.{h,cxx}` (~300 lines) into
  VIAME under the same registered name and config keys (`enabled`,
  `checkpoint_percent_overlap`, `homography_filename`, `max_checkpoint_frames`,
  plus its nested `feature_matcher`).
- Its only VXL use is `arrows/vxl/compute_homography_overlap` (~260 lines): the
  fraction of an ni x nj frame still covered after a 3x3 homography, via vgl
  convex hull, polygon intersection and area. Reimplement as plain C++ in
  `image_ops`: Sutherland-Hodgman clip of the warped quad against the frame
  rectangle, shoelace area, ratio. `vnl_double_3x3` becomes a plain 3x3.
- Golden: record `overlap()` over a spread of homographies (identity, pure
  translation partly off frame, rotation, scale up and down, a degenerate
  projective one) before the port, and hold the replacement to it.
Done when:
- `vxl_homography_guided` resolves with VXL off; the recorded overlap values
  match; `register_using_homographies.pipe` and
  `common_stabilized_iou_tracker.pipe` still bake in `pipe-check --all`.

### P3-T12 `vidl_ffmpeg` video_input bridge
Depends: P3-T02. Blocks P3-T07.
Added by: P3-T07, whose registry check found the name missing after the flip.
Context: `lite-removals.md` §1 does not mention `vidl_ffmpeg`, and neither did
the usage scan in STATUS.md, which looked for names containing "vxl". It is a
VXL (`vidl`) video reader and it is the video path of VIAME's own tooling:
`tools/run_bulk.py`, `tools/launch_annotator.py`, `tools/train.cxx`,
`plugins/core/utilities_training.cxx` (which sets
`vidl_ffmpeg:stop_after_frame`) and `plugins/vertex-ai/process_handler.py` all
select it with `-s input:video_reader:type=vidl_ffmpeg`. No shipped `.pipe`
names it as a `:type`, which is why the scan missed it; the shipped pipelines
carry inert `block video_reader:vidl_ffmpeg` settings for when it is selected
at run time.
Do not alias it to the `ffmpeg` video_input: the two share **no** config keys.
`vidl_ffmpeg` has `output_nth_frame`, `start_at_frame`, `stop_after_frame`,
`time_scan_frame_limit`, `time_source` and `use_metadata`; `ffmpeg` has twelve
entirely different ones. Aliasing silently drops all six, including the
`stop_after_frame` the trainer relies on. This is the same trap as the image_io
in P3-T06.
Do:
- Write a video_input in VIAME registered as `vidl_ffmpeg`, keeping those six
  keys and their defaults, delegating decode to a nested video_input (`ffmpeg`
  for now; phase 4 replaces it with PyAV) and implementing on top of it:
  frame selection (`start_at_frame`, `stop_after_frame`, `output_nth_frame`)
  and the timestamp policy (`time_source`, `use_metadata`,
  `time_scan_frame_limit`).
- Golden: record frame count, per frame timestamps and image digests over a
  short video for each `time_source` value and a couple of frame ranges,
  before the flip on a VXL build, and hold the replacement to it.
Done when:
- `baseline:registry` no longer reports `vidl_ffmpeg` missing.
- `viame run` over a video produces the same frames and timestamps as recorded.
