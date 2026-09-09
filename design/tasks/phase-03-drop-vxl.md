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

### P3-T06 `vxl` image_io alias and unreferenced vxl names
Depends: P3-T05
Do:
- Register `vxl` as alias of `ocv` image_io (both reader and writer paths). Add to `removed.json`: `vxl` bundle_adjust, estimate_canonical_transform, estimate_essential_matrix, estimate_fundamental_matrix, estimate_homography, estimate_similarity, optimize_cameras, split_image, triangulate_landmarks, `vxl_kd_tree`, and `kw_archive_writer` if open decision 3 is "drop" (else task P3-T09).
Done when:
- BASELINE passes with `pending.json` empty except `kw_archive_writer` if undecided.

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
