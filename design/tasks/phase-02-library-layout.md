# Phase 2: functional library layout

Goal: `plugins/` becomes `library/<functional>/` with one CMake helper
family, one `register.cxx` per library, one python package per library.
Kwiver stays a subdirectory; the helpers wrap the kwiver macros for now.
References: lite-library-layout.md §1, §3, §4; lite-build-system.md §3-4.

### P2-T01 CMake helpers (kwiver-backed)
Depends: P1-T10
Do:
- `cmake/viame_macros.cmake` implementing the signatures in lite-build-system.md §3 on top of `kwiver_add_library`, `kwiver_add_plugin`, `kwiver_add_python_module`, `kwiver_add_python_library`. `viame_add_python_package` globs the library directory's `**/*.py` (excluding `__pycache__`, tests) and installs each preserving relative paths under `viame/<name>/`; python sits alongside the C++, with no `python/` subdirectory.
- `viame_add_library(... REGISTER register.cxx)`: `register.cxx` defines `void register_<name>(kwiver::vital::plugin_loader&)` registering algorithms and processes in one function. Transitionally (until P8) the helper also builds a dynamic module `viame_<name>_plugin` whose `register_factories` calls that function, because kwiver's loader is still in use. Write `register.cxx` so it has no dependency on being in a module (no module-static state), so P8 can call it from `register_builtins()` unchanged.
- Alias support: `viame_register_alias(vpm, interface, name, alias)` helper that adds a second factory for the same class with attribute `viame.alias_of = name`. `registry-dump` emits it under `aliases`.
Done when:
- `plugins/examples` converted to the helpers as the pilot; build + BASELINE pass.

### P2-T02 Generate the file map
Depends: P2-T01
Do:
- Write `design/lite-file-map.tsv` (`source<TAB>destination`) covering every file under `plugins/` per lite-library-layout.md §3, and `design/scripts/apply_file_map.py` that performs `git mv`, creates directories, and rewrites `#include` lines for moved headers (`plugins/core/x.h`, `../core/x.h`, `<plugins/core/x.h>` -> `<viame/<dir>/x.h>`). Files in §4 (dead code) map to `DELETE`.
- Verify: every file under `plugins/` appears exactly once; every destination directory is in the layout doc.
Done when:
- `apply_file_map.py --check` reports 0 unmapped files.

### P2-T03 Skeleton `library/` with empty targets
Depends: P2-T02
Do:
- Create `library/CMakeLists.txt` adding every functional directory in DAG order (lite-library-layout.md §1). Each directory gets a `CMakeLists.txt` calling `viame_add_library` with empty source lists and a stub `register.cxx`, plus an `__init__.py` beside it with `__viame_register__` / `__sprokit_register__` stubs.
- Top-level: `add_subdirectory(library)` before `plugins`.
Done when:
- Build passes with both trees present (empty libraries link).

### P2-T04 Move `utilities`, `video_io`, `file_io`
Depends: P2-T03
Do:
- Apply the file map for those destinations. Fill their `CMakeLists.txt` (sources, gating: `TinyXML` for cvat, `ZLIB` for `camera_rig_io`, `VIAME_ENABLE_POSTGRESQL` for `file_io/database`). Move the corresponding registrations out of `plugins/core/register_algorithms.cxx` / `register_processes.cxx` into the new `register.cxx` files. Python: `viame.file_io` package with the coco modules; keep `viame.core` importing them for compatibility.
Done when:
- Build + `ctest -L BASELINE` pass (registry identical: same names, now registered from new modules).

### P2-T05 Move `image_processing`, `object_detectors`, `classifiers`, `segmentation`
Depends: P2-T04
Do:
- As P2-T04. Merge the two `windowed_*` sets now: keep the OpenCV-capable implementation, register both `windowed` and `ocv_windowed` (alias) for detector, refiner, trainer; drop the OpenCV-free copies. Golden: `detector_*windowed*` pipelines unchanged output (bitwise on detections CSV).
- `plugins/pytorch` detectors/classifiers/segmenters go to `<dir>/`; their `VIAME_ENABLE_PYTORCH-*` gating becomes `viame_add_python_package(... CONDITION ...)` per subgroup (split into `mmdet/`, `sam/`, ... subfolders where a gate applies).
Done when:
- Build + BASELINE + CRITICAL pass.

### P2-T06 Move `object_trackers`, `descriptors`, `measurement`
Depends: P2-T05
Do:
- As above. Merge `pair_stereo_detections`/`pair_stereo_tracks` core+opencv variants into `measurement/` (opencv variant is the superset; keep both registered names). Merge `measure_objects_process` (core) and `measure_using_stereo` (opencv) into one process class with two registered names. DB and CSV descriptor processes share one class with a `backend` chosen by registered name.
- Shared `kalman.py`/`track_state.py` in `object_trackers/common/` used by bytetrack, ocsort, deepsort, botsort; `reid_data.py` shared by deepsort/botsort trainers.
Done when:
- Build + BASELINE + CRITICAL + tracker unit tests pass; `tests/plugins/core/test_pair_stereo_detections.cxx` relocated and passing.

### P2-T07 Move `training`, `evaluation`, `examples`, `tools` links
Depends: P2-T06
Do:
- Trainers, netharn, `train_supervisor` (plugins/claude), `evaluate_models`, `plot_metrics` moved. `tools/CMakeLists.txt` links `viame_training`, `viame_evaluation`, `viame_utilities` instead of `viame_core`, `viame_claude`, `viame_opencv`. Templates in `library/examples/templates` rewritten for the new helpers; the stale `vital.algo` import fixed.
Done when:
- `viame train --help`, `viame score --help` work; build + BASELINE pass.

### P2-T08 Move remaining plugins and delete `plugins/`
Depends: P2-T07
Do:
- `onnx`, `darknet`, `svm`, `cppdb`, `vxl` (as-is for now; P3 replaces), `colmap`, `seagis`, `vertex-ai` per the map. Delete `plugins/` and `plugins/CMakeLists.txt`. `removed.json` gains the dead names from lite-library-layout.md §4.
- Python compatibility packages `viame.core`, `viame.pytorch`, `viame.opencv`, `viame.onnx`, `viame.colmap` re-export the moved modules and delegate registration to the new packages; `setup_viame.sh` `SPROKIT_PYTHON_MODULES` now lists `viame.<dir>` packages.
Done when:
- `plugins/` does not exist; build from clean; BASELINE, CRITICAL, unit tests pass; `git grep -l "plugins/" -- CMakeLists.txt cmake library tools` empty.

### P2-T09 Relocate tests
Depends: P2-T08
Do:
- `tests/plugins/*` -> `library/<dir>/tests/`, registered via `viame_add_test`; `tests/CMakeLists.txt` keeps examples/pipelines/tools/baseline.
Done when:
- `ctest -N | wc -l` equal to before the move; all pass.

### P2-T10 Docs and DIVE smoke
Depends: P2-T09
Do:
- Update `docs/manual` plugin-creation pages and `examples/plugin_creation` for `library/` and the helpers. Re-run the P1-T10 DIVE smoke.
Done when:
- Smoke passes; docs build (`VIAME_ENABLE_DOCS`) if enabled.
