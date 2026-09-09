# Phase 5: import the kwiver subset

Goal: the used kwiver code lives in `library/core_types`,
`library/algorithm_framework`, `library/pipeline_framework`, and the
functional dirs; `packages/kwiver` is gone. Code is copied, not rewritten
(P6-P8 do that). Namespaces and include paths inside the copied code are
rewritten to the new locations. References: lite-library-layout.md §2,
lite-removals.md §5, lite-dependencies.md §2.

### P5-T01 Reachability list
Depends: P4-T05
Do:
- Script `design/scripts/kwiver_reachable.py`: starting from every header included by `library/` and `tools/`, plus the registrations named in `registry.json` (minus `removed.json`), plus the sprokit engine/runner/adapters, compute the transitive closure of kwiver source files (parse `#include` lines; map registration names to files via `grep -l "\"<name>\""` in `arrows/*/register_algorithms.cxx` and `sprokit/processes/*/register_processes.cxx`). Output `design/lite-kwiver-files.txt`.
- Sanity: `arrows/{klv,serialize,dbow2,vtk,kpf,ceres,qt,super3d,geocalc,gdal,cuda,zlib,pdal,proj,uuid,matlab}` and all `tests/` must be absent; if present, find the include that pulls them and cut it (note in STATUS.md).
Done when:
- List committed; line count of listed files recorded (expect roughly 120k-150k).

### P5-T02 Copy `vital` into `core_types` and `algorithm_framework`
Depends: P5-T01
Do:
- Copy listed `vital/types/*` -> `library/core_types/`, the rest of listed `vital/*` -> `library/algorithm_framework/<sub>/` per lite-library-layout.md §2. `vital/kwiversys` -> `third_party/kwiversys`, `vital/internal/cereal` -> `third_party/cereal` (+ its rapidjson to `third_party/rapidjson`), `cxxopts.hpp` -> `third_party/cxxopts`.
- Rewrite includes in the copied files and in `library/`, `tools/`: `vital/types/X` -> `viame/core_types/X`; other `vital/Y` -> `viame/algorithm_framework/Y`; `kwiversys/` unchanged (third_party include dir); `vital/internal/cereal` -> `cereal/`.
- `viame_add_library(core_types)`, `viame_add_library(algorithm_framework)` with the kwiver CMake logic for export headers, `vital_config.h` generation, python bindings (`python/kwiver/vital/*` -> `core_types/`, `algorithm_framework/`, alongside the C++ they wrap, including the castxml generation step, moved verbatim into `cmake/kwiver_compat/`).
- Kwiver still added as subdirectory for sprokit and arrows at this point; its vital is built but nothing links it (temporary duplication for one task).
Done when:
- `library/` and `tools/` compile against the copied vital; BASELINE passes.

### P5-T03 Copy sprokit into `pipeline_framework`
Depends: P5-T02
Do:
- Per the map: engine, pipeline_util, schedulers, runner applet, adapters, `kwiver_type_traits.h` -> `type_traits.h`, python bindings and `pythread_per_process`. Include rewrite `sprokit/pipeline/X` -> `viame/pipeline_framework/X`, `sprokit/processes/adapters/X` -> `viame/pipeline_framework/adapters/X`, `sprokit/processes/kwiver_type_traits.h` -> `viame/pipeline_framework/type_traits.h`.
- `downsample_process`, `filter_frame*`, `image_to_image_set` live in `pipeline_framework/processes`.
Done when:
- `viame runner` runs `configs/pipelines/detector_simple_hough.pipe` (or another dependency-free pipe) from the new build; BASELINE passes.

### P5-T04 Copy used arrows and processes into functional dirs
Depends: P5-T03
Do:
- Per lite-library-layout.md §2 table: `arrows/core` (5 used registrations + helpers), `arrows/ocv` (7 used + `image_container`, `camera_intrinsics`), `arrows/mvg` (`triangulate` + camera helpers), used `sprokit/processes/core/*`, `ocv/image_viewer_process`. Each dir's `register.cxx` absorbs the registrations. Not copied: the 25 unused processes (`close_loops`, `collate`, `compute_homography`, `deserializer`, `detect_features`, `detect_features_if_keyframe`, `distribute`, `draw_tracks`, `extract_descriptors`, `feature_matcher`, `feature_tracker`, `file_transport_send`, `frame_list_input`, `keyframe_selection`, `multiplexer`, `pass`, `print_config`, `read_d_vector`, `read_track_descriptor`, `serializer`, `shift_detected_object_set`, `sink`, `test`, `zmq_transport_receive`, `zmq_transport_send`) and unused arrow registrations; all go to `removed.json` with `reason: never referenced by a pipeline`.
Done when:
- BASELINE passes (registry identical minus the new removed entries); CRITICAL passes.

### P5-T05 Remove the kwiver submodule
Depends: P5-T04
Do:
- `git rm packages/kwiver`; delete `add_subdirectory(packages/kwiver)`, `cmake/kwiver_aliases.cmake`, kwiver option plumbing. Kwiver CMake macros still needed by `viame_macros.cmake` live in `cmake/kwiver_compat/` (already copied in P5-T02).
- `viame-config.cmake` export for out-of-tree plugins (`lib/cmake/viame/`) replacing `kwiver-config.cmake`; `examples/plugin_creation` builds against it.
Done when:
- Build from clean; `git submodule status` lists darknet, dive, pyav, pytorch-libs only; BASELINE, CRITICAL, GOLDEN, unit tests pass.

### P5-T06 Prune unused `vital` code by compile
Depends: P5-T05
Do:
- Delete `core_types` headers nothing includes (start with `metadata*`, `geo_*`, `geodesy`, `mesh*`, `pointcloud`, `activity*`, `video_raw_*`, `video_settings`, `video_uninterpreted_data`, `tiled_image_*`, `sfm_constraints`, `ground_control_point`, `local_tangent_space`, `camera_rpc`, `protobuf`, `class_map`, `context`, `signal`; keep everything the P5-T01 reachability list names) and their `.cxx`, then rebuild; repeat until the include graph is closed. Same for `algorithm_framework/util` and `algo` interfaces no implementation or process uses (keep the 18 python-implemented interfaces even if C++ has no impl).
- Remove Boost/log4cxx/log4cplus/ZeroMQ/Protobuf detection remnants from `cmake/kwiver_compat/`.
Done when:
- Line count of `library/{core_types,algorithm_framework}` recorded in STATUS.md; all tests pass.

### P5-T07 DIVE smoke and docs
Depends: P5-T06
Do:
- Smoke as P1-T10. Update `docs/manual` references from kwiver to the new locations where they name paths.
Done when:
- Smoke passes.
