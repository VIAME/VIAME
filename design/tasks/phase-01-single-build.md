# Phase 1: single build with code in place

Goal: delete the superbuild. One configure builds kwiver (as a
subdirectory), plugins, tools, tests. OpenCV, FFmpeg, Eigen still present.
No source files move.
References: lite-build-system.md §1-2, lite-dependencies.md §1, §3.

### P1-T01 Inventory the normal-build branch
Depends: P0-T05
Do:
- Extract the `else()` branch of the top-level `CMakeLists.txt` (`VIAME_BUILD_DEPENDENCIES` OFF path) into `cmake/viame_project.cmake` unchanged, and list every variable it consumes that the superbuild used to pass down (`fletch_DIR`, `kwiver_DIR`, `OpenCV_DIR`, `Darknet_DIR`, `VIAME_ARGS_*`, `PYTHON_*`, `CUDA_*`). Write the list to STATUS.md notes.
Done when:
- The list exists; nothing built yet.

### P1-T02 New top-level CMakeLists and options file
Depends: P1-T01
Do:
- Replace the top-level `CMakeLists.txt` with the shape in lite-build-system.md §1. Move all `option()`s into `cmake/viame_options.cmake`, deleting the options listed under "Removed in P1" and declaring `VIAME_ENABLE_WEB_EXCLUDES`, `VIAME_ENABLE_VERTEX_AI`.
- Keep the CUDA/cuDNN detection block and the PyTorch version matrix checks (they are still needed for wheels).
Done when:
- `cmake -S . -B build` configures up to the point of the first missing dependency; option list in `cmake -LH` matches the doc.

### P1-T03 `third_party/` for the small libraries
Depends: P1-T02
Do:
- Add `third_party/{eigen,tinyxml,libsvm,pybind11,cppdb}` with sources at the versions in lite-dependencies.md §1 and a `VERSION.md` each (upstream URL, version, local patches). libsvm carries fletch's HISTOGRAM kernel patch (compare with the fletch-built `svm.h`).
- `third_party/CMakeLists.txt` defines targets `Eigen3::Eigen` (INTERFACE), `tinyxml`, `svm`, `pybind11::module`, `cppdb` (only if `VIAME_ENABLE_POSTGRESQL`).
- Point `plugins/svm/CMakeLists.txt` and `plugins/cppdb/CMakeLists.txt` at these targets instead of `find_library`.
Done when:
- `cmake --build build --target svm tinyxml` succeeds.

### P1-T04 `viame_dependencies.cmake`
Depends: P1-T03
Do:
- `find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc imgcodecs calib3d video photo ml features2d objdetect highgui OPTIONAL_COMPONENTS ximgproc)`, `find_package(FFMPEG REQUIRED)` (copy kwiver's `FindFFMPEG.cmake` into `cmake/`), `find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)`, `ZLIB`, `Threads`, `OpenMP`, optional `CUDAToolkit`.
- Document in `docs/manual/building.md` the apt/dnf/brew package lists that satisfy these (`libopencv-dev` + contrib, `libavcodec-dev libavformat-dev libswscale-dev libavutil-dev libavfilter-dev`, `python3-dev`, `zlib1g-dev`).
Done when:
- Configure passes the dependency step on the reference Ubuntu machine with system packages.

### P1-T05 Kwiver as a subdirectory
Depends: P1-T04
Do:
- `add_subdirectory(packages/kwiver)` guarded by `EXCLUDE_FROM_ALL` off. Before it, set the `KWIVER_ENABLE_*` cache values exactly as `cmake/add_project_kwiver.cmake` did: ON for `ARROWS`, `TOOLS`, `SPROKIT`, `PROCESSES`, `BUILD_SHARED_LIBS`; OFF for `UUID`, `PROJ4`, `PROJ`, `GDAL`, `VTK`, `DBOW2`, `TESTS`, `FFMPEG_CUDA`, `KPF`, `TRACK_ORACLE`, `CERES`, `SUPER3D`, `PDAL`, `EXAMPLES`, `EXTRAS`, `SERIALIZE_JSON`, `SERIALIZE_PROTOBUF`; from VIAME flags: `FFMPEG`, `OPENCV`, `PYTHON`+`C_BINDINGS`, `PYTORCH`, `CUDA`; also force OFF the kwiver defaults VIAME never used: `KLV`, `GEOCALC`, `MVG` stays ON (triangulate), `ZLIB` ON (camera_rig_io). Plus `KWIVER_ENABLE_VXL` from `VIAME_ENABLE_VXL`, `fletch_DIR` unset, `KWIVER_PYTHON_USE_SYS_PATH=OFF`.
- Kwiver expects `find_package(fletch)`; patch its `CMake/kwiver-depends-*.cmake` minimally (in the submodule working tree, committed to our kwiver fork branch `viame/lite`) to skip fletch when `VIAME_LITE` is set.
- VXL: kwiver needs `VXL_DIR`. Since VXL is dropped in P3, build P1 with `VIAME_ENABLE_VXL=OFF` and accept that `vxl_*` names are missing from the registry until P3; record them in a temporary `tests/baseline/pending.json` consumed by `compare_registry.py --pending`. (This is the one place the contract is knowingly relaxed; P3-T08 empties the file.)
Done when:
- `cmake --build build` builds all kwiver targets; `viame registry-dump` diff vs baseline shows only the pending VXL names and nothing else.

### P1-T06 Plugins, tools, tests inside the same build
Depends: P1-T05
Do:
- `add_subdirectory(plugins)`, `tools`, `configs`, `examples`, `tests` from the new top level. Fix any `${KWIVER_SOURCE_DIR}`/`kwiver_DIR` uses to point at the subdirectory targets (`kwiver::vital` etc. are now plain targets; add `ALIAS` targets in `cmake/kwiver_aliases.cmake` if kwiver exports namespaced names only via its config file).
- `tools/CMakeLists.txt`: `vital/internal` include path now comes from the subdirectory.
- Delete `cmake/add_project_*.cmake`, `common_args.cmake`, `custom_*.cmake`, `setup_ctest.cmake`, `build_package.cmake`, `installation_blacklist.cmake`, `viame-install-fixup.cmake.in`, `linux-remove-duplicate-cvs.cmake`, `windows-ksys-hack.cmake`, `old_version_checks.cmake`, `FormatPassdowns*`/`OnDemandGitPackage` from `common_macros.cmake`.
Done when:
- `cmake --build build && cmake --install build` yields an install where `ctest -L BASELINE` and `ctest -L CRITICAL` pass (VXL names pending).

### P1-T07 Python dependency step
Depends: P1-T06
Do:
- Create `python/requirements/{base,cuda12,cuda13,cpu,learn,stereo,sam,colmap,test}.in` from the lists in the deleted `add_project_python_deps.cmake` and `add_project_pytorch.cmake` (`VIAME_PYTHON_DEPS_REQ_TORCH` included). Exclude `triton`, `wandb`, `castxml`, `pygccxml`, `decord` per `lite-install-size.md` §3 (after its §4 checks); `pycolmap`/`open3d` go to `colmap.in`. Compile `.lock` files with `pip-compile` on the reference machine.
- `cmake/viame_python_deps.cmake`: target `viame_python_deps` (in `all` when `VIAME_INSTALL_PYTHON_DEPS`) running one `pip install --user --no-deps -r <lock>` with the env from the old `PYTHON_DEP_ENV_VARS`, hash-stamped like `custom_pip_check_install.cmake`.
- Until the wheel index exists (P9), the forks are installed from `packages/pytorch-libs/*` with `pip wheel --no-build-isolation --no-deps` by a second target `viame_python_forks` reusing the source-hash guard from `custom_build_python_dep.cmake`.
- Carry over the post-install patches from `custom_install_viame.cmake` as a `python/patches/apply.py` step for now (removed in P9).
Done when:
- Clean `build/` -> configure -> build produces an install where `python -c "import torch, mmdet, viame.pytorch"` works under `setup_viame.sh`.

### P1-T08 Remove dead trees and submodules
Depends: P1-T07
Do:
- `git rm` submodules `packages/fletch`, `packages/vivia`, `packages/seal-tk`, `packages/tensorrt`, `packages/tensorflow-libs/models`, `packages/itk-modules/*`, `packages/pytorch-libs/{mmdet-to-tensorrt,fast-foundation-stereo}`. Delete `plugins/matlab`, `plugins/tensorflow` (open decision 4 acknowledged: matlab is removed; record the `matlab` image_object_detector and `lanl_scallop_finder` in `removed.json`), `cmake/build_server_*`, `cmake/build_cmake_*`, `cmake/launch_seal_interface.*`, `cmake/msi_*` (kept in git history; P10 re-creates what packaging needs), `cmake/setup_internal_python.*`, `packages/patches/fletch`.
- Delete the fake `opencv_python*.dist-info` logic; add `opencv-python-headless` to `base.in`.
- Install blacklist (`cmake/viame_install.cmake`): no `*.a` from `lib/`, no `bin/postgres`, sqlite CLI, GeographicLib/libtiff tools (`lite-install-size.md` §3 rows 1, 7).
Done when:
- `git submodule status` lists only kwiver, darknet, dive, pyav, pytorch-libs; build and BASELINE/CRITICAL still pass.

### P1-T09 Docker and CI for the single build
Depends: P1-T08
Do:
- `docker/lite.docker`: Ubuntu 22.04 + system deps from P1-T04 + CUDA base image variant; runs configure/build/install/`ctest -L "BASELINE|CRITICAL"`.
- `.github/workflows/lite.yml` (or the existing CI system): build the docker image on push to `lite`.
Done when:
- CI green on `lite`.

### P1-T10 DIVE smoke test
Depends: P1-T08
Do:
- Point DIVE desktop at `build/install`; run: open image folder, run `detector_default_fish.pipe` (or the generic detector), run a tracker, launch a short training, run scoring. Record results in STATUS.md.
Done when:
- All four actions succeed.
