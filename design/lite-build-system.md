# VIAME lite: build system

## 1. Top-level shape

```cmake
cmake_minimum_required(VERSION 3.21)
project(VIAME VERSION 1.0.0 LANGUAGES C CXX)
include(cmake/viame_options.cmake)       # option()s, dependent options, sanity checks
include(cmake/viame_dependencies.cmake)  # find_package calls (shrinks every phase)
include(cmake/viame_macros.cmake)        # helpers in §3
add_subdirectory(third_party)
add_subdirectory(library)
add_subdirectory(tools)
add_subdirectory(python)
add_subdirectory(configs)
add_subdirectory(examples)
if(VIAME_ENABLE_TESTS) enable_testing(); add_subdirectory(tests) endif()
include(cmake/viame_install.cmake)       # setup scripts, add-on downloads, packaging
```

No `ExternalProject`, no `VIAME_BUILD_DEPENDENCIES`, no nested build.
`CMAKE_INSTALL_PREFIX` defaults to `build/install` so scripts and DIVE
settings keep pointing at the same place.

`viame_dependencies.cmake` by phase:

| After phase | Contents |
|---|---|
| P1 | `find_package(OpenCV)`, `find_package(FFMPEG)`, `find_package(Python3)`, `find_package(ZLIB)`, `Threads`, `OpenMP`, optional `CUDAToolkit`; `Eigen3::Eigen` from `third_party` |
| P3 | unchanged (VXL was only found inside kwiver) |
| P4 | minus FFMPEG |
| P6 | minus Eigen |
| P7 | minus OpenCV, minus ZLIB |
| end | `Python3`, `Threads`, optional `OpenMP`, optional `CUDAToolkit` (darknet only), optional `OpenSSL` (vertex-ai) |

## 2. Options

Kept: `VIAME_ENABLE_{CUDA,CUDNN,PYTHON,PYTORCH,ONNX,DARKNET,SVM,POSTGRESQL,
COLMAP,SEAGIS,DIVE,DOCS,TESTS,WEB_EXCLUDES,VERTEX_AI}`, `VIAME_ENABLE_PYTORCH-*`,
`VIAME_DOWNLOAD_MODELS*`, `VIAME_INSTALL_EXAMPLES`, `VIAME_VERSION_RELEASE`,
`VIAME_BUILD_DIVE_FROM_SOURCE`, `VIAME_PYTORCH_VERSION`.

Temporary (deleted in the phase that removes the dependency):
`VIAME_ENABLE_OPENCV` (P7), `VIAME_ENABLE_FFMPEG` (P4), `VIAME_ENABLE_VXL` (P3).

Removed in P1: `VIAME_ENABLE_{KWIVER,VIAME_PLUGINS,VIVIA,SEAL,KEYPOINT,MATLAB,
TENSORFLOW,TENSORRT,WIN32GUI,GDAL}`, `VIAME_BUILD_{FLETCH_DIR,KWIVER_DIR,PLUGINS_DIR,
FORCE_REBUILD,CORE_IMAGE_LIBS,PYTHON_FROM_SOURCE,PYTORCH_FROM_SOURCE,
TORCHVISION_FROM_SOURCE,LIMIT_NINJA,PACKAGING_CONT,CHECKS}`, `VIAME_FIXUP_BUNDLE`,
`VIAME_OPENCV_VERSION`, `VIAME_PYTHON_SYMLINK`, `VIAME_FORCE_CUDA_CSTD98`.

New: `VIAME_INSTALL_PYTHON_DEPS` (run pip as part of the build, default ON),
`VIAME_PYTHON_INDEX_URL`, `VIAME_PYTHON_STANDALONE` (P10: download
python-build-standalone into the install on Windows/desktop).

`CMakePresets.json` (P10) replaces `-C build_cmake_*.cmake`: `base`, `gpu`,
`cpu`, `desktop`, `docker-web`, `linux`, `windows`, `macos`, `ci`.

## 3. CMake helpers (`cmake/viame_macros.cmake`)

```cmake
viame_add_library(<name>
  [SOURCES ...] [HEADERS ...]
  [LINK_PUBLIC ...] [LINK_PRIVATE ...]
  [REGISTER register.cxx]        # register_<name>() called from register_builtins() (module only in P2-P7)
  [PYTHON]                       # installs <dir>'s .py files as viame.<name>
  [REQUIRES VIAME_ENABLE_X ...]) # whole library skipped if any is OFF

viame_add_sources(<name> CONDITION <expr>
  SOURCES ... [HEADERS ...] [LINK ...] [DEFINITIONS ...])

viame_add_python_package(<name> [DIRECTORY <dir>] [CONDITION ...] [EXCLUDE_GLOB ...])
viame_add_python_extension(<name> MODULE _x SOURCES ... PACKAGE viame.<name>)
viame_add_applets(<name> SOURCES ...)
viame_add_test(<name> SOURCES ... | PYTEST file.py [SOURCE_SETUP] LABELS ...)
viame_add_golden_test(<name> PIPELINE x.pipe INPUT <dir> EXPECTED <dir> TOLERANCE ...)
```

Each functional directory is a CMake library target `viame_<name>`.
`REGISTER` names the file whose `void register_<name>(viame::registry&)`
function is called at startup. Export headers `viame/<name>/export.h`;
headers install under `include/viame/<name>/`.

How `REGISTER` is realised changes once during the port:

| Phases | Mechanism |
|---|---|
| P2 to P7 | Kwiver's loader is still the registry, so `REGISTER` also builds a dynamic module `viame_<name>_plugin` into `lib/viame/plugins/`, exactly like `kwiver_add_plugin` did. Call sites do not know this |
| P8 onward | Every `viame_<name>` is an OBJECT/STATIC library folded into **one shared library `libviame`**; CMake generates `register_builtins.cxx` that calls each `register_<name>()` in DAG order. No modules, no plugin directory. `viame_<name>` targets remain for build organisation and tests only |

Until P5 the helpers wrap the kwiver macros (`kwiver_add_library`,
`kwiver_add_plugin`, `kwiver_add_python_module`) so P2 can land before the
kwiver import. P8 replaces the implementation; the call sites do not change.

## 4. Registration (static-first)

Design principle: the algorithm interfaces and the name registry stay; the
dynamic plugin loader goes. Everything built in tree registers statically.

- **One registry, one library.** `viame::registry` lives in
  `algorithm_framework/registry/` and is process-global because both the
  `viame` executable and the python extension `viame._core` link the single
  `libviame` shared library. It maps `(interface, name)` to a factory and
  carries an alias table and per-entry attributes (library of origin, python
  import path, description).
- **Built-ins.** `register.cxx` per functional library registers its
  algorithms, processes, and applets in one function. CMake generates
  `register_builtins.cxx` from the list of libraries with `REGISTER`; the
  executable and the python module both call `viame::register_builtins()`
  once. Because everything is linked explicitly there is no unreferenced-
  object problem and no static-initialiser ordering trick.
- **Python implementations register lazily.** `viame.<dir>/__init__.py`
  declares entries as `(interface, name, "viame.<dir>.module:Class")`
  without importing the module; `viame::registry` stores the import path and
  imports on first instantiation. `registry-dump` can list them without
  importing torch. Add-ons that ship python declare the same way through
  `VIAME_PYTHON_PLUGINS` (a list of packages to scan), which replaces
  `SPROKIT_PYTHON_MODULES`. Old names `viame.core`, `viame.pytorch`,
  `viame.opencv`, `viame.onnx` remain as re-export packages that forward
  their declarations.
- **Aliases.** `reg.add<Impl>("convert_image").alias("vxl_convert_image")`.
  The alias resolves to the same factory; first use logs at debug level.
  `registry-dump` lists aliases separately so the baseline diff can tell a
  rename from a removal. Until P8 the alias is a second factory registration
  of the same class in kwiver's loader; P8 makes it a first-class table.
  Process types get the same alias table, resolved in the pipe bakery.
- **Optional native code.** darknet and cppdb are compiled in when their
  option is on; a build with them requires their libraries at runtime, which
  is already true today. If open decision 10 wants darknet to be loadable on
  CUDA-less machines from the same build, it becomes the one dlopen'd module
  through the hook below.
- **Out-of-tree C++ plugins (open decision 10).** If kept: a ~150-line
  `external_plugins.cxx` that iterates `VIAME_PLUGIN_PATH`, dlopens each
  file, and calls its exported `viame_register_plugin(viame::registry&)`.
  No directory scanning by default, no attribute files, no
  `KWIVER_PLUGIN_PATH`. If dropped: `examples/plugin_creation` documents
  building in tree (add a directory under `library/`) or writing python.
- **Startup budget.** `viame --version` and `viame runner --help` must
  complete in under 0.5 s with no python import; a benchmark test enforces
  it from P8-T10.

## 5. Python build

- `python/CMakeLists.txt` installs `viame/__init__.py`, the `kwiver` shim
  (P11), and calls `viame_add_python_package` for each library, whose `.py`
  files sit alongside its C++ rather than in a `python/` subdirectory.
- Bindings: `core_types/` (types with numpy buffer views),
  `algorithm_framework/` (config, algorithm trampolines, registry
  access), `pipeline_framework/` (process, datum, port, pipeline,
  scheduler), each alongside the C++ it wraps. All compiled into one extension module `viame._core` that
  links `libviame`, so C++ and python share the registry. Until P8 these
  are the copied kwiver bindings incl. the castxml step; P8 replaces them
  with hand-written pybind11 and deletes castxml.
- Registration from python is declarative (§4): packages list their
  implementations; nothing heavy is imported until an implementation is
  instantiated.
- Dependencies: `python/requirements/*.in` compiled to `*.lock`. Target
  `viame_python_deps` runs
  `pip install --user --no-deps -r <lock> --extra-index-url <VIAME_PYTHON_INDEX_URL>`
  with `PYTHONUSERBASE=<install>` and `PYTHONNOUSERSITE=1`, hash-stamped.

## 6. Install layout

```
install/
  bin/viame[.exe]                       single entry point
  setup_viame.sh|.bat                   env setup (DIVE sources this)
  viame_train_detector
  lib/libviame.so                       all C++ (P8 onward; lib/viame/plugins/*.so only in P2-P7)
  lib/python3.X/site-packages/viame/    package + _core extension linking libviame
  lib/python3.X/site-packages/kwiver/   shim (P11)
  include/viame/<dir>/                  headers for out-of-tree plugins
  lib/cmake/viame/                      viame-config.cmake
  configs/pipelines/, configs/*.py      unchanged
  examples/, doc/
```

`setup_viame.sh` after P10: `VIAME_INSTALL`, `PATH`, `LD_LIBRARY_PATH`,
`PYTHONPATH`, `PYTHONNOUSERSITE`, `VIAME_LOG_LEVEL` (`KWIVER_DEFAULT_LOG_LEVEL`
still read), optional CUDA paths. No plugin path variables; `VIAME_PLUGIN_PATH`
and `VIAME_PYTHON_PLUGINS` exist only for add-ons and are unset by default.

## 7. Python provisioning

There is no deps bundle: after P7 the only native prerequisite is Python.
Linux/macOS use the system or pyenv interpreter. Windows and desktop
release builds set `VIAME_PYTHON_STANDALONE=ON`, which downloads a pinned
python-build-standalone archive into `install/python` at configure time and
uses it for both bindings and pip.

## 8. Packaging

- Linux: `cpack -G TGZ` of the install tree.
- Windows: `install(RUNTIME_DEPENDENCY_SET ...)` for the remaining DLLs
  (python, CUDA runtime if darknet); `msi_generate_installer.py` reads the
  tree. `fixup_bundle`, `installation_blacklist.cmake`,
  `viame-install-fixup.cmake.in` deleted in P1.
- Docker: one Dockerfile, build args select the preset.

## 9. Tests

`tests/` keeps the pytest helper (renamed `viame_add_test`) and ctest-time
pipeline discovery. Unit and golden tests live in `library/<dir>/tests/`
and are collected by `viame_add_library`. `setup_ctest.cmake` and the
`CTestTestfile.cmake` append hack go in P1.
