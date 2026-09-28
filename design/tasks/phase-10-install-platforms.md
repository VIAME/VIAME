# Phase 10: install layout, platforms, packaging

References: lite-build-system.md §2, §6-8.

## State 2026-09-28

T01, T02 and T05 are done. T03 is in progress, T04 is not started, T06 is
drafted and waiting on the user.

Three things the tasks below did not anticipate:

* **`setup_viame.sh` exports `VIAME_LOG_LEVEL`**, which takes precedence over
  `KWIVER_DEFAULT_LOG_LEVEL`. A test that sets the second and sources the
  first tests nothing.
* **The runtime image needs `libgl1`.** With no imaging library named, pip
  installs both OpenCV distributions and the full one needs `libGL.so.1`.
* **The Windows scripts still describe `main`.** `build_server_windows_msi`
  has stages for VIVIA and SEAL, removed in P1, and sets eight options
  nothing reads. Two of those, the from-source torch flags, are wanted back
  (P12-T06); the rest are decisions 11 and 12 in `lite-completion.md`. Do
  not delete them as part of T03 without the user's say.

### P10-T01 Install layout and setup script
Depends: P8-T09
Do:
- Single `lib/libviame.so` plus `viame._core`; `setup_viame.{sh,bat}.in` reduced to the variables in lite-build-system.md §6 (no plugin path variables); `viame_train_detector` and `configs/*.py` paths unchanged. `KWIVER_PLUGIN_PATH`, if set by an old environment, is ignored with a one-time warning.
Done when:
- DIVE smoke passes with the new install; `env | grep -c "SPROKIT\|KWIVER_PLUGIN"` is 0 after sourcing.

### P10-T02 CMakePresets
Depends: P10-T01
Do:
- `CMakePresets.json` per lite-build-system.md §2; delete remaining `cmake/build_*` scripts; `docs/manual/building.md` updated.
Done when:
- `cmake --preset linux-gpu && cmake --build --preset linux-gpu` works.

### P10-T03 Windows
Depends: P10-T02
Do:
- `VIAME_PYTHON_STANDALONE` download step; MSVC build of the whole tree; `install(RUNTIME_DEPENDENCY_SET)`; `msi_generate_installer.py` restored from history and pointed at the new tree; darknet optional.
Done when:
- Windows CI produces an installer; DIVE Windows smoke passes.

### P10-T04 macOS
Depends: P10-T02
Do:
- Build and unit tests on macOS arm64 (no CUDA, no darknet); `build_server_mac.sh` replaced by a preset.
Done when:
- CI green.

### P10-T05 Docker images and release packaging
Depends: P10-T02
Do:
- One Dockerfile with preset arg; `default`, `web`, `everything` images; `cpack -G TGZ` excluding `include/` and `lib/cmake` unless `VIAME_PACKAGE_DEVEL=ON`; add-on model download at install still works (`download_viame_addons.sh`); base package carries only the default model packs (`lite-install-size.md` §3 rows 4, 18).
Done when:
- Images build; `ctest -L CRITICAL` inside the `web` image passes.

### P10-T06 Release notes and migration guide
Depends: P10-T05
Do:
- `RELEASE_NOTES.md`: removed names (`removed.json`), aliases, env var changes, include path shims (`include/viame/compat/`), python module renames, how add-ons should migrate.
Done when:
- Reviewed by the user.
