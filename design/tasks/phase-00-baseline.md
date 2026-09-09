# Phase 0: baseline and compatibility contract

Goal: capture what the current build registers and what every pipeline
resolves to, as machine-checkable JSON, before anything moves.
Prerequisite: a working `main` superbuild install at `build/install`.
References: lite-plan.md §5, AGENT_GUIDE.md.

### P0-T01 Create the `lite` branch
Depends: -
Do:
- `git checkout -b lite main`. Commit the `design/` folder if not yet committed.
Done when:
- `git branch --show-current` prints `lite`; `design/` is tracked.

### P0-T02 `registry-dump` applet
Depends: P0-T01
Do:
- Add `tools/registry_dump.{h,cxx}` as a viame applet (pattern: `tools/get_configs.cxx`, registered in `tools/register_applets.cxx`).
- Load all plugins (`kwiver::vital::plugin_manager::instance().load_all_plugins()`), then emit JSON: for every algorithm interface -> impl name -> sorted config keys with defaults and descriptions; every sprokit process type -> config keys and input/output port names; every applet name; every python module reachable via `SPROKIT_PYTHON_MODULES` that registers something. Include a top-level `aliases: {}` object (empty until aliases exist).
- Deterministic ordering, stable key names. Write to stdout or `--output`.
Done when:
- `viame registry-dump --json > /tmp/r.json` succeeds after `source build/install/setup_viame.sh`; running twice gives identical files.

### P0-T03 `pipe-check` applet
Depends: P0-T01
Do:
- Add `tools/pipe_check.{h,cxx}`: for a `.pipe` path, run sprokit's pipe bakery (`sprokit::bake_pipe_from_file`) with the same include paths the runner uses, then for each process report its type and, for every `:type` key under it, whether the impl name resolves. Output JSON: `{file: {status: ok|error, message, processes: {name: {type, algos: {key: impl}}}}}`.
- `--all` walks `configs/pipelines`, `configs/add-ons`, `examples`, and every zip in `cmake/download_viame_addons.csv` already present under `packages/downloads` (extract to a temp dir). `.conf` files are checked with `kwiver::vital::read_config_file` and algorithm resolution only.
- Process `pythread_per_process` scheduler references and `include` statements must work (use the install `configs/pipelines` as include root).
Done when:
- `viame pipe-check --all --json > /tmp/p.json` succeeds; every file in `configs/pipelines` appears; today's known-stale names (`pytorch_augmentation`, `mdnet_tracker`) show as errors and nothing else does.

### P0-T04 Baseline files and compare scripts
Depends: P0-T02, P0-T03
Do:
- Commit `tests/baseline/registry.json`, `tests/baseline/pipes.json` from the current `main` install (GPU build with default options, plus a CPU build merged in: union of registrations, marked with `configs: [gpu, cpu]`).
- `tests/baseline/removed.json` = `[]` initially, entries `{kind, interface, name, phase, reason}`.
- `tests/baseline/compare_registry.py old new --removed removed.json`: fails on any missing name, missing config key, changed default; ignores descriptions; treats an alias as satisfying a name. `compare_pipes.py`: fails on any file whose status or resolved impls changed.
- Record the `CRITICAL` label list from `tests/examples/CMakeLists.txt` in `tests/baseline/critical.txt`.
Done when:
- Both compare scripts exit 0 against a fresh dump of the same install and exit non-zero when a name is deleted from the new dump by hand.

### P0-T05 Wire baseline checks into ctest
Depends: P0-T04
Do:
- `tests/CMakeLists.txt`: add tests `baseline:registry` and `baseline:pipes` (label `BASELINE`) that run the dump + compare. They run against the install prefix like the other SOURCE_SETUP tests.
Done when:
- `ctest -L BASELINE` passes on the current install.

### P0-T06 Merge a CPU-build registry dump into the baseline
Depends: P0-T05. Independent of phase 1 onward; may run at any time.
Added by: P0-T04, which recorded a CUDA-only baseline.
Do:
- Configure a second build with `VIAME_ENABLE_CUDA=OFF` and `VIAME_ENABLE_CUDNN=OFF`, everything else as the reference build, into its own prefix.
- `viame registry-dump --json` from it; merge into `tests/baseline/registry.json` as the union of the two, each entry gaining `configs: [gpu]`, `[cpu]` or `[gpu, cpu]`.
- Teach `compare_registry.py` a `--config {gpu,cpu}` option that only requires the entries carrying that config, so either build can be checked.
Done when:
- `compare_registry.py --config cpu` passes against the CPU dump and `--config gpu` against the GPU dump, and both fail when a name of that config is deleted by hand.
