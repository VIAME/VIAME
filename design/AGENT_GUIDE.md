# Agent guide: executing the lite plan

This plan is meant to be executed incrementally by a coding agent (or a
person) one task at a time, with each task leaving the branch buildable and
the compatibility checks green.

## Operating loop

0. Read `lite-findings.md` before starting a phase. It records what the work
   so far has taught that the plan did not know, and every entry there is
   something that already cost a debugging session once.
1. Read `STATUS.md`. Pick the first task whose status is `todo` and whose
   `Depends` are all `done`. Do not skip ahead within a phase unless the
   task says it is independent.
2. Open the task in `tasks/phase-NN-*.md`. Read the reference sections it
   links. If the task's file lists look stale, regenerate them with the grep
   or script the task names and note the delta in STATUS.md; do not silently
   widen or narrow scope.
3. Do the work. Stay inside the task's scope. If you discover adjacent work,
   add a new task at the end of the phase file (ID `Pn-Txx`, next free
   number) instead of doing it now.
4. Run every command under the task's **Done when**. All must pass. If one
   cannot pass, set the task to `blocked` with the reason and stop; do not
   weaken a check to make it pass.
5. Commit: one task per commit, subject `lite: <task id> <short summary>`,
   body of at most three lines. No `Co-Authored-By` trailers. Do not push
   unless asked.
5b. If the work taught something a later phase needs to know, add it to
   `lite-findings.md`. The bar is: would someone doing a later task get this
   wrong without it?
6. Update `STATUS.md`: status, commit hash, one-line note (what was
   different from the task text, decisions taken, follow-ups added).
7. Stop after one task unless instructed to continue for N tasks or a
   whole phase.

## Invariants (hold after every task)

- `cmake --build build` from the current configure succeeds, and a clean
  configure from scratch succeeds at every phase boundary.
- Registry check passes: `viame registry-dump --introspect` matches
  `tests/baseline/registry.json`, allowing only entries listed in
  `tests/baseline/removed.json`. A task that removes or renames a name must
  add it to `removed.json` (renames register an alias instead and stay out
  of `removed.json`). **`registry.json` is never regenerated whole.** It is
  `main`'s surface, not this build's, and a fresh dump drops every name
  `removed.json` gives a reason for; `tests/baseline/README.md` says where
  each kind of change goes.
- A python process is declared in its package's
  `__sprokit_process_declarations__`. A package that declares anything is
  not scanned, so after a merge from `main` a process that arrived with
  only its `__sprokit_register__` hook does not exist.
  `baseline:declared_processes` checks it.
- No python that VIAME owns imports `cv2`, at import time or at call time,
  outside `tests/reference/opencv/`. `baseline:lazy_cv2` checks it.
- Pipe check passes: `viame pipe-check --all` matches
  `tests/baseline/pipes.json`.
- `ctest -L CRITICAL` passes on the reference machine.
- No new third-party dependency is added unless a task says so. The
  allowed end-state list is in `lite-dependencies.md` §4.
- Algorithm behaviour does not change except where a task installs a
  replacement with a golden test and a documented tolerance.
- Config keys and their defaults are preserved for every registered name.
- Existing user-visible entry points keep working: `viame <applet>`,
  `setup_viame.sh`, `viame_train_detector`, `configs/pipelines/*`,
  add-on zips from `cmake/download_viame_addons.csv`, DIVE desktop.

These checks do not exist until Phase 0 creates them. Phase 0 tasks say
what to build; from P0-T05 onward the checks are mandatory.

## Verification commands

Run from the repo root. `build/` is the configure directory,
`build/install` the install prefix (unchanged from today's layout).

```
# configure + build (Phase 1 onward; before that use the existing superbuild)
cmake -S . -B build -DVIAME_ENABLE_CUDA=ON  # or --preset linux-gpu once presets exist
cmake --build build -j$(nproc) && cmake --install build

# compatibility contract
source build/install/setup_viame.sh
viame registry-dump --json --introspect --output /tmp/registry.json
python3 tests/baseline/compare_registry.py tests/baseline/registry.json /tmp/registry.json \
        --removed tests/baseline/removed.json --pending tests/baseline/pending.json
viame pipe-check --all --json --output /tmp/pipes.json
python3 tests/baseline/compare_pipes.py tests/baseline/pipes.json /tmp/pipes.json \
        --removed tests/baseline/removed_pipes.json

# behaviour. -j2, not -j$(nproc): the reference machine is shared, and at
# -j3 seven unit tests fail under load and pass on a rerun
ctest --test-dir build -L "BASELINE|UNIT|CORE" -j2 --output-on-failure
ctest --test-dir build -L "GOLDEN|CRITICAL"    -j1 --output-on-failure
ctest --test-dir build -R '^tools:'            -j2 --output-on-failure

# hygiene
git grep -n "#include <vital/"   -- library tools   # must be empty after P5
git grep -n "#include <opencv2/" -- library tools   # must be empty after P7
git grep -n "Eigen::"            -- library tools   # must be empty after P6
git grep -n "libav\|avcodec"     -- library tools   # must be empty after P4
```

## Conventions

- Keep tests under the root `tests/` tree. Library tests mirror their source
  component at `tests/library/<component>/`, including fixtures and nested
  Python suites. Register them from the root tests CMake tree.

- C++17, format with `cmake/style.clang_format`. Comments only for
  non-obvious "why".
- New C++ files carry the existing VIAME license header.
- Functional directories only: no directory or target named after a
  dependency. Per-file gating uses `viame_add_sources(CONDITION ...)`.
- Registered names never disappear: a renamed implementation registers
  its old name as an alias (`lite-build-system.md` §4). A removal goes to
  `removed.json` and is called out in STATUS.md.
- Reference tests: input from `pipelines_test_data`, expected output
  committed under `tests/reference/<group>/`, tolerance stated in the test
  file. A replacement implementation is not done until its recording
  exists. (`tests/golden` until 2026-09-28; the ctest names are still
  `golden:*`.)
- A test that **calls** the library being replaced lives in
  `tests/reference/opencv/` and begins with `pytest.importorskip`. A test
  that only cites it as where its expected values came from does not belong
  there, and a skip is never conditioned on whether a reference library has
  a feature that VIAME provides itself.
- Neither C++ nor python uses OpenCV. What python needs is in
  `viame.image_kernels`, `viame.utilities.imageops`,
  `viame.utilities.geometry` and `viame.video_io.frames`; if a function is
  missing, it is added there. There is no compatibility module and one is
  not to be written.
- Images are RGB, or RGBA, inside VIAME. A channel swap happens only at a
  boundary that demands BGR, and is written at that boundary.
- Every new python module lives in `library/<dir>/` alongside that
  library's C++ and is picked up by `viame_add_python_package`; no per-file
  CMake lines and no `python/` subdirectory.
- Built-in code registers statically through its library's `register.cxx`
  (or the package's lazy declaration list for python). Do not add new dlopen
  modules, plugin directories, or plugin-path environment variables; the
  dynamic modules that exist in P2 to P7 are transitional.
- When the mapping in `lite-library-layout.md` and reality disagree, follow
  the functional rule (put the file where its function is) and record the
  choice in STATUS.md.

## Decisions you may not make alone

The open decisions in `lite-plan.md` §7 and `lite-completion.md` §6. If a
task needs one, mark the task `blocked (decision N)` and stop. Do not pick a
side to keep moving.

**Removing something is a decision.** A flag nothing reads, a submodule
nothing builds, a comment that names a library that is gone: each has
looked dead and been wanted. "Removed in P1" has meant deferred. Report it,
with what reads it and what does not, and leave it where it is.

## What is not yours to edit

- `RELEASE_NOTES.md`. Drafts go in `design/drafts/`.
- Anything under `packages/` other than `packages/patches/`. The forks are
  submodules; a change to one is a diff in `packages/patches/<fork>.patch`
  and the gitlink does not move.
- DIVE. It is its own repository and the pipelines are its contract.

## Reading order for a fresh agent

1. This file
2. `STATUS.md`
3. `lite-completion.md` (what is done, what is wrong, what is next)
3b. `lite-plan.md` §1 to §3 (goals, end state, decisions)
4. The current phase's task file, fully
5. The sections of `lite-removals.md`, `lite-library-layout.md`,
   `lite-build-system.md` that the task links
