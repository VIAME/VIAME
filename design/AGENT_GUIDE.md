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
- Registry check passes: `viame registry-dump` matches
  `tests/baseline/registry.json`, allowing only entries listed in
  `tests/baseline/removed.json`. A task that removes or renames a name must
  add it to `removed.json` (renames register an alias instead and stay out
  of `removed.json`).
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
viame registry-dump --json > /tmp/registry.json
python3 tests/baseline/compare_registry.py tests/baseline/registry.json /tmp/registry.json \
        --removed tests/baseline/removed.json
viame pipe-check --all --json > /tmp/pipes.json
python3 tests/baseline/compare_pipes.py tests/baseline/pipes.json /tmp/pipes.json

# behaviour
ctest --test-dir build -L CRITICAL --output-on-failure
ctest --test-dir build -L GOLDEN   --output-on-failure   # exists from P3-T03

# hygiene
git grep -n "#include <vital/"   -- library tools   # must be empty after P5
git grep -n "#include <opencv2/" -- library tools   # must be empty after P7
git grep -n "Eigen::"            -- library tools   # must be empty after P6
git grep -n "libav\|avcodec"     -- library tools   # must be empty after P4
```

## Conventions

- C++17, format with `cmake/style.clang_format`. Comments only for
  non-obvious "why".
- New C++ files carry the existing VIAME license header.
- Functional directories only: no directory or target named after a
  dependency. Per-file gating uses `viame_add_sources(CONDITION ...)`.
- Registered names never disappear: a renamed implementation registers
  its old name as an alias (`lite-build-system.md` §4). A removal goes to
  `removed.json` and is called out in STATUS.md.
- Golden tests: input from `pipelines_test_data`, expected output committed
  under `tests/golden/<name>/`, tolerance stated in the test file. A
  replacement implementation is not done until its golden test exists.
- Python that needs OpenCV imports `cv2` from the `opencv-python-headless`
  wheel. C++ never includes OpenCV after Phase 7.
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

The open decisions in `lite-plan.md` §7. If a task needs one, mark the task
`blocked (decision N)` and stop. Do not pick a side to keep moving.

## Reading order for a fresh agent

1. This file
2. `STATUS.md`
3. `lite-plan.md` §1 to §3 (goals, end state, decisions)
4. The current phase's task file, fully
5. The sections of `lite-removals.md`, `lite-library-layout.md`,
   `lite-build-system.md` that the task links
