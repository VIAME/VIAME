# VIAME `lite` branch design

Plan for turning VIAME into a single CMake project with no third-party
C++ library dependencies (no fletch, kwiver, OpenCV, VXL, Eigen, or FFmpeg)
and a source tree organised by function under `library/`.

## Documents

| Document | Role |
|---|---|
| [AGENT_GUIDE.md](AGENT_GUIDE.md) | How to execute this plan one task at a time: operating loop, invariants, verification commands, commit rules. Start here if you are doing the work |
| [STATUS.md](STATUS.md) | Ledger of every task: status, commit, notes, decisions taken. Updated after every task |
| [tasks/phase-NN-*.md](tasks/) | Ordered, atomic tasks per phase, each with dependencies, steps, and a checkable "done when" |
| [lite-plan.md](lite-plan.md) | Goals, end state, guiding decisions, phase overview, risks, open decisions |
| [lite-removals.md](lite-removals.md) | Per-dependency removal design: what each of VXL, OpenCV, FFmpeg, Eigen, kwiver is used for, and what replaces each use |
| [lite-dependencies.md](lite-dependencies.md) | Every current dependency with its disposition and the end-state list of what remains vendored |
| [lite-library-layout.md](lite-library-layout.md) | Target `library/` tree and source-to-destination mapping for `plugins/` and retained kwiver code |
| [lite-build-system.md](lite-build-system.md) | CMake design: options, helper functions, registration and aliases, python packaging, install layout, setup scripts |
| [lite-install-size.md](lite-install-size.md) | Install size audit (14 GB breakdown), why triton is 697 MB, and a ranked list of size-reduction candidates with verification steps |

Reference docs (`lite-*.md`) describe the end state and the reasoning.
Task files describe the path. When they disagree, the task file wins for
"what to do now" and the disagreement is logged in STATUS.md.

## Survey baseline (2026-09-08, `main` @ 637e275ab)

| Component | Lines | Notes |
|---|---:|---|
| `packages/kwiver` | ~427k | ~145k reachable from VIAME; sprokit engine has no Eigen dependency; Eigen sits in 35 of 151 `vital/types` files |
| `plugins/` | ~324k | ~200k is vendored python/C++ under `plugins/pytorch`; 9 files use Eigen directly; OpenCV used in ~60 C++ files |
| `configs/` + `examples/` | 355 `.pipe`, 88 `.conf` | The compatibility contract |
| fletch packages on a default build | 26 | All go: 6 become vendored headers/sources, the rest are removed or become python wheels |
