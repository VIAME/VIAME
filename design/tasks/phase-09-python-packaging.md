# Phase 9: python packaging

Goal: forks and vendored python are wheels from an index; lock files are
the single source of truth; no install-time source patching.
References: lite-build-system.md §5, lite-dependencies.md §3.

## Restated 2026-09-28

**Do not start P9-T01, T02 or T03 as written.** The goal was reached by
another road for everything but the submodules, and those three tasks
describe infrastructure the project no longer needs in that form.
`lite-completion.md` section 5.1 has the comparison. In short:

| The goal's clause | State |
|---|---|
| lock files are the single source of truth | **met**: sixteen locks, installed `--no-deps` |
| no install-time source patching | **met**: removed in `168a97c30` |
| forks are wheels from an index | **not met, and perhaps not wanted**: forks build from submodules against `forks.lock`, patched by `apply_fork_patch.cmake`; `rfdetr` and `sam2` are vendored into `viame.*` |
| vendored python is wheels | **not met**: `library/object_detectors/learn` and `netharn` are first-party trees |

What is left is three tasks, below the original four.

### P9-T05 Decide whether the forks become wheels
Depends: -
Do:
- Open decision 9, asked again. An index was wanted so that a build would not compile `mmcv`. Record what a build costs now -- per fork, from a cold tree -- and whether the wheel the project publishes already covers the forks a user needs.
- If wheels: prove the path on one pure-python fork before any other, as `design/drafts/decision-09-wheel-index.md` recommends.
Done when:
- The decision and the measurement it rests on are in STATUS.md.

### P9-T06 Make `VIAME_ENABLE_PYTORCH-LEARN` importable
Depends: -
Do:
- `cutler` imports `pydensecrf` unconditionally. `pydensecrf` is in the tree, excluded from the install, absent from `learn.lock`, and its `setup.py` wants an `EIGEN_INCLUDE_DIR` nothing has set since P6. Either it is built and installed, with Eigen fetched for that one extension behind the option, or `pydensecrf` comes from PyPI through `learn.in` and the vendored copy is deleted.
- `panopticapi` likewise; its one importer guards the import, so it is the smaller half.
Done when:
- With the option on, `python -c "import viame.object_detectors.learn.cutler.crf"` succeeds in a fresh install.

### P9-T07 Submodules that nothing builds leave the tree
Depends: P9-T05
Do:
- For each of the 19, record which option builds it and whether any shipped pipeline reaches it. A submodule no option builds goes.
- `packages/python-utils/pyav` is the first candidate: P4 moved VIAME to the `av` wheel and the fork's patched `setup.py` existed to build against fletch's FFmpeg.
Done when:
- `git submodule status` lists only what an option builds, and `viame_python_forks.cmake` names nothing else.

## As originally written

### P9-T01 Wheel CI
Depends: P5-T07 (can run in parallel with P6-P8)
Do:
- New repo or workflow `viame-wheels`: builds each `packages/pytorch-libs/*` and `python-utils/pyav` with `packages/patches/*` applied, per (python, cuda) variant, publishes to the index (open decision 9). Versions `X.Y.Z+viame.N`.
Done when:
- Index serves wheels for the current lock set; documented in `docs/manual/wheels.md`.

### P9-T02 Vendored python -> wheels
Depends: P9-T01
Do:
- Package `pydensecrf`, `tokencut`, `cutler`, `panopticapi`, `remax`, `siammask` (library part), `mdnet` (with `roi_align` ext) from their current tree locations into the wheel CI; remove them from `library/`; import paths preserved via the wheel names (`viame_learn_extras` etc. re-exporting old module names where trainers import them).
Done when:
- `VIAME_ENABLE_PYTORCH-LEARN/SIAMMASK/MDNET` builds pull wheels; their pipelines pass.

### P9-T03 Remove submodules and install-time patches
Depends: P9-T02
Do:
- `git rm packages/pytorch-libs packages/python-utils`; delete `viame_python_forks` target and `python/patches/apply.py`; patched `torch_liberator`, `liberator`, `kwplot`, `iopath` come from the index.
Done when:
- `git submodule status` lists darknet and dive only; clean build works with `VIAME_INSTALL_PYTHON_DEPS=ON` from the index alone.

### P9-T04 Lock file hygiene
Depends: P9-T03
Do:
- One `pip-compile` run per variant in CI; a test that `pip check` passes in the install; `requirements/README.md` on how to bump.
Done when:
- CI job green.
