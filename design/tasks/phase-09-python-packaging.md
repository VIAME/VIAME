# Phase 9: python packaging

Goal: forks and vendored python are wheels from an index; lock files are
the single source of truth; no install-time source patching.
References: lite-build-system.md §5, lite-dependencies.md §3.

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
