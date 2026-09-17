# Decision 9: hosting for the python wheel index

Written because the question as asked ("where should the index live?") did
not carry enough with it to answer. This says what the index is for, what
would be on it, what breaks without it, and what each option costs. Nothing
here is decided; the recommendation at the end is a recommendation.

## What the index is for

VIAME builds 17 python packages from source at install time. They are
research repositories with no wheel on any public index, or with a VIAME
patch applied over them:

    imgaug  mmcv  mmdetection  mmdeploy  torchvideo  mit-yolo  rf-detr
    sam2  sam3  foundation-stereo  detectron2  litdet  dino3  sleap-nn
    darknet-to-pytorch-onnx  roi-align (mdnet)          -- 16, via
                                             `_viame_fork()`
    pyav                                     -- 1, under python-utils

`packages/pytorch-libs/fast-foundation-stereo` is a submodule the build
declares no fork for, so it is not in this list and an index would not serve
it. Worth resolving separately: either it is dead and P9-T03 removes it with
the rest, or something meant to build it and does not.

Each is a git submodule under `packages/pytorch-libs/` or
`packages/python-utils/`. `cmake/viame_python_forks.cmake` builds each into a
wheel with `pip wheel --no-build-isolation --no-deps` and installs it
`--no-deps`, gated on the `VIAME_ENABLE_PYTORCH-*` option that selects it.

An index would let a build **download** those wheels instead of compiling
them.

## What it costs not to have one

* **Build time.** mmcv compiles CUDA ops one file at a time (lite-findings
  2.15). The CI job that installs python deps turns PyTorch and ONNX *off*
  specifically because "mmcv alone would spend most of the time budget", so
  today no CI anywhere builds the forks. Nothing catches a fork that stops
  compiling except a person doing a full local build.
* **160 MB of vendored source** in 19 submodules, which P9-T03 exists to
  delete.
* **An install-time patch step**, `packaging/patches/apply.py`, down to two
  `kwplot` patches but still a step that edits site-packages after the fact.
* **Reproducibility.** A source build depends on the machine's compiler,
  CUDA and torch ABI. Two installs of the same commit can differ.

## What already exists, and what does not

Already built:

* `VIAME_PYTHON_INDEX_URL` is a cache variable, passed to pip as
  `--extra-index-url` (`cmake/viame_python_deps.cmake:115`). Pointing the
  build at an index is a one-line configure argument today.
* Locks install `--no-deps`, so pip is a fetcher and not a resolver. The
  exclusions that keep a second `cv2` and `wandb` out depend on this.
* `--extra-index-url` is already used in anger: the locks name
  `download.pytorch.org/whl/cu126`, `/cu130` and `/cpu`.
* `data.kitware.com` already distributes VIAME's release archives.

Not built, and needed:

* **The forks are not in the lock.** `forks.lock` is 358 lines and pins
  their *dependencies*; **0 of the 17 are pinned by name**. An index would
  serve wheels the lock does not mention, so P9-T01 has to add them to the
  lock as well as build them.
* **CI capacity.** The existing `VIAME_INSTALL_PYTHON_DEPS=ON` job is CPU
  only with pytorch off. Building these wheels needs a CUDA runner and a
  much larger time budget, per torch/CUDA/python combination.
* **A matrix decision.** A compiled wheel is specific to python version,
  torch version and CUDA version. The lock already has `py3.10`/`py3.12` and
  `cu126`/`cu130`/`cpu` variants: that is up to 12 combinations per fork.

## What a wheel costs to store

Not measured, and worth measuring before choosing. No wheels are built in
this tree. **Source** sizes for the six forks checked out here:

    sam2 64M   mmdetection 46M   mmdeploy 20M   mmcv 18M
    imgaug 12M   rf-detr 3.7M

mmcv and sam2 build CUDA extensions, so their wheels will be larger than
their sources; the pure-python ones smaller. A single-digit GB per matrix
combination is the right order of magnitude to plan for, not a few hundred
MB.

## The options

**A. `data.kitware.com`.** Already hosts VIAME's releases, already trusted by
the install instructions, no new account or billing. A PEP 503 index is a
directory of files and an index page, which that can serve. Least new
infrastructure; check whether it can be written to from CI and whether the
layout can be generated.

**B. GitHub releases on a `viame-wheels` repo.** Free, versioned, and CI can
publish to it directly with a token. `--extra-index-url` does not read a
releases page, so it needs either a generated PEP 503 index page in GitHub
Pages or `--find-links` instead of `--extra-index-url`. Simple, but a second
place VIAME lives.

**C. A real package index (Artifactory, Nexus, Gemfury, self-hosted
devpi).** Proper PyPI semantics, access control, retention. Most capable and
the only option with an ongoing cost and an owner.

**D. Publish to public PyPI.** Not viable as stated: these are forks of
other projects' packages, so naming them `mmcv` or `sam2` on PyPI is not
ours to do. It would require renaming each (`viame-mmcv`), which changes
every import.

## Recommendation

**A, with B as the fallback**, and P9-T01 scoped to prove it on *one* fork
before all seventeen. The plumbing already exists, so the first milestone is
small: build `imgaug` (pure python, 12M, no CUDA) into a wheel in CI, publish
it, and install a VIAME build from the index with
`-DVIAME_PYTHON_INDEX_URL=...`. That answers the questions this memo cannot
-- can CI write there, does pip read the layout, how big is a real wheel --
for a fraction of the work, and none of P9-T02/T03 has to be committed to
until it does.

**Do not start P9-T03 first.** It deletes 160 MB of vendored source across
19 submodules and is referenced by `viame_python_forks.cmake`,
`viame_python_deps.cmake` and `build_server_windows_msi.cmake`. Until the
index demonstrably serves every fork a build needs, that source is the only
way to get them.

## What is needed from the user

1. Which of A, B or C, or a host not listed here.
2. Whether CI can be given a CUDA runner, since that gates building the
   compiled forks at all.
3. Whether the python/torch/CUDA matrix should be narrowed -- one supported
   combination per release would reduce the build and storage cost roughly
   twelvefold.
