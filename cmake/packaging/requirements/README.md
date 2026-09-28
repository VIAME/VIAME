# VIAME's python dependencies

`*.in` says what VIAME needs and why. `py3.X/*.lock` is the pinned
resolution `pip-compile` produced from it with python 3.X, committed beside
it. The build installs the locks for the python it was configured with,
`--no-deps`, so pip fetches and does not resolve; with no directory for that
python it stops at configure.

A lock belongs to the python that compiled it. pip-compile keeps only the
requirements whose markers hold for its interpreter, so `base.in`'s
`python_version >= "3.12"` lines -- numpy, numba, scikit-image, matplotlib,
pandas, kwcoco and more -- are not in a 3.10 lock at all, and installing a
3.10 lock into a 3.12 python leaves them out without an error.

| file | what it is |
|---|---|
| `base.in` | everything a default build needs |
| `cuda12.in`, `cuda13.in`, `cpu.in` | `base` plus torch and the ONNX runtime for that accelerator, from the index that carries them |
| `forks.in` | what the source-built packages in `packages/pytorch-libs` need, which nothing else declares |
| `learn.in`, `sleap.in`, `colmap.in`, `test.in` | what `VIAME_ENABLE_PYTORCH-LEARN`, `-SLEAP`, `-COLMAP` and `-TESTS` add |

An accelerator file pulls `base.in` in and compiles to a complete lock. The
others are *additive*: they list only what the option adds, and are compiled
with the accelerator's lock as a constraint so every version they share with
a default install is the same version. `cmake/viame_python_deps.cmake`
passes one `-r` per selected lock to a single `pip install`.

## Changing a dependency

Edit the `.in`, recompile the `.lock`s it feeds, commit both:

Once for each python there is a directory for, run with that python:

```sh
python -m pip install pip-tools
cd cmake/packaging/requirements
PY=py$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')

EXCLUDE="--unsafe-package=triton --unsafe-package=wandb \
         --unsafe-package=decord"

for variant in cuda12 cuda13 cpu; do
    python -m piptools compile --strip-extras $EXCLUDE -o $PY/$variant.lock $variant.in
done

for extra in forks learn sleap colmap test; do
    python -m piptools compile --strip-extras $EXCLUDE -c $PY/cuda12.lock -o $PY/$extra.lock $extra.in
done
```

A lock is per (python, accelerator). `py3.10` was compiled on the reference
machine, python 3.10 with CUDA 12.6; `py3.12` with python-build-standalone's
3.12.14, the python `VIAME_PYTHON_STANDALONE` downloads and the version the
Docker images' Ubuntu 24.04 has. Phase 9 moves the compilation into CI so
that every set is produced the same way.

## What is deliberately excluded

`--unsafe-package` keeps a package out of the lock even when something in
the graph asks for it. Three are excluded, and each one is a decision:

| package | why |
|---|---|
| `triton` | 697 MB. sam3's `edt.py` is the only user: it takes a Triton kernel for its euclidean distance transform when one is present and an OpenCV CPU implementation when it is not, which is the path Windows already takes |
| `wandb` | 86 MB of telemetry client that the trainers use only if configured, and nothing configures it. `mit-yolo` requires it |
| `decord` | 26 MB of video reader. Nothing in VIAME, and nothing in any submodule this build compiles, imports it -- VIAME feeds frames itself |

## OpenCV

VIAME declares none. `library/image_kernels` and `library/utilities` are the
implementation, and no module VIAME owns imports cv2 -- `baseline:lazy_cv2`
is the test that keeps it that way.

It still arrives. `ultralytics`, `albumentations`, `albucore`, `mmengine`,
`pylabel` and `bbox_visualizer` all require it unconditionally -- three of
them cannot be imported without it -- and `kwimage` is declared here as
`kwimage[headless]` because its warp, resize and mask paths are cv2-backed
with no fallback and it declares that only in the extra. Naming the extra
puts the dependency on the package that has it.

With no distribution named here the resolution carries *both* of them at the
same version: `opencv-python` for what asks for the full build,
`opencv-python-headless` for what asks for headless. They write the same
`cv2/` directory and whichever pip installs last wins, so the runtime image
installs `libgl1` (`docker/Dockerfile`) -- the full build needs `libGL.so.1`
and the headless one does not -- and works either way. Nothing here uses
`highgui`, and nothing uses a `contrib` module outside a test that skips when
it is absent.
