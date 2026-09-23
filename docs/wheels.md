# Building a VIAME wheel

    make install
    make wheel

The `.whl` lands in `<build>/wheel`. The target is not part of `all`: a wheel
is packed from an install prefix, and a `make wheel` that reinstalled first
would write over a prefix somebody may be running from.

## What decides the contents

Two files, and the split between them is the point:

| | |
|---|---|
| `cmake/wheel/build_wheel.py` | shared between branches; knows nothing about any particular layout |
| `cmake/wheel/contents.txt` | this branch's layout, as data |

`main` installs a `kwiver` package beside `viame` and 65 shared libraries;
the `lite` branch installs one `viame` package and a single `libviame.so`.
Both are described by their own `contents.txt`, so merging one branch into
the other is a conflict in a list of paths, if anywhere, rather than in a
builder that has grown two modes.

The contents syntax:

    include <glob relative to the install prefix> -> <destination in the wheel>
    exclude <glob relative to the install prefix>

`**` crosses directory separators and `*` does not, both ends are anchored,
and whatever the last `**` matched becomes the path under a destination that
ends in `/`. Excludes run after includes, so a package can be taken whole and
then thinned. `{data}` expands to `<name>-<version>.data/data`.

## Why nothing is patched

VIAME's extension modules are built with a `$ORIGIN`-relative RUNPATH:

    $ORIGIN/../../../../../lib

From `site-packages/<pkg>/<sub>/x.so` that resolves to the environment's
`lib/`, and a wheel can place a file exactly there through the `.data/data/`
scheme, which pip unpacks into the environment prefix. So the shared
libraries go to `{data}/lib/` and the RUNPATH that worked in the install
prefix keeps working once installed. There is no `patchelf` or `auditwheel`
step, and neither tool needs to be present to build a wheel.

**Where this does not hold.** The RUNPATH assumes site-packages is exactly
five levels below the prefix, which is `<prefix>/lib/python3.X/site-packages`.
A `lib64` layout, a `local/lib` layout, a `--target` install or a zipped egg
puts the modules somewhere else and the libraries will not be found. Such a
layout needs the RPATH rewritten, which is what `auditwheel repair` is for,
and it is the thing to reach for when the wheel has to be a `manylinux` one
for an index rather than an artifact built and installed on a known machine.

The wheel is tagged for the interpreter the extensions were built against —
`cp310-cp310-linux_x86_64`, not `manylinux`, and deliberately so. It is not
portable to another python minor version, and it does not claim to be.

## Verifying one

`cmake/wheel/test_build_wheel.py` covers the glob and packing logic and needs
no build:

    python3 cmake/wheel/test_build_wheel.py

That the wheel *imports* is a separate check, and worth doing on a real one:
install it into a virtual environment with no VIAME environment sourced, and
import something that loads native code.

    python3 -c "from kwiver.vital.types import BoundingBoxD; print(BoundingBoxD(1,2,3,4).min_x())"

If the native libraries were not found this fails with a missing shared
object; if a dependency such as `numpy` is absent it fails with
`initialization failed` wrapping a `ModuleNotFoundError`, which is a
different problem and means the packing worked.

## What the wheel needs at runtime

Established by installing it into an empty virtual environment with `pip` and
running the GFIT pipelines until they produced detections, rather than by
reading imports. The dependency list below is what that took, in the order it
was discovered.

**To import `viame` at all:** `numpy`, which the wheel declares.

**To run the GFIT detector and classifier:** `torch`, `torchvision`,
`scriptconfig`, `ubelt`, `pillow`, `opencv-python-headless`, then the Kitware
stack `kwimage`, `kwcoco`, `kwarray`, `ndsampler`, and then `astunparse`,
`pygtrie`, `networkx_algo_common_subtree`, `torch_liberator` and `liberator`,
which the vendored netharn reaches through several layers of lazy import.

None of these are in the wheel, and none need to be: they are all on PyPI.

**`rfdetr` is the exception, and it is the interesting one.** `pip install
rfdetr` gets 1.10.1 from PyPI, and the pipeline fails on it:

    1 validation error for RFDETRLargeConfig
    resolution: Input should be a valid integer
                [input_value=(960, 1728), input_type=tuple]

VIAME does not use that package. It builds a fork, `rfdetr 1.8.0.dev0`, from
`packages/pytorch-libs/rf-detr`, and that fork takes a `(height, width)`
tuple. This is what open decision 9 and P9-T01 are about: seventeen forks with
no wheel on any public index. Until that index exists a wheel cannot declare
`rfdetr` as a requirement, because the name resolves to something else.

**CUDA does not come from the wheel, and does not currently come from torch
either.** The extension modules link `libcudart`, `libcudnn`, `libcublas`,
`libcublasLt` and `libcurand`, and in a fresh environment they resolve
against `/usr/local/cuda` -- a system installation the wheel neither declares
nor ships. On a machine without one, `import viame.types` fails. Making them
come from the `nvidia-*` wheels that torch already pulls in is the fix, and it
is not done: it needs those directories on the modules' RUNPATH, or a
preload, since the `nvidia-*` wheels put their libraries under
`site-packages/nvidia/*/lib` rather than anywhere the loader looks by default.

## What a code-only wheel cannot do on its own

`configs/` is not in the wheel, so pipelines and models come from elsewhere.
Point `VIAME_INSTALL` at an install that has them:

    VIAME_INSTALL=/path/to/install \
      kwiver runner /path/to/configs/pipelines/detector_gfit_groups_v3.pipe \
        -s input:video_filename=input_list.txt

That is the arrangement the code-only decision implies, and it works: the GFIT
groups pipeline run this way, with the code from a pip-installed wheel and the
configs and models from an install, produced **detections identical to the
same pipeline run entirely from that install**.

## CUDA comes from the pip wheels

The extension modules, `libviame` and the `viame`/`kwiver` tools all link
`libcudart`, `libcudnn`, `libcublas`, `libcublasLt` and `libcurand`. Built
against a system CUDA they carry no path for them, so a fresh environment
found whatever `/usr/local/cuda` happened to hold -- and on a machine with no
system CUDA, `import viame.types` failed outright.

torch already installs those libraries as wheels, under
`site-packages/nvidia/<component>/lib`, and that is the copy to use: the same
one torch itself loaded, rather than a second system copy of a possibly
different version in one process.

Nothing on those paths is discoverable by the loader by default, so
`build_wheel.py` appends `$ORIGIN`-relative RUNPATH entries pointing at them.
The relative path differs per file, which is why this is done when packing
rather than at link time -- only here is the destination known:

| lands at | entry |
|---|---|
| `site-packages/viame/types/_types.so` | `$ORIGIN/../../nvidia/<c>/lib` |
| `site-packages/viame/pipeline/util/load.so` | `$ORIGIN/../../../nvidia/<c>/lib` |
| `<env>/lib/libviame.so.1` | `$ORIGIN/python3.X/site-packages/nvidia/<c>/lib` |
| `<env>/bin/viame` | `$ORIGIN/../lib/python3.X/site-packages/nvidia/<c>/lib` |

The `python3.X` is taken from the wheel's own python tag, so it agrees with
the interpreter the modules were built against by construction.

The existing entries are kept rather than replaced -- `$ORIGIN/../../../../..`
`/lib` is what finds `libviame` itself -- and the new ones are appended, so a
component whose wheel is *not* installed falls through to the system copy
instead of failing. Measured on an environment with only
`nvidia-cuda-runtime-cu12` and `nvidia-curand-cu12` present: `libcudart` and
`libcurand` resolve inside the environment, `libcublas` and `libcudnn` fall
back to `/usr/local/cuda`.

Needs `patchelf`, which is on PyPI (`pip install patchelf`). Without it the
wheel still builds and says which binaries were left alone, because a wheel
that cannot be produced is worse than one that needs a system CUDA.

**Not declared as a requirement.** The wheel does not list `torch` or the
`nvidia-*` wheels in `Requires-Dist`. Declaring `torch` would impose a
multi-gigabyte dependency on someone who only wants `import viame.types`, and
declaring the `nvidia-*` wheels directly would pin a CUDA major version
against torch's own choice of one. The RUNPATH is the mechanism; which CUDA
arrives is torch's decision, which is the point.

## Default configs, and what is deliberately absent

The wheel ships the pipelines that need no model -- 99 files, 266 KB, chosen
by `select_default_configs.py` rather than listed by hand. A pipeline whose
`relativepath weight = models/x.pth` pointed at nothing would be worse than
absent: it looks installed and fails at configure time.

**Nothing from `configs/add-ons/` is ever shipped.** Add-on packs are model
distributions -- `DEFAULT-FISH`, `GFIT`, `SAM3` and sixteen others, 8.6 GB of
weights on this install -- and they are fetched at runtime. The selector
refuses add-on paths explicitly rather than merely not looking at them.

So a pipeline that *does* need a model still runs, with its models supplied
the way add-ons always supply them:

    VIAME_INSTALL=/path/with/models \
      kwiver runner /path/to/detector_gfit_groups_v3.pipe ...

## Declared dependencies

`cmake/wheel/requirements.txt`, which the `wheel` target passes as
`--requires-from`. Kept as a file so the reasoning for each entry lives
beside it.

The list was established by **running** the GFIT and DEFAULT-FISH pipelines
in an empty environment, not by scanning imports. Five entries --
`astunparse`, `pygtrie`, `networkx_algo_common_subtree`, `torch_liberator`,
`liberator` -- appear nowhere in VIAME's sources and were found only by a
pipeline failing on them, reached through vendored netharn's lazy imports.

Vendoring moves a fork's requirements onto us: `viame.rfdetr` and
`viame.sam2` are inside this package, so `transformers`, `pydantic`,
`hydra-core`, `iopath` and the rest are ours now. Where floors overlap the
highest wins -- torch is `>=2.3.1`, sam2's, above rfdetr's `>=2.2.0`.

### The CUDA major is ours to declare, and the default variant has nothing to declare

Which CUDA a binary needs is a property of how it was compiled, not of what
torch later chooses. So the question is only ever whether the build agrees
with torch.

**The default wheel (cu13) declares no CUDA at all.** A plain
`pip install torch` brings the whole runtime -- `nvidia-cuda-runtime`,
`nvidia-cublas`, `nvidia-curand` and `nvidia-cudnn-cu13`, via
`cuda-toolkit` -- and those are the copies this wheel's RUNPATH points at.
Verified on a clean install: every CUDA library `bin/viame` loads comes from
`site-packages/nvidia/`, none from `/usr/local/cuda`. `torch>=2.11` is the
floor that makes it true, since torch moved to cu13 at 2.11.

**The `+cu12` variant declares all of them**, plus `torch<2.11`. It has to:
current torch is cu13, so this build's `libcudart.so.12` would find nothing
of torch's and fall back to a system CUDA 12 -- which is the dependency the
RUNPATH work existed to remove. An earlier clean install did exactly that,
silently. The bound and the declarations are what make the mismatch safe.

Nothing CUDA is ever *bundled* in either variant. The wheel ships RUNPATHs
and, for cu12, dependency names.

## The platform tag

PyPI refuses `linux_x86_64`; a wheel has to say which glibc it needs, as a
manylinux tag (PEP 600). `build_wheel.py` **computes** the tag from the
binaries' glibc symbol versions rather than taking it on trust, because a
declared tag that is wrong installs cleanly and then fails at import on the
machine it was wrong about. This build comes out
`manylinux_2_35_x86_64`, which is what `auditwheel show` independently says.

`auditwheel repair` is deliberately not used. It assumes the native
libraries live inside the importable package, and this wheel's do not:
`libviame.so.1` is in `.data/data/lib/` and the tools are in
`.data/scripts/`. Pointed at that layout it relocates the library into the
package and rewrites the RUNPATHs, undoing the `$ORIGIN` paths that make
CUDA resolve from the nvidia wheels. The two things it does that we want --
compute the glibc floor, and check nothing outside the policy is relied on
-- are done directly instead.

`--bundle` packs a system library the manylinux allowlist does not cover,
and only if something in the wheel actually needs it. `libgomp.so.1` is the
one: it goes beside `libviame.so.1` in `{data}/lib/`, whose RUNPATH already
has `$ORIGIN`, so nothing is patched. A build without OpenMP does not carry
it.

### Uploading

The default wheel's name and tag are both PyPI-legal:
`viame-<version>-cp310-cp310-manylinux_2_35_x86_64.whl`. The `+cu12`
variant carries a PEP 440 local version, which **PyPI refuses** -- that
variant needs its own index.

The version comes from the first line of `RELEASE_NOTES.md`. A release
number can never be reused on PyPI, so uploading is a deliberate,
irreversible act and is not part of `make wheel`.
