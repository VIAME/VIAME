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
