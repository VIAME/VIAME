# pybind11, vendored

Headers only, and only the ones VIAME reaches.

pybind11 is header-only, so what is carried is `include/pybind11` -- but not
all of it. The set is the transitive closure of the eleven headers VIAME and
its bindings actually include (`pybind11.h`, `stl.h`, `embed.h`, `numpy.h`,
`eval.h`, `iostream.h`, `operators.h`, `stl_bind.h`, `attr.h`, `complex.h`,
`options.h`), plus every `detail/` header: two of those are reached only
through a conditional include, and a missing one is a build break rather
than a saving.

What that leaves behind is the whole `eigen/` subtree -- three headers for a
library phase 6 removed from VIAME entirely -- along with `chrono.h`,
`functional.h`, `stl/filesystem.h` and `type_caster_pyobject_ptr.h`, none of
which anything here includes. The distribution's tests, documentation and
CMake tooling are not carried at all.

28 headers. Upstream ships 36 in `include/` and a repository many times that.

Licence: BSD 3-Clause, `LICENSE` beside this file.
