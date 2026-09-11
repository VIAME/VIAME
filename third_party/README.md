# third_party

Dependencies carried in the tree rather than found on the system or built by
fletch.

**Only what is needed is carried.** Not a distribution, not a git checkout --
the files VIAME compiles or includes, and the licence. Each directory's
`CMakeLists.txt` or `README.md` says what was left behind and why, because
"why is this file not here" is the question a future reader will have.

| Package | Carried | Left behind | Why vendored |
|---|---|---|---|
| `cereal` | headers | tests, docs, sandbox | serialisation, header only |
| `cxxopts` | `cxxopts.hpp` | everything else | one header |
| `kwiversys` | sources | — | was kwiver's; P5-T05 |
| `stb` | `stb_image.h`, `stb_image_write.h` | the other twenty stb headers | the codecs P7 replaced OpenCV's with |
| `libsvm` | `svm.cpp`, `svm.h`, `COPYRIGHT` | java, matlab and python bindings, the three command line tools, svm-toy, Windows projects, example data | **not stock**: carries VIAME's `HISTOGRAM` and `NMI` kernels, which no upstream or system libsvm has |
| `tinyxml` | `tinyxml.{h,cpp}`, `tinyxmlerror.cpp`, `tinyxmlparser.cpp` | `tinystr.{h,cpp}` (only used without STL strings, and VIAME always builds with them), docs, test program, sample XML, Visual Studio projects | one caller, the CVAT reader; not worth a fletch package |
| `miniz` | `miniz.c`, `miniz.h`, `LICENSE` | examples, bazel build, changelog, readme | replaces zlib: VIAME wants raw deflate, raw inflate and crc32, and builds its own ZIP container |
| `pybind11` | 28 of 36 headers | the `eigen/` subtree, `chrono.h`, `functional.h`, `stl/filesystem.h`, `type_caster_pyobject_ptr.h`; tests, docs, CMake tooling | header only; the set is the transitive closure of what VIAME includes |

Not here, and deliberately:

* **GTest** stays a found dependency. It is only ever linked by tests, which
  are not in a release, so carrying it would add source to the repository for
  something that never ships.
* **darknet** is next: `packages/darknet` is a submodule of VIAME's fork and
  moves to `third_party/darknet` built by `add_subdirectory`, per
  `lite-dependencies.md`.
* **ZLib** is replaced by `miniz` above. Measured rather than assumed, since
  the instruction was whichever is smaller: `libviame_core.so` grew 61 KB and
  `libz.so.1`, 121 KB, left VIAME's own dependency set -- a net 60 KB, and
  one fewer shared object to ship, RPATH and fix up.

  The strip is where the saving is. Compiled `-O2 -fPIC`, miniz is 105 KB of
  object code with its defaults, 94 KB without stdio and time, and **48 KB**
  once the archive APIs go -- which VIAME does not want, since it builds and
  parses the ZIP container itself. Going further to `tdefl`/`tinfl` directly
  and dropping the zlib-style API saves another 6 KB and would mean
  rewriting two working call sites; not worth it, and recorded here so the
  question does not have to be asked again.
