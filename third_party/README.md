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
| `pybind11` | 28 of 36 headers | the `eigen/` subtree, `chrono.h`, `functional.h`, `stl/filesystem.h`, `type_caster_pyobject_ptr.h`; tests, docs, CMake tooling | header only; the set is the transitive closure of what VIAME includes |

Not here, and deliberately:

* **GTest** stays a found dependency. It is only ever linked by tests, which
  are not in a release, so carrying it would add source to the repository for
  something that never ships.
* **darknet** is next: `packages/darknet` is a submodule of VIAME's fork and
  moves to `third_party/darknet` built by `add_subdirectory`, per
  `lite-dependencies.md`.
* **ZLib** is still open. One caller -- `camera_rig_io` reading NPZ -- and the
  choice is a vendored `miniz` (one file) or moving the NPZ reading to python.
