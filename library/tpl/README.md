# library/tpl

Dependencies carried in the tree rather than found on the system or built by
fletch.

**Only what is needed is carried.** Not a distribution, not a git checkout --
the files VIAME compiles or includes, and the licence. Each directory's
`CMakeLists.txt` or `README.md` says what was left behind and why, because
"why is this file not here" is the question a future reader will have.

| Package | Carried | Left behind | Why vendored |
|---|---|---|---|
| `rapidjson` | `rapidjson/`, 35 headers | cereal, which shipped it | **was inside cereal**: P8-T06 deleted the archive library and kept the JSON parser it vendored, renaming its `CEREAL_RAPIDJSON_*` macros back to `RAPIDJSON_*`. Header only |
| `cxxopts` | `cxxopts.hpp` | everything else | one header |
| `stb` | `stb_image.h`, `stb_image_write.h` | the other twenty stb headers | the codecs P7 replaced OpenCV's with |
| `libsvm` | `svm.cpp`, `svm.h`, `COPYRIGHT` | java, matlab and python bindings, the three command line tools, svm-toy, Windows projects, example data | **not stock**: carries VIAME's `HISTOGRAM` and `NMI` kernels, which no upstream or system libsvm has |
| `tinyxml` | `tinyxml.{h,cpp}`, `tinyxmlerror.cpp`, `tinyxmlparser.cpp` | `tinystr.{h,cpp}` (only used without STL strings, and VIAME always builds with them), docs, test program, sample XML, Visual Studio projects | one caller, the CVAT reader; not worth a fletch package |
| `miniz` | `miniz.c`, `miniz.h`, `LICENSE` | examples, bazel build, changelog, readme | replaces zlib: VIAME wants raw deflate, raw inflate and crc32, and builds its own ZIP container |
| `darknet` | `src/` minus the drivers, `include/`, `LICENSE` | the 17 command line drivers, the demo loop, the sample program, Windows-only files, `3rdparty/`, cfg, weights, data, scripts, Docker, python bindings | inference only; 18 MB to 1.9 MB |
| `pybind11` | 28 of 36 headers | the `eigen/` subtree, `chrono.h`, `functional.h`, `stl/filesystem.h`, `type_caster_pyobject_ptr.h`; tests, docs, CMake tooling | header only; the set is the transitive closure of what VIAME includes |

Not here, and deliberately:

* **GoogleTest** is fetched and built, not vendored and not found. Its source
  is not in the repository -- only tests link it and tests are not in a
  release -- and it is not looked up either, because
  `find_package( GTest REQUIRED )` resolved through fletch's prefix and was
  the last thing in VIAME's own build that needed fletch at all.
  `library/tpl/googletest` downloads a pinned version at configure time and
  builds it with the tree, and only when `VIAME_ENABLE_TESTS` is on. For an
  offline build, `VIAME_GTEST_SOURCE_DIR` points at an existing checkout.
* **darknet** is vendored above, **inference only**, which was the
  instruction. The 17 command line drivers built the `darknet` executable,
  and that executable is what `darknet_trainer` shelled out to -- so no
  training, no executable, no drivers, and the trainer is removed with them.
  Detection is untouched and all 24 of its golden cases reproduce exactly.

  Two files that look droppable and are not, both found by the linker rather
  than by reading: `image_opencv.cpp`, which is entirely behind
  `#ifdef OPENCV` **except** an `#else` branch defining `wait_key_cv` and
  three friends as no-ops, which unguarded training paths still call; and
  `http_stream.cpp`, which is not only the MJPEG server -- `get_time_point`,
  `this_thread_sleep_for` and the `custom_atomic_*` helpers live there,
  outside the OpenCV guard.

  `CUDNN_HALF` has to be on. darknet's own build turns it on by default and
  forces it for any architecture from 70 up, and it is not a tuning knob: it
  decides whether convolutions take cuDNN's tensor-core FP16 path. Without
  it every box was identical and every confidence moved in the fifth decimal
  -- 0.9102494 against a recorded 0.9102724 -- and all 24 goldens failed.
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
