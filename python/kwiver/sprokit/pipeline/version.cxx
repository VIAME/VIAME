// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <pybind11/pybind11.h>

#include <sprokit/pipeline/version.h>

#include <vital/version.h>

/**
 * \file version.cxx
 *
 * \brief Python bindings for version.
 */

using namespace pybind11;
namespace kwiver{
namespace sprokit{
namespace python{
class compile
{
  public:
    typedef  ::sprokit::version::version_t version_t;

    static version_t const major;
    static version_t const minor;
    static version_t const patch;
    static std::string const version_string;
    static bool const git_build;
    static std::string const git_hash;
    static std::string const git_hash_short;
    static std::string const git_dirty;

    static bool check(version_t major_, version_t minor_, version_t patch_);
};

 ::sprokit::version::version_t const compile::major = KWIVER_VERSION_MAJOR;
 ::sprokit::version::version_t const compile::minor = KWIVER_VERSION_MINOR;
 ::sprokit::version::version_t const compile::patch = KWIVER_VERSION_PATCH;
std::string const compile::version_string = KWIVER_VERSION;
bool const compile::git_build =
#ifdef KWIVER_BUILT_FROM_GIT
      true;
#else
      false;
#endif
std::string const compile::git_hash = KWIVER_GIT_HASH;
std::string const compile::git_hash_short = KWIVER_GIT_HASH_SHORT;
std::string const compile::git_dirty = KWIVER_GIT_DIRTY;

class runtime
{
};
}
}
}
using namespace kwiver::sprokit::python;

PYBIND11_MODULE(version, m)
{
  class_<compile>(m,"compile"
    , "Compile-time version information.")
    .def_readonly_static("major", &compile::major)
    .def_readonly_static("minor", &compile::minor)
    .def_readonly_static("patch", &compile::patch)
    .def_readonly_static("version_string", &compile::version_string)
    .def_readonly_static("git_build", &compile::git_build)
    .def_readonly_static("git_hash", &compile::git_hash)
    .def_readonly_static("git_hash_short", &compile::git_hash_short)
    .def_readonly_static("git_dirty", &compile::git_dirty)
    .def_static("check", &compile::check
      , arg("major"), arg("minor"), arg("patch")
      , "Check for a sprokit of at least the given version.")
  ;

  class_<runtime>(m, "runtime"
    , "Runtime version information.")
    .def_readonly_static("major", &sprokit::version::major)
    .def_readonly_static("minor", &sprokit::version::minor)
    .def_readonly_static("patch", &sprokit::version::patch)
    .def_readonly_static("version_string", &sprokit::version::version_string)
    .def_readonly_static("git_build", &sprokit::version::git_build)
    .def_readonly_static("git_hash", &sprokit::version::git_hash)
    .def_readonly_static("git_hash_short", &sprokit::version::git_hash_short)
    .def_readonly_static("git_dirty", &sprokit::version::git_dirty)
    .def_static("check", &sprokit::version::check
      , arg("major"), arg("minor"), arg("patch")
      , "Check for a sprokit of at least the given version.")
  ;
}

// If any of the version components are 0, we get compare warnings. Turn
// them off here.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif
namespace kwiver{
namespace sprokit{
namespace python{
bool
compile
::check(version_t major_, version_t minor_, version_t patch_)
{
  return KWIVER_VERSION_CHECK(major_, minor_, patch_);
}
}
}
}
#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
