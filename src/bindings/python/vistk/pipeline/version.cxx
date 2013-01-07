/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include <vistk/pipeline/version.h>

#include <vistk/version.h>

#include <boost/python/class.hpp>
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>

/**
 * \file version.cxx
 *
 * \brief Python bindings for version.
 */

using namespace boost::python;

class compile
{
  public:
    typedef vistk::version::version_t version_t;

    static bool check(version_t major_, version_t minor_, version_t patch_);
};

class runtime
{
};

BOOST_PYTHON_MODULE(version)
{
  class_<compile>("compile"
    , "Compile-time version information."
    , no_init)
    .def_readonly("major", VISTK_VERSION_MAJOR)
    .def_readonly("minor", VISTK_VERSION_MINOR)
    .def_readonly("patch", VISTK_VERSION_PATCH)
    .def_readonly("version_string", VISTK_VERSION)
    .def_readonly("git_build",
#ifdef VISTK_BUILT_FROM_GIT
      true
#else
      false
#endif
    )
    .def_readonly("git_hash", VISTK_GIT_HASH)
    .def_readonly("git_hash_short", VISTK_GIT_HASH_SHORT)
    .def_readonly("git_dirty", VISTK_GIT_DIRTY)
    .def("check", &compile::check
      , (arg("major"), arg("minor"), arg("patch"))
      , "Check for a vistk of at least the given version.")
    .staticmethod("check")
  ;

  class_<runtime>("runtime"
    , "Runtime version information."
    , no_init)
    .def_readonly("major", vistk::version::major)
    .def_readonly("minor", vistk::version::minor)
    .def_readonly("patch", vistk::version::patch)
    .def_readonly("version_string", vistk::version::version_string)
    .def_readonly("git_build", vistk::version::git_build)
    .def_readonly("git_hash", vistk::version::git_hash)
    .def_readonly("git_hash_short", vistk::version::git_hash_short)
    .def_readonly("git_dirty", vistk::version::git_dirty)
    .def("check", &vistk::version::check
      , (arg("major"), arg("minor"), arg("patch"))
      , "Check for a vistk of at least the given version.")
    .staticmethod("check")
  ;
}

// If any of the version components are 0, we get compare warnings. Turn
// them off here.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif

bool
compile
::check(version_t major_, version_t minor_, version_t patch_)
{
  return VISTK_VERSION_CHECK(major_, minor_, patch_);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
