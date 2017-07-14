/*ckwg +29
 * Copyright 2012 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#if WIN32
#pragma warning (push)
#pragma warning (disable : 4244)
#endif
#include <boost/python/def.hpp>
#include <boost/python/module.hpp>
#include <boost/python/class.hpp>
#if WIN32
#pragma warning (pop)
#endif

#include <sprokit/pipeline/version.h>

#include <sprokit/version.h>


/**
 * \file version.cxx
 *
 * \brief Python bindings for version.
 */

using namespace boost::python;

class compile
{
  public:
    typedef sprokit::version::version_t version_t;

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
    .def_readonly("major", SPROKIT_VERSION_MAJOR)
    .def_readonly("minor", SPROKIT_VERSION_MINOR)
    .def_readonly("patch", SPROKIT_VERSION_PATCH)
    .def_readonly("version_string", SPROKIT_VERSION)
    .def_readonly("git_build",
#ifdef SPROKIT_BUILT_FROM_GIT
      true
#else
      false
#endif
    )
    .def_readonly("git_hash", SPROKIT_GIT_HASH)
    .def_readonly("git_hash_short", SPROKIT_GIT_HASH_SHORT)
    .def_readonly("git_dirty", SPROKIT_GIT_DIRTY)
    .def("check", &compile::check
      , (arg("major"), arg("minor"), arg("patch"))
      , "Check for a sprokit of at least the given version.")
    .staticmethod("check")
  ;

  class_<runtime>("runtime"
    , "Runtime version information."
    , no_init)
    .def_readonly("major", sprokit::version::major)
    .def_readonly("minor", sprokit::version::minor)
    .def_readonly("patch", sprokit::version::patch)
    .def_readonly("version_string", sprokit::version::version_string)
    .def_readonly("git_build", sprokit::version::git_build)
    .def_readonly("git_hash", sprokit::version::git_hash)
    .def_readonly("git_hash_short", sprokit::version::git_hash_short)
    .def_readonly("git_dirty", sprokit::version::git_dirty)
    .def("check", &sprokit::version::check
      , (arg("major"), arg("minor"), arg("patch"))
      , "Check for a sprokit of at least the given version.")
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
  return SPROKIT_VERSION_CHECK(major_, minor_, patch_);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
