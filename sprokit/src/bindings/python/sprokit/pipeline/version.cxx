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

#include <pybind11/pybind11.h>

#include <sprokit/pipeline/version.h>

#include <vital/version.h>


/**
 * \file version.cxx
 *
 * \brief Python bindings for version.
 */

using namespace pybind11;

class compile
{
  public:
    typedef sprokit::version::version_t version_t;

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

sprokit::version::version_t const compile::major = KWIVER_VERSION_MAJOR;
sprokit::version::version_t const compile::minor = KWIVER_VERSION_MINOR;
sprokit::version::version_t const compile::patch = KWIVER_VERSION_PATCH;
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

bool
compile
::check(version_t major_, version_t minor_, version_t patch_)
{
  return KWIVER_VERSION_CHECK(major_, minor_, patch_);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
