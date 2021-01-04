// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "version.h"

#include <vital/version.h>

/**
 * \file version.cxx
 *
 * \brief Runtime version checks.
 */

namespace sprokit
{

version::version_t const version::major = KWIVER_VERSION_MAJOR;
version::version_t const version::minor = KWIVER_VERSION_MINOR;
version::version_t const version::patch = KWIVER_VERSION_PATCH;
std::string const version::version_string = KWIVER_VERSION;

bool const version::git_build =
#ifdef KWIVER_BUILT_FROM_GIT
  true
#else
  false
#endif
  ;
std::string const version::git_hash = KWIVER_GIT_HASH;
std::string const version::git_hash_short = KWIVER_GIT_HASH_SHORT;
std::string const version::git_dirty = KWIVER_GIT_DIRTY;

// If any of the version components are 0, we get compare warnings. Turn
// them off here.
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wtype-limits"
#endif

bool
version
::check(version_t major_, version_t minor_, version_t patch_)
{
  return KWIVER_VERSION_CHECK(major_, minor_, patch_);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}
