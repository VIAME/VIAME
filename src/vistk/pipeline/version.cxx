/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "version.h"

#include <sprokit/version.h>

/**
 * \file version.cxx
 *
 * \brief Runtime version checks.
 */

namespace sprokit
{

version::version_t const version::major = SPROKIT_VERSION_MAJOR;
version::version_t const version::minor = SPROKIT_VERSION_MINOR;
version::version_t const version::patch = SPROKIT_VERSION_PATCH;
std::string const version::version_string = SPROKIT_VERSION;

bool const version::git_build =
#ifdef SPROKIT_BUILT_FROM_GIT
  true
#else
  false
#endif
  ;
std::string const version::git_hash = SPROKIT_GIT_HASH;
std::string const version::git_hash_short = SPROKIT_GIT_HASH_SHORT;
std::string const version::git_dirty = SPROKIT_GIT_DIRTY;

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
  return SPROKIT_VERSION_CHECK(major_, minor_, patch_);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}
