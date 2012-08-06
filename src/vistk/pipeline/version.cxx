/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "version.h"

#include <vistk/version.h>

/**
 * \file version.cxx
 *
 * \brief Runtime version checks.
 */

namespace vistk
{

version::version_t const version::major = VISTK_VERSION_MAJOR;
version::version_t const version::minor = VISTK_VERSION_MINOR;
version::version_t const version::patch = VISTK_VERSION_PATCH;
std::string const version::version_string = VISTK_VERSION;

bool const version::git_build =
#ifdef VISTK_BUILT_FROM_GIT
  true
#else
  false
#endif
  ;
std::string const version::git_hash = VISTK_GIT_HASH;
std::string const version::git_hash_short = VISTK_GIT_HASH_SHORT;
std::string const version::git_dirty = VISTK_GIT_DIRTY;

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
  return VISTK_VERSION_CHECK(major_, minor_, patch_);
}

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
}
