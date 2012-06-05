/*ckwg +5
 * Copyright 2012 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_PIPELINE_VERSION_H
#define VISTK_PIPELINE_VERSION_H

#include "pipeline-config.h"

#include <boost/cstdint.hpp>

#include <string>

/**
 * \file version.h
 *
 * \brief Runtime version checks.
 */

namespace vistk
{

class VISTK_PIPELINE_EXPORT version
{
  public:
    typedef uint64_t version_t;

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

}

#endif // VISTK_PIPELINE_VERSION_H
