// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_UTIL_PATH_H
#define SPROKIT_PIPELINE_UTIL_PATH_H

#include <filesystem>
#include <vector>

/**
 * \file path.h
 *
 * \brief Types for paths.
 */

namespace sprokit
{

/// A path on the filesystem.
typedef std::filesystem::path path_t;
/// A collection of paths on the filesystem.
typedef std::vector<path_t> paths_t;

}

#endif // SPROKIT_PIPELINE_UTIL_PATH_H
