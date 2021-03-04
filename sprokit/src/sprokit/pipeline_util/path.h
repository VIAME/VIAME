// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef SPROKIT_PIPELINE_UTIL_PATH_H
#define SPROKIT_PIPELINE_UTIL_PATH_H

#ifndef BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_NO_DEPRECATED
#endif

// XXX(boost): 1.50.0
#ifndef BOOST_FILESYSTEM_VERSION
#define BOOST_FILESYSTEM_VERSION 3
#else
#if BOOST_FILESYSTEM_VERSION == 2
#error "Only boost::filesystem version 3 is supported."
#endif
#endif

#include <boost/filesystem/path.hpp>

#include <vector>

/**
 * \file path.h
 *
 * \brief Types for paths.
 */

//+ this file will be deleted during the deboostification effort

namespace sprokit
{

/// A path on the filesystem.
typedef boost::filesystem::path path_t;
/// A collection of paths on the filesystem.
typedef std::vector<path_t> paths_t;

}

#endif // SPROKIT_PIPELINE_UTIL_PATH_H
