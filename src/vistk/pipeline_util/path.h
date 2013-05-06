/*ckwg +5
 * Copyright 2011-2013 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef SPROKIT_PIPELINE_UTIL_PATH_H
#define SPROKIT_PIPELINE_UTIL_PATH_H

#ifndef BOOST_FILESYSTEM_NO_DEPRECATED
#define BOOST_FILESYSTEM_NO_DEPRECATED
#endif

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

namespace sprokit
{

/// A path on the filesystem.
typedef boost::filesystem::path path_t;
/// A collection of paths on the filesystem.
typedef std::vector<path_t> paths_t;

}

#endif // SPROKIT_PIPELINE_UTIL_PATH_H
