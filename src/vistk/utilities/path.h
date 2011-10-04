/*ckwg +5
 * Copyright 2011 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef VISTK_UTILITIES_PATH_H
#define VISTK_UTILITIES_PATH_H

#include <boost/filesystem/path.hpp>

#include <vector>

/**
 * \file path.h
 *
 * \brief Types for paths.
 */

namespace vistk
{

/// A path on the filesystem.
typedef boost::filesystem::path path_t;
/// A collection of paths on the filesystem.
typedef std::vector<path_t> paths_t;

}

#endif // VISTK_UTILITIES_PATH_H
