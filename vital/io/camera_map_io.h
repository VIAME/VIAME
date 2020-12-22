// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header file for loading camera maps
 */

#ifndef VITAL_CAMERA_MAP_IO_H_
#define VITAL_CAMERA_MAP_IO_H_

#include <vital/types/camera_map.h>

#include <vital/vital_types.h>

#include <vector>

namespace  kwiver {
namespace vital {

/// Load a camera map from krtd files stored in a directory.
/**
 * This function assumes that krtd files stored in the directory have the
 * same names as those in an image file list, only with a .krtd extension
 * instead of an image extension.
 *
 * \throws invalid_data
 *   Unable to find any camera krtd files in the given directory
 * \throw path_not_exists
 *   The specified directory does not exist
 *
 * \param img_files a list of image files
 * \param dir directory path containing krtd files for the given images
 * \return a new camera map created after parsing all krtd files
 */
camera_map_sptr
VITAL_EXPORT read_krtd_files( std::vector< path_t > const& img_files, path_t const& dir );

} } // end namespace vital

#endif // VITAL_CAMERA_MAP_IO_H_
