// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File IO functions for a \ref kwiver::vital::landmark_map
 *
 * Uses the PLY file format
 */

#ifndef VITAL_LANDMARK_MAP_IO_H_
#define VITAL_LANDMARK_MAP_IO_H_

#include <vital/vital_export.h>

#include <vital/types/landmark_map.h>

namespace kwiver {
namespace vital {

/// Output the given \c landmark_map object to the specified PLY file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.
 *
 * \throws file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param landmarks The \c landmark_map object to output.
 * \param file_path The path to output the file to.
 */
void
VITAL_EXPORT write_ply_file( landmark_map_sptr const& landmarks,
                                 path_t const&            file_path );

/// Load a given \c landmark_map object from the specified PLY file path
/**
 * This function does not read all ply files, only ply files which have
 * been output by the landmark write ply function.
 *
 * \throws file_read_exception
 *    Thrown when something prevents input of the file.
 *
 * \param file_path The path to output the file to.
 */
landmark_map_sptr
VITAL_EXPORT read_ply_file( path_t const& file_path );

} } // end namespace

#endif // VITAL_LANDMARK_MAP_IO_H_
