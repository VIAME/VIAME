// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File IO functions for a \ref kwiver::vital::camera
 *
 * File format is the KRTD file.
 */

#ifndef VITAL_CAMERA_IO_H_
#define VITAL_CAMERA_IO_H_

#include <vital/vital_config.h>
#include <vital/vital_export.h>
#include <vital/vital_types.h>

#include <vital/types/camera_perspective.h>

namespace kwiver {
namespace vital {

/// Read in a KRTD file, producing a camera object
/**
 * \throws file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \return A new camera object representing the contents of the read-in file.
 */
camera_perspective_sptr
VITAL_EXPORT read_krtd_file( path_t const& file_path );

/// Read in a KRTD file, producing a camera object
/**
 * \throws file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param image_file
 *    The path to an image file associated with the camera.
 * \param camera_dir
 *    The directory path containing the KRTD file for the given image.
 * \return
 *    A new camera object representing the contents of the read-in file.
 */
camera_perspective_sptr
VITAL_EXPORT read_krtd_file( path_t const& image_file,
                             path_t const& camera_dir );

/// Output the given \c camera object to the specified file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.
 *
 * \throws file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param cam       The \c camera object to output.
 * \param file_path The path to output the file to.
 */
void
VITAL_EXPORT write_krtd_file( camera_perspective const& cam,
                              path_t const& file_path );

} } // end namespace

#endif // VITAL_CAMERA_IO_H_
