// This file is part of VIAME, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.

/// \file
/// \brief File IO functions for a camera (KRTD format)

#ifndef VIAME_CAMERA_IO_H_
#define VIAME_CAMERA_IO_H_

#include <plugins/core/viame_core_export.h>

#include <vital/vital_types.h>
#include <vital/types/camera_perspective.h>

namespace viame {

/// Read in a KRTD file, producing a camera object
///
/// \throws file_not_found_exception
///    Thrown when the file could not be found on the file system.
/// \throws file_not_read_exception
///    Thrown when the file could not be read or parsed for whatever reason.
///
/// \param file_path   The path to the file to read in.
/// \return A new camera object representing the contents of the read-in file.
kwiver::vital::camera_perspective_sptr
VIAME_CORE_EXPORT read_krtd_file( kwiver::vital::path_t const& file_path );

/// Read in a KRTD file, producing a camera object
///
/// \throws file_not_found_exception
///    Thrown when the file could not be found on the file system.
/// \throws file_not_read_exception
///    Thrown when the file could not be read or parsed for whatever reason.
///
/// \param image_file
///    The path to an image file associated with the camera.
/// \param camera_dir
///    The directory path containing the KRTD file for the given image.
/// \return
///    A new camera object representing the contents of the read-in file.
kwiver::vital::camera_perspective_sptr
VIAME_CORE_EXPORT read_krtd_file( kwiver::vital::path_t const& image_file,
                                   kwiver::vital::path_t const& camera_dir );

/// Output the given \c camera object to the specified file path
///
/// If a file exists at the target location, it will be overwritten. If the
/// containing directory of the given path does not exist, it will be created
/// before the file is opened for writing.
///
/// \throws file_write_exception
///    Thrown when something prevents output of the file.
///
/// \param cam       The \c camera object to output.
/// \param file_path The path to output the file to.
void
VIAME_CORE_EXPORT write_krtd_file( kwiver::vital::camera_perspective const& cam,
                                    kwiver::vital::path_t const& file_path );

} // namespace viame

#endif // VIAME_CAMERA_IO_H_
