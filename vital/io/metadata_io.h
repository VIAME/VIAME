// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File IO functions for metadata objects.
 *
 * Currently functions are provided to use the POS metadata format from AFRL.
 * These read POS files into a metadata structure and write a
 * metadata structure to a POS file.
 */

#ifndef VITAL_METADATA_IO_H_
#define VITAL_METADATA_IO_H_

#include <vital/types/metadata.h>
#include <vital/vital_types.h>
#include <vital/vital_export.h>

namespace kwiver {
namespace vital {

/// Extract an image file basename from metadata and (if needed) frame number
/**
 * The purpose of this function is to provide a standard way to get a base file
 * name (no file extension) from metadata.  The caller of this function should
 * append a file extenstion and can then use this file to store data relevant
 * to the source frame.
 *
 * The generated base file name is determined as follows.  If a source image
 * file name is provided in \c md then the basename (file extension removed)
 * of the image is returned. Otherwise, if a source video file name is provided
 * the video file basename is appended with the frame number and returned.
 * If no source file name is provided, the base name is "frame" with the frame
 * number appended.
 */
std::string
VITAL_EXPORT
basename_from_metadata(metadata_sptr md,
                       frame_id_t frame);

std::string
VITAL_EXPORT
basename_from_metadata(kwiver::vital::metadata_vector const& mds,
                       frame_id_t frame);

/// Read in a POS file, producing a metadata object
/**
 * \throws file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \return A new camera object representing the contents of the read-in file.
 */
metadata_sptr
VITAL_EXPORT
read_pos_file( path_t const& file_path );

/// Output the given metadata object to the specified file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.  This function only writes out
 * metadata fields that are relevant to the POS file format.
 *
 * \throws file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param metadata  The \c metadata object to output.
 * \param file_path The path to output the file to.
 */
void
VITAL_EXPORT
write_pos_file( metadata const& md,
                path_t const& file_path );

} } // end namespace

#endif // VITAL_METADATA_IO_H_
