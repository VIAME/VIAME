/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief File IO functions for the POS metadata file format from AFRL
 *
 * These read POS files into a video_metadata structure and write a
 * video_metadata structrue to a POS file.
 */

#ifndef VITAL_POS_METADATA_IO_H_
#define VITAL_POS_METADATA_IO_H_

#include <vital/vital_types.h>
#include <vital/video_metadata/video_metadata.h>
#include <vital/video_metadata/vital_video_metadata_export.h>

namespace kwiver {
namespace vital {

/// Read in a POS file, producing a video_metadata object
/**
 * \throws file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \return A new camera object representing the contents of the read-in file.
 */
video_metadata_sptr
VITAL_VIDEO_METADATA_EXPORT read_pos_file( path_t const& file_path );


/// Output the given video_metadata object to the specified file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.  This function only writes out
 * metadata fields that are relevant to the POS file format.
 *
 * \throws file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param metadata  The \c video_metadata object to output.
 * \param file_path The path to output the file to.
 */
void
VITAL_VIDEO_METADATA_EXPORT write_pos_file( video_metadata const& md,
                                            path_t const& file_path );

} } // end namespace

#endif // VITAL_POS_METADATA_IO_H_
