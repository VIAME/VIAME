// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File IO functions for a \ref kwiver::vital::track_set
 *
 * \todo Describe format here.
 */

#ifndef VITAL_TRACK_SET_IO_H_
#define VITAL_TRACK_SET_IO_H_

#include <vital/vital_export.h>

#include <vital/types/feature_track_set.h>

namespace kwiver {
namespace vital {

/// Read in a track file, producing a track_set
/**
 * \note The simple track file format does not represent all data within
 *       a track.  This version only reads the track topology.
 * \throws file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \return A \c track_set object representing the contents of the read-in file.
 */
track_set_sptr
VITAL_EXPORT read_track_file( path_t const& file_path );

/// Output the given \c track_set object to the specified file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.
 *
 * \note The simple track file format does not represent all data within
 *       a track. This version only stores the track topology.
 * \throws file_write_exception
 *    Thrown when something prevents output of the file.
 *
 * \param tracks    The \c track_set object to output.
 * \param file_path The path to output the file to.
 */
void
VITAL_EXPORT write_track_file( track_set_sptr const&  tracks,
                               path_t const&          file_path );

/// Read in a feature track file, producing a feature_track_set
/**
 * \note The simple track file format does not represent all data within
 *       a track. This version only handles track topology and feature points.
 * \throws file_not_found_exception
 *    Thrown when the file could not be found on the file system.
 * \throws file_not_read_exception
 *    Thrown when the file could not be read or parsed for whatever reason.
 *
 * \param file_path   The path to the file to read in.
 * \return A \c track_set object representing the contents of the read-in file.
 */
feature_track_set_sptr
VITAL_EXPORT read_feature_track_file( path_t const& file_path );

/// Output the given \c feature_track_set object to the specified file path
/**
 * If a file exists at the target location, it will be overwritten. If the
 * containing directory of the given path does not exist, it will be created
 * before the file is opened for writing.
 *
 * \note The simple track file format does not represent all data within
 *       a track.  This version only stores the track topology and
 *       feature points.
 * \throws file_write_exception
 *    Thrown when something prevents output of the file.
 * \throws invalid_data
 *    Thrown when a track state does not contain a valid feature
 *
 * \param tracks    The \c track_set object to output.
 * \param file_path The path to output the file to.
 */
void
VITAL_EXPORT write_feature_track_file( feature_track_set_sptr const&  tracks,
                                       path_t const&                  file_path );

} } // end namespace

#endif // VITAL_TRACK_SET_IO_H_
