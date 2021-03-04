// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C/C++ interface to vital::track_set class
 */

#ifndef VITAL_C_TRACK_SET_HXX_
#define VITAL_C_TRACK_SET_HXX_

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/track_set.h>
#include <vital/types/track_set.h>

// -----------------------------------------------------------------------------
// These two functions are a bridge between C++ and the internal C smart pointer
// management.
// -----------------------------------------------------------------------------

/// Create a vital_trackset_t around an existing shared pointer.
/**
 * If an error occurs, a NULL pointer is returned.
 *
 * \param ds Shared pointer to a vital::track_set instance.
 * \param eh Vital error handle instance. May be null to ignore errors.
 */
VITAL_C_EXPORT
vital_trackset_t*
vital_track_set_new_from_sptr( kwiver::vital::track_set_sptr ts_sptr,
                               vital_error_handle_t* eh );

/// Get the vital::track_set shared pointer for a handle.
/**
 * If an error occurs, an empty shared pointer is returned.
 *
 * \param ds Vital C handle to the track_set instance to get the shared
 *   pointer reference of.
 * \param eh Vital error handle instance. May be null to ignore errors.
 */
VITAL_C_EXPORT
kwiver::vital::track_set_sptr
vital_track_set_to_sptr( vital_trackset_t* ts,
                         vital_error_handle_t* eh );

#endif // VITAL_C_TRACK_SET_HXX_
