// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C/C++ interface to vital::descriptor_set class
 */

#ifndef VITAL_C_DESCRIPTOR_SET_HXX_
#define VITAL_C_DESCRIPTOR_SET_HXX_

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/descriptor_set.h>
#include <vital/types/descriptor_set.h>

// -----------------------------------------------------------------------------
// These two functions are a bridge between C++ and the internal C smart pointer
// management.
// -----------------------------------------------------------------------------

/// Create a vital_descriptor_set_t around an existing shared pointer.
/**
 * If an error occurs, a NULL pointer is returned.
 *
 * \param ds Shared pointer to a vital::descriptor_set instance.
 * \param eh Vital error handle instance. May be null to ignore errors.
 */
VITAL_C_EXPORT
vital_descriptor_set_t*
vital_descriptor_set_new_from_sptr( kwiver::vital::descriptor_set_sptr ds_sptr,
                                    vital_error_handle_t* eh );

/// Get the vital::descriptor_set shared pointer for a handle.
/**
 * If an error occurs, an empty shared pointer is returned.
 *
 * \param ds Vital C handle to the descriptor_set instance to get the shared
 *   pointer reference of.
 * \param eh Vital error handle instance. May be null to ignore errors.
 */
VITAL_C_EXPORT
kwiver::vital::descriptor_set_sptr
vital_descriptor_set_to_sptr( vital_descriptor_set_t* ds,
                              vital_error_handle_t* eh );

#endif // VITAL_C_DESCRIPTOR_SET_HXX_
