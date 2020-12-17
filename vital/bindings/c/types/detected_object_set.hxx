// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C++ interface to vital::detected_object_set class
 */

#ifndef VITAL_C_DETECTED_OBJECT_SET_HXX_
#define VITAL_C_DETECTED_OBJECT_SET_HXX_

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/detected_object_set.h>
#include <vital/types/detected_object_set.h>

// ------------------------------------------------------------------
// These two functions are a bridge between c++ and the internal smart pointer management
// ------------------------------------------------------------------

// Adopt previously created image container
VITAL_C_EXPORT
vital_detected_object_set_t* vital_detected_object_set_from_sptr( kwiver::vital::detected_object_set_sptr sptr );

// Return sptr from handle
VITAL_C_EXPORT
kwiver::vital::detected_object_set_sptr vital_detected_object_set_to_sptr( vital_detected_object_set_t* handle );

#endif // VITAL_C_DETECTED_OBJECT_SET_HXX_
