// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C++ Helper utilities for C interface of vital::detected_object_set
 *
 * Private header for use in cxx implementation files.
 */

#ifndef VITAL_C_HELPERS_DETECTED_OBJECT_SET_H_
#define VITAL_C_HELPERS_DETECTED_OBJECT_SET_H_

#include <vital/types/detected_object_set.h>

#include <vital/bindings/c/types/detected_object_set.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

/// Declaration of C interface shared_ptr cache of vital::detected_object_set
extern SharedPointerCache< kwiver::vital::detected_object_set,
                           vital_detected_object_set_t > DOBJ_SET_SPTR_CACHE;

} }

#endif // VITAL_C_HELPERS_DETECTED_OBJECT_SET_H_
