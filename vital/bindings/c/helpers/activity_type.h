// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C++ Helper utilities for C interface of vital::activity_type
 *
 * Private header for use in cxx implementation files.
 */

#ifndef VITAL_C_HELPERS_ACTIVTY_TYPE_H_
#define VITAL_C_HELPERS_ACTIVTY_TYPE_H_

#include <vital/types/activity_type.h>

#include <vital/bindings/c/types/activity_type.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

/// Declaration of C interface shared_ptr cache of vital::activity_type
extern SharedPointerCache< kwiver::vital::activity_type,
                           vital_activity_type_t > AT_SPTR_CACHE;

} }

#endif //VITAL_C_HELPERS_ACTIVTY_TYPE_H_
