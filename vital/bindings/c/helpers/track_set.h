// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface vital::track_set helpers
 */

#ifndef VITAL_C_HELPERS_TRACK_SET_H_
#define VITAL_C_HELPERS_TRACK_SET_H_

#include <vital/types/track_set.h>

#include <vital/bindings/c/types/track_set.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

/// Cache for saving shared pointer references for pointers in use
extern
SharedPointerCache< vital::track_set, vital_trackset_t >
  TRACK_SET_SPTR_CACHE;

} }

#endif // VITAL_C_HELPERS_TRACK_SET_H_
