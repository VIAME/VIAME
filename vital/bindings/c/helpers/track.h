// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface vital::track helpers
 */

#ifndef VITAL_C_HELPERS_TRACK_H_
#define VITAL_C_HELPERS_TRACK_H_

#include <vital/types/track.h>

#include <vital/bindings/c/types/track.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

/// Cache for saving shared pointer references for pointers in use
extern
SharedPointerCache< vital::track, vital_track_t > TRACK_SPTR_CACHE;

/// Cache for saving shared pointer references for pointers in use
extern
SharedPointerCache< vital::track_state, vital_track_state_t > TRACK_STATE_SPTR_CACHE;

}
}

#endif // VITAL_C_HELPERS_TRACK_H_
