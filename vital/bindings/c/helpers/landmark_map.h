// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File description here.
 */

#ifndef VITAL_C_HELPERS_LANDMARK_MAP_H_
#define VITAL_C_HELPERS_LANDMARK_MAP_H_

#include <vital/types/landmark_map.h>
#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/types/landmark_map.h>

namespace kwiver {
namespace vital_c {

/// Cache for landmark_map shared pointers that exit the C++ barrier
extern
SharedPointerCache< vital::landmark_map, vital_landmark_map_t >
LANDMARK_MAP_SPTR_CACHE;

}
}

#endif //VITAL_C_HELPERS_LANDMARK_MAP_H_
