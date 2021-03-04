// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief File description here.
 */

#ifndef VITAL_C_HELPERS_LANDMARK_H_
#define VITAL_C_HELPERS_LANDMARK_H_

#include <vital/types/landmark.h>
#include <vital/bindings/c/helpers/c_utils.h>
#include <vital/bindings/c/types/landmark.h>

namespace kwiver {
namespace vital_c {

/// Cache for landmark shared pointers that exit the C++ barrier
extern
SharedPointerCache< vital::landmark, vital_landmark_t > LANDMARK_SPTR_CACHE;

}
}

#endif //VITAL_C_HELPERS_LANDMARK_H_
