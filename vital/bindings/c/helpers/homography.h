// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::homography shared pointer cache definition
 */

#ifndef VITAL_C_HELPERS_HOMOGRAPHY_H_
#define VITAL_C_HELPERS_HOMOGRAPHY_H_

#include <vital/types/homography.h>
#include <vital/bindings/c/types/homography.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

extern SharedPointerCache< vital::homography, vital_homography_t >
  HOMOGRAPHY_SPTR_CACHE;

} // end namespace vital_c
} // end namespace kwiver

#endif //VITAL_C_HELPERS_HOMOGRAPHY_H_
