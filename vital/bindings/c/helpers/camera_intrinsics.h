// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_C_HELPERS_CAMERA_INTRINSICS_H_
#define VITAL_C_HELPERS_CAMERA_INTRINSICS_H_

#include <vital/types/camera_intrinsics.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

/// Cache for saving shared pointer references
extern
SharedPointerCache< vital::camera_intrinsics, vital_camera_intrinsics_t >
  CAMERA_INTRINSICS_SPTR_CACHE;

} // end namespace vital_c
} // end namespace kwiver

#endif //VITAL_C_HELPERS_CAMERA_INTRINSICS_H_
