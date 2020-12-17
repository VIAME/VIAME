// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Internal helpers for C interface of vital::camera_map
 *
 * This is intended for use only in C++ implementation files.
 */

#ifndef VITAL_C_HELPERS_CAMERA_MAP_H_
#define VITAL_C_HELPERS_CAMERA_MAP_H_

#include <vital/types/camera_map.h>
#include <vital/bindings/c/types/camera_map.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

extern SharedPointerCache< vital::camera_map,
                           vital_camera_map_t > CAM_MAP_SPTR_CACHE;

} }

#endif // VITAL_C_HELPERS_CAMERA_MAP_H_
