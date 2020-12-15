// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief vital::camera C interface shared pointer cache declaration
 *
 * Private header for use in cxx implementation files.
 */

#ifndef VITAL_C_HELPERS_CAMERA_H_
#define VITAL_C_HELPERS_CAMERA_H_

#include <vital/types/camera_perspective.h>
#include <vital/bindings/c/types/camera.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

  extern SharedPointerCache< kwiver::vital::camera_perspective,
                           vital_camera_t > CAMERA_SPTR_CACHE;

} } // end namespace

#endif // VITAL_C_HELPERS_CAMERA_H_
