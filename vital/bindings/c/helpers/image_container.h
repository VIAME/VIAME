// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C++ Helper utilities for C interface of vital::image_container
 *
 * Private header for use in cxx implementation files.
 */

#ifndef VITAL_C_HELPERS_IMAGE_CONTAINER_H_
#define VITAL_C_HELPERS_IMAGE_CONTAINER_H_

#include <vital/types/image_container.h>

#include <vital/bindings/c/types/image_container.h>
#include <vital/bindings/c/helpers/c_utils.h>

namespace kwiver {
namespace vital_c {

/// Declaration of C interface shared_ptr cache of vital::image_container
extern SharedPointerCache< kwiver::vital::image_container,
                           vital_image_container_t > IMGC_SPTR_CACHE;

} }

#endif //VITAL_C_HELPERS_IMAGE_CONTAINER_H_
