// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C++ interface to vital::image_container class
 */

#ifndef VITAL_C_IMAGE_CONTAINER_HXX_
#define VITAL_C_IMAGE_CONTAINER_HXX_

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/image_container.h>
#include <vital/types/image_container.h>

// ------------------------------------------------------------------
// These two functions are a bridge between c++ and the internal smart pointer management
// ------------------------------------------------------------------

// Adopt previously created image container
VITAL_C_EXPORT
vital_image_container_t* vital_image_container_from_sptr( kwiver::vital::image_container_sptr sptr );

// Return sptr from handle
VITAL_C_EXPORT
kwiver::vital::image_container_sptr vital_image_container_to_sptr( vital_image_container_t* handle );

#endif // VITAL_C_IMAGE_CONTAINER_HXX_
