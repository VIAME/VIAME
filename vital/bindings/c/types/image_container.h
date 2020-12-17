// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::image_container class
 */

#ifndef VITAL_C_IMAGE_CONTAINER_H_
#define VITAL_C_IMAGE_CONTAINER_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stddef.h>

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/image.h>

/// VITAL Image opaque structure
typedef struct vital_image_container_s vital_image_container_t;

/// Create a new, simple image container around an image
VITAL_C_EXPORT
vital_image_container_t* vital_image_container_new_simple( vital_image_t *img );

/// Destroy a vital_image_container_t instance
VITAL_C_EXPORT
void vital_image_container_destroy( vital_image_container_t *img_container,
                                    vital_error_handle_t *eh );

/// Get the size in bytes of an image container
/**
 * Size includes all allocated image memory, which could be larger than
 * the product of width, height and depth.
 */
VITAL_C_EXPORT
size_t vital_image_container_size( vital_image_container_t *img_c );

/// Get the width of the given image in pixels
VITAL_C_EXPORT
size_t vital_image_container_width( vital_image_container_t *img_c );

/// Get the height of the given image in pixels
VITAL_C_EXPORT
size_t vital_image_container_height( vital_image_container_t *img_c );

/// Get the depth (number of channels) of the image
VITAL_C_EXPORT
size_t vital_image_container_depth( vital_image_container_t *img_c );

/// Get the in-memory image class to access data
VITAL_C_EXPORT
vital_image_t* vital_image_container_get_image( vital_image_container_t *img_c );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_IMAGE_CONTAINER_H_
