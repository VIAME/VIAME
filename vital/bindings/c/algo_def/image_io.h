// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to image_io algorithm definition
 */

#ifndef VITAL_C_ALGO_DEF_IMAGE_IO_H_
#define VITAL_C_ALGO_DEF_IMAGE_IO_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/types/image_container.h>

/// Declare common type-specific functions
DECLARE_COMMON_ALGO_API( image_io );

/// Load image from file
/**
 * \param algo Opaque pointer to algorithm instance.
 * \param filename The string file path to where the image should be loaded
 *                 from.
 * \param eh Error handle instance.
 * \return New image container instance containing the image memory for the
 *         loaded image file.
 */
VITAL_C_EXPORT
vital_image_container_t* vital_algorithm_image_io_load( vital_algorithm_t *algo,
                                                        char const *filename,
                                                        vital_error_handle_t *eh);

/// Save an image to file
/**
 * \param algo Opaque pointer to algorithm instance.
 * \param filename String file path to where the image should be saved.
 * \param ic The image containing containing the image data to save to file.
 * \param eh Error handle instance.
 */
VITAL_C_EXPORT
void vital_algorithm_image_io_save( vital_algorithm_t *algo,
                                    char const *filename,
                                    vital_image_container_t *ic,
                                    vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_ALGO_DEF_IMAGE_IO_H_
