// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C Interface to convert_image algorithm definition
 */

#ifndef VITAL_C_ALGO_DEF_CONVERT_IMAGE_H_
#define VITAL_C_ALGO_DEF_CONVERT_IMAGE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/algorithm.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/image_container.h>

/// Declare common type-specific functions
DECLARE_COMMON_ALGO_API( convert_image );

/// Convert image base type
/**
 * Returns new image container with a converted underlying representation
 * based on the implementation configured.
 *
 * \param algo Opaque pointer to algorithm instance.
 * \param ic Image container with the image data to convert.
 * \param eh Error handle instance.
 * \return New image container with the converted underlying data.
 */
VITAL_C_EXPORT
vital_image_container_t*
vital_algorithm_convert_image_convert( vital_algorithm_t *algo,
                                       vital_image_container_t *ic,
                                       vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_ALGO_DEF_CONVERT_IMAGE_H_
