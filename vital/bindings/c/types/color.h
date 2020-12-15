// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief C interface to vital::rgb_color
 */

#ifndef VITAL_C_COLOR_H_
#define VITAL_C_COLOR_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>

/// Opaque structure type
typedef struct vital_rgb_color_s vital_rgb_color_t;

/// Create a new rgb_color instance
/**
 * \param cr Red color value
 * \param cg Green color value
 * \param cb Blue color value
 * \param eh Vital error handle instance
 * \returns New instance
 */
VITAL_C_EXPORT
vital_rgb_color_t*
vital_rgb_color_new( unsigned char cr, unsigned char cg, unsigned char cb,
                     vital_error_handle_t *eh );

/// Create a new default (white) rgb_color instance
/**
 * \returns New rgb_color instance with default color values (white)
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
vital_rgb_color_t*
vital_rgb_color_new_default( vital_error_handle_t *eh );

/// Destroy an rgb_color instance
/**
 * \param c the instance to destroy
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_rgb_color_destroy( vital_rgb_color_t *c, vital_error_handle_t *eh );

/// Get the red value
/**
 * \param c rgb color instance
 * \param eh Vital error handle instance
 * \return Red color value
 */
VITAL_C_EXPORT
unsigned char
vital_rgb_color_r( vital_rgb_color_t *c, vital_error_handle_t *eh );

/// Get the green value
/**
 * \param c rgb color instance
 * \param eh Vital error handle instance
 * \return Green color value
 */
VITAL_C_EXPORT
unsigned char
vital_rgb_color_g( vital_rgb_color_t *c, vital_error_handle_t *eh );

/// Get the blue value
/**
 * \param c rgb color instance
 * \param eh Vital error handle instance
 * \return Blue color value
 */
VITAL_C_EXPORT
unsigned char
vital_rgb_color_b( vital_rgb_color_t *c, vital_error_handle_t *eh );

/// Test equality between two rgb_color instances
/**
 * \param a First color to compare
 * \param b Second color to compare
 * \returns If \c a and \c b are equal in color values.
 */
VITAL_C_EXPORT
bool
vital_rgb_color_is_equal( vital_rgb_color_t *a, vital_rgb_color_t *b,
                          vital_error_handle_t *eh );

#ifdef __cplusplus
}
#endif

#endif //VITAL_C_COLOR_H_
