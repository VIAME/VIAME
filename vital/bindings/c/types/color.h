/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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
typedef struct {} vital_rgb_color_t;


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
