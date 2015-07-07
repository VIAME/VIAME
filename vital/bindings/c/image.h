/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief C interface to vital::image classes
 */

#ifndef VITAL_C_IMAGE_H_
#define VITAL_C_IMAGE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <stdbool.h>
#include <stddef.h>

#include <vital/bindings/c/vital_c_export.h>


/// VITAL Image opaque structure
typedef struct vital_image_s vital_image_t;


/// Create a new, empty image
VITAL_C_EXPORT
vital_image_t* vital_image_new();


/// Create a new image with dimensions, allocating memory
VITAL_C_EXPORT
vital_image_t* vital_image_new_with_dim( size_t width, size_t height,
                                         size_t depth, bool interleave );


/// Create a new image from existing data
VITAL_C_EXPORT
vital_image_t* vital_image_new_from_data( unsigned char const *first_pixel,
                                          size_t width, size_t height,
                                          size_t depth, ptrdiff_t w_step,
                                          ptrdiff_t h_step, ptrdiff_t d_step );


/// Create a new image from an existing image, sharing the same memory
/**
 * The new image will share the same memory as the old image
 */
VITAL_C_EXPORT
vital_image_t* vital_image_new_from_image( vital_image_t *other_image );


/// Destroy an image instance
VITAL_C_EXPORT
void vital_image_destroy( vital_image_t *image );


/// Get the number of bytes allocated in the given image
VITAL_C_EXPORT
size_t vital_image_size( vital_image_t *image );


#ifdef __cplusplus
}
#endif

#endif // VITAL_C_IMAGE_H_
