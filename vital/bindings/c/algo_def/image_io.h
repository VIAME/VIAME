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
