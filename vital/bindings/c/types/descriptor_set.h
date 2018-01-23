/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief vital::descriptor_set interface functions
 */

#ifndef VITAL_C_DESCRIPTOR_SET_H_
#define VITAL_C_DESCRIPTOR_SET_H_

#include <cstddef>

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/descriptor.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// Base opaque descriptor instance type
typedef struct vital_descriptor_set_s vital_descriptor_set_t;


/// Create a new descriptor set from the array of descriptors.
/**
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
vital_descriptor_set_t*
vital_descriptor_set_new( vital_descriptor_t const **d_array,
                          size_t d_array_length,
                          vital_error_handle_t *eh );


/// Destroy a descriptor set
/**
 * \param ds Handle of the descriptor set instance to destroy.
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
void
vital_descriptor_set_destroy( vital_descriptor_set_t const *ds,
                              vital_error_handle_t *eh );


/// Get the size of a descriptor set
/**
 * \param ds The handle of the descriptor set instance.
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
size_t
vital_descriptor_set_size( vital_descriptor_set_t const *ds,
                           vital_error_handle_t *eh );


/// Get the descritpors stored in this set.
/**
 * \param ds The handle descriptor set instance.
 * \param[out] d_array Output array of descriptor instance handles. This array
 *   was created via malloc and the caller is responsible for freeing the
 *   array.
 * \param[out] d_array_length Output array length.
 * \param eh Vital error handle instance.
 */
VITAL_C_EXPORT
void
vital_descriptor_set_get_descriptors( vital_descriptor_set_t const *ds,
                                      vital_descriptor_t ***out_d_array,
                                      size_t *out_d_array_length,
                                      vital_error_handle_t *eh );


#ifdef __cplusplus
}
#endif

#endif // VITAL_C_DESCRIPTOR_SET_H_
