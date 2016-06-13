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
 * \brief vital::descriptor interface functions
 */

#ifndef VITAL_C_DESCRIPTOR_H_
#define VITAL_C_DESCRIPTOR_H_


#ifdef __cplusplus
extern "C"
{
#endif

#include <cstddef>

#include <vital/bindings/c/error_handle.h>
#include <vital/bindings/c/vital_c_export.h>


////////////////////////////////////////////////////////////////////////////////
// General Descriptor functions

/// Base opaque descriptor instance type
typedef struct {} vital_descriptor_t;


/// Destroy a descriptor instance
/**
 * \param d descriptor instance to destroy
 * \param eh Vital error handle instance
 */
VITAL_C_EXPORT
void
vital_descriptor_destroy( vital_descriptor_t const *d,
                          vital_error_handle_t *eh );


/// Get the number of elements in the descriptor
/**
 * \param d descriptor instance to destroy
 * \param eh Vital error handle instance
 * \returns Number of value elements in the descriptor
 */
VITAL_C_EXPORT
size_t
vital_descriptor_size( vital_descriptor_t const *d,
                       vital_error_handle_t *eh );


/// Get the number of bytes used to represent the descriptor's data
/**
 * \param d descriptor instance to destroy
 * \param eh Vital error handle instance
 * \returns number of bytes the descriptor occupies
 */
VITAL_C_EXPORT
size_t
vital_descriptor_num_bytes( vital_descriptor_t const *d,
                            vital_error_handle_t *eh );


/// Convert the descriptor into a new array of bytes
/**
 * Length of returned array is the same as ``vital_descriptor_num_bytes``.
 *
 * \param d descriptor instance to destroy
 * \param eh Vital error handle instance
 * \returns new array of bytes
 */
VITAL_C_EXPORT
unsigned char*
vital_descriptor_as_bytes( vital_descriptor_t const *d,
                           vital_error_handle_t *eh );


/// Convert the descriptor into a new array of doubles
/**
 * Length of the returned array is the same as ``vital_descriptor_size``.
 *
 * \param d descriptor instance to destroy
 * \param eh Vital error handle instance
 * \returns new array of doubles
 */
VITAL_C_EXPORT
double*
vital_descriptor_as_doubles( vital_descriptor_t const *d,
                             vital_error_handle_t *eh );


////////////////////////////////////////////////////////////////////////////////
// Type-specific functions (and constructors)

#define DECLARE_TYPED_OPERATIONS( T, S ) \
\
/**
 * Create a new descriptor of a the given size (vital::descriptor_dynamic<T>)
 *
 * \param size The size of the initialized descriptor
 * \param eh Vital error handle instance.
 */ \
VITAL_C_EXPORT \
vital_descriptor_t* \
vital_descriptor_new_##S( size_t size, \
                          vital_error_handle_t *eh ); \
\
/**
 * Get the pointer to the descriptor's data array
 *
 * Length of the returned array is the same as ``vital_descriptor_size``.
 *
 * \param d descriptor instance to destroy
 * \param eh Vital error handle instance
 * \returns Pointer to our data buffer of type T
 */ \
VITAL_C_EXPORT \
T* \
vital_descriptor_get_##S##_raw_data( vital_descriptor_t *d, \
                                     vital_error_handle_t *eh );


DECLARE_TYPED_OPERATIONS( double, d )
DECLARE_TYPED_OPERATIONS( float,  f )

#undef DECLARE_TYPED_OPERATIONS


#ifdef __cplusplus
}
#endif

#endif //VITAL_C_DESCRIPTOR_H_
