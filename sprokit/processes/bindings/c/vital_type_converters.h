/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
@file Interface to kwiver level vital type converters
 */

#ifndef KWIVER_VITAL_TYPE_CONVERTERS_H
#define KWIVER_VITAL_TYPE_CONVERTERS_H

#include <processes/bindings/c/vital_type_converters_export.h>

#include <vital/bindings/c/common.h>
#include <vital/bindings/c/types/descriptor_set.h>
#include <vital/bindings/c/types/detected_object_set.h>
#include <vital/bindings/c/types/image_container.h>
#include <vital/bindings/c/types/track_set.h>

#include <sprokit/python/util/python.h>

#ifdef __cplusplus
extern "C"
{
#endif

VITAL_TYPE_CONVERTERS_EXPORT
vital_image_container_t* vital_image_container_from_datum( PyObject* args );

VITAL_TYPE_CONVERTERS_EXPORT
PyObject* vital_image_container_to_datum( vital_image_container_t* handle );

VITAL_TYPE_CONVERTERS_EXPORT
vital_detected_object_set_t* vital_detected_object_set_from_datum( PyObject* args );

VITAL_TYPE_CONVERTERS_EXPORT
PyObject* vital_detected_object_set_to_datum( vital_detected_object_set_t* handle );

VITAL_TYPE_CONVERTERS_EXPORT
double* double_vector_from_datum( PyObject* args );

VITAL_TYPE_CONVERTERS_EXPORT
PyObject* double_vector_to_datum( PyObject* list );

VITAL_TYPE_CONVERTERS_EXPORT
vital_trackset_t* vital_trackset_from_datum( PyObject* dptr );

VITAL_TYPE_CONVERTERS_EXPORT
PyObject* vital_trackset_to_datum( vital_trackset_t* handle );

VITAL_TYPE_CONVERTERS_EXPORT
vital_trackset_t* vital_object_trackset_from_datum( PyObject* dptr );

VITAL_TYPE_CONVERTERS_EXPORT
PyObject* vital_object_trackset_to_datum( vital_trackset_t* handle );

/// Convert a sprokit::datum boost::any value into an array of null-terminated
/// strings.
/**
 * \param args sprokit datum wrapped in a PyCapsule object.
 * \param[out] out_vec Output array of new char* instances.
 *   The caller of this function is responsible for freeing these strings.
 * \param[out] out_vec_size Output size_t value that is the size of the output
 *   array.
 */
VITAL_TYPE_CONVERTERS_EXPORT
void
vital_string_vector_from_datum( PyObject *args,
                                char ***out_strings,
                                size_t *out_size );

/// Convert a python list of strings into a string vector datum for sprokit.
/**
 * \param list Python list object of python string objects.
 * \return PyCapsule object containing the sprokit datum.
 */
VITAL_TYPE_CONVERTERS_EXPORT
PyObject*
vital_string_vector_to_datum( PyObject *list );

/// Convert a sprokit::datum into a vital_descriptor_set_t.
/**
 * \oaram args sprokit::datum wrapped in a PyCapsule object.
 * \return Vital C descriptor_set instance handle.
 */
VITAL_TYPE_CONVERTERS_EXPORT
vital_descriptor_set_t*
vital_descriptor_set_from_datum( PyObject *args );

/// Convert a vital_descriptor_set_t into a sprokit::datum capsule.
/**
 * \param vital_ds Vital C descriptor_set handle.
 * \return PyCapsule object wrapping the output sprokit::datum.
 */
VITAL_TYPE_CONVERTERS_EXPORT
PyObject*
vital_descriptor_set_to_datum( vital_descriptor_set_t* vital_ds );

  // others

#ifdef __cplusplus
}
#endif

#endif /* KWIVER_VITAL_TYPE_CONVERTERS_H */
