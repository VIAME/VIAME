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
 * \brief C interface to vital::detected_object_set class
 */

#ifndef VITAL_C_DETECTED_OBJECT_SET_H_
#define VITAL_C_DETECTED_OBJECT_SET_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/detected_object.h>
#include <vital/bindings/c/types/detected_object_set.h>

#include <stddef.h>

/// VITAL Image opaque structure
typedef struct vital_detected_object_set_s vital_detected_object_set_t;

VITAL_C_EXPORT
vital_detected_object_set_t* vital_detected_object_set_new();

VITAL_C_EXPORT
vital_detected_object_set_t* vital_detected_object_set_new_from_list( vital_detected_object_t** dobj,
                                                                      size_t n);

VITAL_C_EXPORT
void vital_detected_object_set_destroy( vital_detected_object_set_t* obj);

VITAL_C_EXPORT
void vital_detected_object_set_add( vital_detected_object_set_t* set,
                                    vital_detected_object_t* obj );

VITAL_C_EXPORT
size_t vital_detected_object_set_size( vital_detected_object_set_t* obj);

VITAL_C_EXPORT
vital_detected_object_t** vital_detected_object_set_select_threshold( vital_detected_object_set_t* obj,
                                                                      double thresh,
                                                                      size_t* length );

VITAL_C_EXPORT
vital_detected_object_t** vital_detected_object_set_select_class_threshold( vital_detected_object_set_t* obj,
                                                                            const char* class_name,
                                                                            double thresh,
                                                                            size_t* length );

#ifdef __cplusplus
}
#endif

#endif // VITAL_C_DETECTED_OBJECT_SET_H_
