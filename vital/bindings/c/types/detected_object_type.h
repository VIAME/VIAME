/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * \brief Interface for detected_object_type class
 */

#ifndef VITAL_C_DETECTED_OBJECT_TYPE_H_
#define VITAL_C_DETECTED_OBJECT_TYPE_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>

#include <stddef.h>

/// VITAL Image opaque structure
typedef struct vital_detected_object_type_s vital_detected_object_type_t;


VITAL_C_EXPORT
vital_detected_object_type_t* vital_detected_object_type_new();

VITAL_C_EXPORT
void vital_detected_object_type_destroy(vital_detected_object_type_t* obj);

VITAL_C_EXPORT
vital_detected_object_type_t* vital_detected_object_type_new_from_list( vital_detected_object_type_t* obj,
                                                                        size_t count,
                                                                        char** class_names,
                                                                        double* scores);

VITAL_C_EXPORT
bool vital_detected_object_type_has_class_name( vital_detected_object_type_t* obj, char* class_name );

VITAL_C_EXPORT
double vital_detected_object_type_score( vital_detected_object_type_t* obj, char* class_name );

VITAL_C_EXPORT
char* vital_detected_object_type_get_most_likely_class( vital_detected_object_type_t* obj);

VITAL_C_EXPORT
double vital_detected_object_type_get_most_likely_score( vital_detected_object_type_t* obj);

VITAL_C_EXPORT
void vital_detected_object_type_set_score( vital_detected_object_type_t* obj, char* class_name, double score);

VITAL_C_EXPORT
void vital_detected_object_type_delete_score( vital_detected_object_type_t* obj, char* class_name);

VITAL_C_EXPORT
char** vital_detected_object_type_class_names( vital_detected_object_type_t* obj, double thresh );

VITAL_C_EXPORT
char** vital_detected_object_type_all_class_names(vital_detected_object_type_t* obj);


#ifdef __cplusplus
}
#endif

#endif /* VITAL_C_DETECTED_OBJECT_TYPE_H_ */
