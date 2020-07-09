/*ckwg +29
 * Copyright 2016-2020 by Kitware, Inc.
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
 * \brief Interface for class_map class
 */

#ifndef VITAL_C_CLASS_MAP_H_
#define VITAL_C_CLASS_MAP_H_

#ifdef __cplusplus
extern "C"
{
#endif

#include <vital/bindings/c/vital_c_export.h>

#include <stddef.h>

/// VITAL Image opaque structure
typedef struct vital_class_map_s vital_class_map_t;


VITAL_C_EXPORT
vital_class_map_t* vital_class_map_new();

VITAL_C_EXPORT
void vital_class_map_destroy(vital_class_map_t* obj);

VITAL_C_EXPORT
vital_class_map_t* vital_class_map_new_from_list( vital_class_map_t* obj,
                                                  size_t count,
                                                  char** class_names,
                                                  double* scores);

VITAL_C_EXPORT
bool vital_class_map_has_class_name( vital_class_map_t* obj, char* class_name );

VITAL_C_EXPORT
double vital_class_map_score( vital_class_map_t* obj, char* class_name );

VITAL_C_EXPORT
char* vital_class_map_get_most_likely_class( vital_class_map_t* obj);

VITAL_C_EXPORT
double vital_class_map_get_most_likely_score( vital_class_map_t* obj);

VITAL_C_EXPORT
void vital_class_map_set_score( vital_class_map_t* obj, char* class_name, double score);

VITAL_C_EXPORT
void vital_class_map_delete_score( vital_class_map_t* obj, char* class_name);

VITAL_C_EXPORT
char** vital_class_map_class_names( vital_class_map_t* obj, double thresh );

VITAL_C_EXPORT
char** vital_class_map_all_class_names(vital_class_map_t* obj);


#ifdef __cplusplus
}
#endif

#endif /* VITAL_C_CLASS_MAP_H_ */
