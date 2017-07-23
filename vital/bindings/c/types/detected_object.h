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
 * \brief C interface to vital::image classes
 */

#ifndef VITAL_C_DETECTED_OBJECT_H_
#define VITAL_C_DETECTED_OBJECT_H_

#include <vital/bindings/c/vital_c_export.h>
#include <vital/bindings/c/types/bounding_box.h>
#include <vital/bindings/c/types/detected_object_type.h>
#include <vital/bindings/c/types/image.h>

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// VITAL Image opaque structure
typedef struct vital_detected_object_s vital_detected_object_t;

VITAL_C_EXPORT
void vital_detected_object_destroy(vital_detected_object_t * obj);

VITAL_C_EXPORT
vital_detected_object_t* vital_detected_object_new_with_bbox(
  vital_bounding_box_t* bbox,
  double confidence,
  vital_detected_object_type_t* dot); // optional, could be NULL

VITAL_C_EXPORT
vital_detected_object_t* vital_detected_object_copy(vital_detected_object_t * obj);

VITAL_C_EXPORT
vital_bounding_box_t* vital_detected_object_bounding_box(vital_detected_object_t * obj);

VITAL_C_EXPORT
void vital_detected_object_set_bounding_box( vital_detected_object_t * obj,
                                             vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_detected_object_confidence(vital_detected_object_t * obj);

VITAL_C_EXPORT
void vital_detected_object_set_confidence( vital_detected_object_t * obj,
                                           double conf );

VITAL_C_EXPORT
vital_detected_object_type_t* vital_detected_object_get_type(vital_detected_object_t * obj);

VITAL_C_EXPORT
void vital_detected_object_set_type( vital_detected_object_t * obj,
                                     vital_detected_object_type_t* ob_type);

VITAL_C_EXPORT
int64_t vital_detected_object_index(vital_detected_object_t * obj);

VITAL_C_EXPORT
void vital_detected_object_set_index(vital_detected_object_t * obj,
                                     int64_t idx);

VITAL_C_EXPORT
char* vital_detected_object_detector_name(vital_detected_object_t * obj);

VITAL_C_EXPORT
void vital_detected_object_detector_setname(vital_detected_object_t * obj,
                                            char* name );

VITAL_C_EXPORT
vital_image_t* vital_detected_object_mask(vital_detected_object_t * obj);

VITAL_C_EXPORT
void vital_detected_object_set_mask(vital_detected_object_t * obj,
                                    vital_image_t* mask);

#ifdef __cplusplus
}
#endif

#endif /* VITAL_C_DETECTED_OBJECT_H_ */
