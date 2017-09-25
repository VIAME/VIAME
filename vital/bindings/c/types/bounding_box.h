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
 * \brief Interface for bounding box class
 */

#ifndef VITAL_C_BOUNDING_BOX_H_
#define VITAL_C_BOUNDING_BOX_H_

#include <vital/bindings/c/vital_c_export.h>

#ifdef __cplusplus
extern "C"
{
#endif

/// VITAL bounding_box opaque structure
typedef struct vital_bounding_box_s vital_bounding_box_t;

VITAL_C_EXPORT
vital_bounding_box_t* vital_bounding_box_new_from_vectors(
  double* ul, double* lr);

VITAL_C_EXPORT
vital_bounding_box_t* vital_bounding_box_new_from_point_width_height(
  double* ul, double  width, double height);

VITAL_C_EXPORT
vital_bounding_box_t* vital_bounding_box_new_from_coordinates(
  double xmin, double  ymin, double xmax, double ymax);

VITAL_C_EXPORT
vital_bounding_box_t* vital_bounding_box_copy(
  vital_bounding_box_t* bbox );

VITAL_C_EXPORT
void vital_bounding_box_destroy( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double* vital_bounding_box_center( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double* vital_bounding_box_upper_left( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double* vital_bounding_box_lower_right( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_min_x( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_max_x( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_min_y( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_max_y( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_width( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_height( vital_bounding_box_t* bbox );

VITAL_C_EXPORT
double vital_bounding_box_area( vital_bounding_box_t* bbox );

#ifdef __cplusplus
}
#endif

#endif /* VITAL_C_BOUNDING_BOX_H_ */
