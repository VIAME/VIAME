// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
