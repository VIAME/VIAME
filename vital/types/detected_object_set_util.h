// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef VITAL_DETECTED_OBJECT_SET_UTIL_H
#define VITAL_DETECTED_OBJECT_SET_UTIL_H

#include <vital/vital_export.h>
#include "detected_object_set.h"

namespace kwiver {
namespace vital {

/**
 * @brief Scale all detection locations by some scale factor.
 *
 * This method changes the bounding boxes within all stored detections
 * by scaling them by some scale factor.
 *
 * @param scale Scale factor
 */
void VITAL_EXPORT
scale_detections( detected_object_set_sptr dos,
                  double scale_factor );

/**
 * @brief Shift all detection locations by some translation offset.
 *
 * This method shifts the bounding boxes within all stored detections
 * by a supplied column and row shift.
 *
 * Note: Detections in this set can be shared by multiple sets, so
 * shifting the detections in this set will also shift the detection
 * in other sets that share this detection. If this is going to be a
 * problem, clone() this set before shifting.
 *
 * @param col_shift Column  (a.k.a. x, i, width) translation factor
 * @param row_shift Row (a.k.a. y, j, height) translation factor
 */
void VITAL_EXPORT
shift_detections( detected_object_set_sptr dos,
                  double col_shift, double row_shift );

} } // end namespace

#endif // VITAL_DETECTED_OBJECT_SET_UTIL_H
