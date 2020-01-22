/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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
