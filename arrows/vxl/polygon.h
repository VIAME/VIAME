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
 * \file
 * \brief vxl polygon conversion interface
 */

#ifndef KWIVER_ALGORITHM_VXL_POLYGON_H
#define KWIVER_ALGORITHM_VXL_POLYGON_H

#include <arrows/vxl/kwiver_algo_vxl_export.h>
#include <vital/types/polygon.h>
#include <vgl/vgl_polygon.h>

#include <memory>

namespace kwiver {
namespace arrows {
namespace vxl {

/**
 * @brief Convert vital polygon to vxl polygon.
 *
 * This method converts a vital polygon into a
 * vgl_polygon. A new vgl_polygon is created and returned.
 *
 * @param poly Vital polygon
 *
 * @return Shared pointer to vgl_polygon.
 */
KWIVER_ALGO_VXL_EXPORT
std::shared_ptr< vgl_polygon< double > > vital_to_vxl( kwiver::vital::polygon_sptr poly );

/**
 * @brief Convert vgl_polygon to vital polygon.
 *
 * This method converts a vgl polygon into a vital polygon
 * object. Only the first sheet of the vgl_polygon is converted.
 *
 * @param poly vgl_polygon
 *
 * @return shared pointer to new vital polygon
 *
 * @throws std::out_of_range if the input polygon does not have any sheets/contours.
 */
KWIVER_ALGO_VXL_EXPORT
kwiver::vital::polygon_sptr vxl_to_vital( const vgl_polygon< double >& poly );

} } } // end namespace

#endif // KWIVER_ALGORITHM_VXL_POLYGON_H
