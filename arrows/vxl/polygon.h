// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
