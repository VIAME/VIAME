/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_SEGMENTATION_H
#define VIAME_CORE_UTILITIES_SEGMENTATION_H

#include "viame_core_export.h"

#include <vital/types/point.h>

#include <vector>
#include <cstddef>

namespace viame {

/// Simplify a polygon to have at most max_points vertices
///
/// Uses a modified Ramer-Douglas-Peucker algorithm. Instead of keeping
/// points out of tolerance, we iteratively add the most significant
/// points until we reach the maximum point count.
///
/// \param curve Input polygon as vector of 2D integer points
/// \param max_points Maximum number of points in output (minimum 2)
/// \returns Simplified polygon with at most max_points vertices
VIAME_CORE_EXPORT
std::vector< kwiver::vital::point_2i >
simplify_polygon( std::vector< kwiver::vital::point_2i > const& curve,
                  size_t max_points );

/// Simplify a polygon to have at most max_points vertices (double precision)
///
/// Uses a modified Ramer-Douglas-Peucker algorithm. Instead of keeping
/// points out of tolerance, we iteratively add the most significant
/// points until we reach the maximum point count.
///
/// \param curve Input polygon as vector of 2D double points
/// \param max_points Maximum number of points in output (minimum 2)
/// \returns Simplified polygon with at most max_points vertices
VIAME_CORE_EXPORT
std::vector< kwiver::vital::point_2d >
simplify_polygon( std::vector< kwiver::vital::point_2d > const& curve,
                  size_t max_points );

} // end namespace viame

#endif // VIAME_CORE_UTILITIES_SEGMENTATION_H
