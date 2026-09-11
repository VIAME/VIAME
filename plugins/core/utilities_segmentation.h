/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_CORE_UTILITIES_SEGMENTATION_H
#define VIAME_CORE_UTILITIES_SEGMENTATION_H

#include "viame_core_export.h"

#include <viame/core_types/image.h>
#include <viame/core_types/point.h>

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

/// One `(poly)` or `(hole)` run of a detection's mask, ready to write
struct VIAME_CORE_EXPORT mask_contour
{
  std::vector< kwiver::vital::point_2i > points;

  /// True when this is a hole, which the CSV spells `(hole)`
  bool is_hole = false;
};

/// The polygons a mask becomes in the viame CSV.
///
/// `cv::findContours` under `RETR_CCOMP` with `CHAIN_APPROX_SIMPLE`, then
/// either `cv::approxPolyDP` at \p tolerance -- relative to the shorter side
/// of each contour's own bounding box -- or `simplify_polygon` down to
/// \p max_points, exactly as the two CSV writers did it with OpenCV. A
/// negative \p tolerance selects the point count.
///
/// The points are in mask coordinates; the caller adds the box's origin.
///
/// \param mask the detection's mask, one plane, non-zero inside
/// \param tolerance relative Douglas-Peucker tolerance, or negative
/// \param max_points the cap used when \p tolerance is negative
VIAME_CORE_EXPORT
std::vector< mask_contour >
mask_to_contours( kwiver::vital::image const& mask,
                  double tolerance, int max_points );

} // end namespace viame

#endif // VIAME_CORE_UTILITIES_SEGMENTATION_H
