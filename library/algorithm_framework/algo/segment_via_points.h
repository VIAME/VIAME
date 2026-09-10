// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header defining abstract segment_via_points algorithm

#ifndef VITAL_ALGO_SEGMENT_VIA_POINTS_H_
#define VITAL_ALGO_SEGMENT_VIA_POINTS_H_

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/point.h>

#include <vector>

namespace kwiver {

namespace vital {

namespace algo {

// ----------------------------------------------------------------------------

/// @brief Abstract base class for interactive point-based segmentation.
///
/// This algorithm performs segmentation on an image using point prompts.
/// Points are labeled as foreground (1) or background (0) to guide the
/// segmentation model in determining which regions to segment.
///
/// Typical usage:
/// - User clicks on an object (foreground point, label=1)
/// - Optionally clicks on background (background point, label=0)
/// - Algorithm returns detected objects with segmentation masks
///
class VITAL_ALGO_EXPORT segment_via_points
  : public kwiver::vital::algorithm
{
public:
  segment_via_points();
  PLUGGABLE_INTERFACE( segment_via_points );

  /// Perform point-based segmentation on an image.
  ///
  /// \param image The image to segment
  ///
  /// \param points Vector of 2D point coordinates [x, y] indicating
  ///        locations for segmentation prompts
  ///
  /// \param point_labels Vector of labels corresponding to each point:
  ///        - 1: foreground (object to segment)
  ///        - 0: background (region to exclude)
  ///        Must have same length as points vector.
  ///
  /// \returns DetectedObjectSet containing segmented objects.
  ///          Each DetectedObject includes:
  ///          - Bounding box around the segmented region
  ///          - Confidence score from the segmentation model
  ///          - Binary mask of the segmented region
  ///
  virtual detected_object_set_sptr
  segment(
    image_container_sptr image,
    std::vector< point_2d > const& points,
    std::vector< int > const& point_labels ) const = 0;
};

/// Shared pointer for segment_via_points algorithm definition type.
typedef std::shared_ptr< segment_via_points > segment_via_points_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_SEGMENT_VIA_POINTS_H_
