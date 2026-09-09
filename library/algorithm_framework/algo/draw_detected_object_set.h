// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header for draw_detected_object_set

#ifndef VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H
#define VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H

#include <viame/algorithm_framework/vital_config.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/image_container.h>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class for algorithms which draw tracks on top of
/// images in various ways, for analyzing results.
class VITAL_ALGO_EXPORT draw_detected_object_set
  : public kwiver::vital::algorithm
{
public:
  /// Return the name of this algorithm.
  draw_detected_object_set();
  PLUGGABLE_INTERFACE( draw_detected_object_set );
  /// Draw detected object boxes on Image.
  ///
  /// This method draws the detections on a copy of the image. The
  /// input image is unmodified. The actual boxes that are drawn are
  /// controlled by the configuration for the implementation.
  ///
  /// @param detected_set Set of detected objects
  /// @param image Boxes are drawn in this image
  ///
  /// @return Image with boxes and other annotations added.
  virtual kwiver::vital::image_container_sptr
  draw(
    kwiver::vital::detected_object_set_sptr detected_set,
    kwiver::vital::image_container_sptr image ) = 0;
};

/// A smart pointer to a draw_tracks instance.
typedef std::shared_ptr< draw_detected_object_set >
  draw_detected_object_set_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H
