// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for draw_detected_object_set
 */

#ifndef VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H
#define VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H

#include <vital/vital_config.h>

#include <vital/algo/algorithm.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object_set.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class for algorithms which draw tracks on top of
/// images in various ways, for analyzing results.
class VITAL_ALGO_EXPORT draw_detected_object_set
  : public kwiver::vital::algorithm_def<draw_detected_object_set>
{
public:

  /// Return the name of this algorithm.
  static std::string static_type_name() { return "draw_detected_object_set"; }

  /// Draw detected object boxes on Image.
  /**
   * This method draws the detections on a copy of the image. The
   * input image is unmodified. The actual boxes that are drawn are
   * controlled by the configuration for the implementation.
   *
   * @param detected_set Set of detected objects
   * @param image Boxes are drawn in this image
   *
   * @return Image with boxes and other annotations added.
   */
  virtual kwiver::vital::image_container_sptr
    draw( kwiver::vital::detected_object_set_sptr detected_set,
          kwiver::vital::image_container_sptr image ) = 0;

protected:
  draw_detected_object_set();

};

/// A smart pointer to a draw_tracks instance.
typedef std::shared_ptr<draw_detected_object_set> draw_detected_object_set_sptr;

} } } // end namespace

#endif // VITAL_ALGO_DRAW_DETECTED_OBJECT_SET_H
