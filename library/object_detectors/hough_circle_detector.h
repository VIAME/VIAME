// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_OCV_HOUGH_CIRCLE_DETECTOR_H
#define ARROWS_OCV_HOUGH_CIRCLE_DETECTOR_H

#include "viame_object_detectors_export.h"

#include <viame/algorithm_framework/algo/image_object_detector.h>

namespace kwiver {

namespace arrows {

namespace ocv {

class VIAME_OBJECT_DETECTORS_EXPORT hough_circle_detector
  : public vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    hough_circle_detector,
    "Hough circle detector",

    PARAM_DEFAULT(
      dp, double,
      "Inverse ratio of the accumulator resolution to the image resolution. "
      "For example, if dp=1 , the accumulator has the same resolution as the input image. "
      "If dp=2 , the accumulator has half as big width and height.", 1 ),

    PARAM_DEFAULT(
      min_dist, double,
      "Minimum distance between the centers of the detected circles. "
      "If the parameter is too small, multiple neighbor circles may be falsely "
      "detected in addition to a true one. If it is too large, some circles may be missed.",
      100 ),

    PARAM_DEFAULT(
      param1, double,
      "First method-specific parameter. In case of CV_HOUGH_GRADIENT , "
      "it is the higher threshold of the two passed to the Canny() edge detector "
      "(the lower one is twice smaller).", 200 ),

    PARAM_DEFAULT(
      param2, double,
      "Second method-specific parameter. In case of CV_HOUGH_GRADIENT , "
      "it is the accumulator threshold for the circle centers at the detection stage. "
      "The smaller it is, the more false circles may be detected. Circles, "
      "corresponding to the larger accumulator values, will be returned first.",
      100 ),

    PARAM_DEFAULT(
      min_radius, int,
      "Minimum circle radius.", 0 ),

    PARAM_DEFAULT( max_radius, int, "Maximum circle radius.", 0 )
  );

  virtual ~hough_circle_detector() = default;

  bool check_configuration( vital::config_block_sptr config ) const override;

  // Main detection method
  vital::detected_object_set_sptr detect(
    vital::image_container_sptr image_data ) const override;
};

} // namespace ocv

} // namespace arrows

}     // end namespace

#endif // ARROWS_OCV_HOUGH_CIRCLE_DETECTOR_H
