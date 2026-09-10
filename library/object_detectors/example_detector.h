// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef KWIVER_EXAMPLE_DETECTOR_H
#define KWIVER_EXAMPLE_DETECTOR_H

#include "viame_object_detectors_export.h"

#include <viame/algorithm_framework/algo/image_object_detector.h>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/algo/algorithm.txx>

namespace kwiver {

namespace arrows {

namespace core {

class VIAME_OBJECT_DETECTORS_EXPORT example_detector
  : public vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL(
    example_detector,
    "Simple example detector that just creates a user-specified bounding box.",
    PARAM_DEFAULT(
      center_x, double,
      "Bounding box center x coordinate.",
      100.0 ),
    PARAM_DEFAULT(
      center_y, double,
      "Bounding box center y coordinate.",
      100.0 ),
    PARAM_DEFAULT(
      height, double,
      "Bounding box height.",
      200.0 ),
    PARAM_DEFAULT(
      width, double,
      "Bounding box width.",
      200.0 ),
    PARAM_DEFAULT(
      dx, double,
      "Bounding box x translation per frame.",
      0.0 ),
    PARAM_DEFAULT(
      dy, double,
      "Bounding box y translation per frame.",
      0.0 )
  )

  virtual ~example_detector();

  virtual bool check_configuration( vital::config_block_sptr config ) const;

  // Main detection method
  virtual vital::detected_object_set_sptr detect(
    vital::image_container_sptr image_data ) const;

private:
  void initialize() override;
  /// private implementation class
  class priv;
  KWIVER_UNIQUE_PTR( priv, d );
};

} // namespace core

} // namespace arrows

}     // end namespace

#endif // KWIVER_EXAMPLE_DETECTOR_H
