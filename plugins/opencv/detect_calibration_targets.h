#ifndef VIAME_OPENCV_DETECT_CALIBRATION_TARGETS_H
#define VIAME_OPENCV_DETECT_CALIBRATION_TARGETS_H

#include "viame_opencv_export.h"

#include <vital/algo/image_object_detector.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include "calibrate_stereo_cameras.h"

#include <opencv2/core/core.hpp>

namespace viame {

class VIAME_OPENCV_EXPORT detect_calibration_targets
  : public kwiver::vital::algo::image_object_detector
{
public:
  PLUGGABLE_IMPL( detect_calibration_targets,
                  image_object_detector,
                  "ocv_detect_calibration_targets",
                  "Detects checkerboard corners in input images for camera calibration processes.",
    PARAM_DEFAULT( config_file, std::string,
                   "Name of OCV Target Detector configuration file.", "" )
    PARAM_DEFAULT( target_width, unsigned,
                   "Number of width corners of the detected ocv target", 7 )
    PARAM_DEFAULT( target_height, unsigned,
                   "Number of height corners of the detected ocv target", 5 )
    PARAM_DEFAULT( square_size, float,
                   "Square size of the detected ocv target", 1.0 )
    PARAM_DEFAULT( object_type, std::string,
                   "The detected object type", "unknown" )
    PARAM_DEFAULT( auto_detect_grid, bool,
                   "Automatically detect grid size from the first image", false )
  )

  virtual ~detect_calibration_targets() = default;

  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:
  // Runtime state for auto-detection
  mutable bool m_grid_detected = false;
  mutable cv::Size m_detected_grid_size;

  // Calibration utility
  calibrate_stereo_cameras m_calibrator;
};

} // end namespace

#endif /* VIAME_OPENCV_DETECT_CALIBRATION_TARGETS_H */
