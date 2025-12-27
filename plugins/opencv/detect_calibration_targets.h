#ifndef VIAME_OPENCV_DETECT_CALIBRATION_TARGETS_H
#define VIAME_OPENCV_DETECT_CALIBRATION_TARGETS_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/image_object_detector.h>

namespace viame {

class VIAME_OPENCV_EXPORT detect_calibration_targets :
  public kwiver::vital::algorithm_impl<
    detect_calibration_targets, kwiver::vital::algo::image_object_detector >
{
public:
  PLUGIN_INFO( "ocv_detect_calibration_targets",
               "Detects checkerboard corners in input images for camera calibration processes." )

  detect_calibration_targets();
  virtual ~detect_calibration_targets();

  // Get the current configuration (parameters) for this detector
  virtual kwiver::vital::config_block_sptr get_configuration() const;

  // Set configurations automatically parsed from input pipeline and config files
  virtual void set_configuration( kwiver::vital::config_block_sptr config_in );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  // Main detection method
  virtual kwiver::vital::detected_object_set_sptr detect(
    kwiver::vital::image_container_sptr image_data ) const;

private:
  class priv;
  const std::unique_ptr< priv > d;
};

} // end namespace

#endif /* VIAME_OPENCV_DETECT_CALIBRATION_TARGETS_H */
