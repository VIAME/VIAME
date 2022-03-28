#ifndef VIAME_ocv_target_detector_H
#define VIAME_ocv_target_detector_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/image_object_detector.h>

namespace viame {

class VIAME_OPENCV_EXPORT ocv_target_detector :
  public kwiver::vital::algorithm_impl<
    ocv_target_detector, kwiver::vital::algo::image_object_detector >
{
public:
  ocv_target_detector();
  virtual ~ocv_target_detector();

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

#endif /* VIAME_ocv_target_detector_H */
