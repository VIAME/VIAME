
#ifndef VIAME_OPENCV_RECTIFY_STEREO_DEPTH_H
#define VIAME_OPENCV_RECTIFY_STEREO_DEPTH_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/compute_stereo_depth_map.h>

namespace viame {

class VIAME_OPENCV_EXPORT ocv_rectified_stereo_disparity_map :
  public kwiver::vital::algorithm_impl<
    ocv_rectified_stereo_disparity_map, kwiver::vital::algo::compute_stereo_depth_map >
{
public:
  PLUGIN_INFO( "ocv_rectified_stereo_disparity_map",
               "Rectifies and computes the stereo disparity of two input images" )

  ocv_rectified_stereo_disparity_map();
  virtual ~ocv_rectified_stereo_disparity_map();

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  virtual kwiver::vital::image_container_sptr
  compute( kwiver::vital::image_container_sptr left_image,
           kwiver::vital::image_container_sptr right_image ) const;

private:

  class priv;
  const std::unique_ptr< priv > d;
};

}

#endif // VIAME_OPENCV_RECTIFY_STEREO_DEPTH_H
