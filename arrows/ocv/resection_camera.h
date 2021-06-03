// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV resection_camera algorithm impl interface

#ifndef KWIVER_ARROWS_OCV_RESECTION_CAMERA_H_
#define KWIVER_ARROWS_OCV_RESECTION_CAMERA_H_

#include <arrows/ocv/kwiver_algo_ocv_export.h>

#include <vital/algo/resection_camera.h>

#include <vital/vital_config.h>

namespace kwiver {

namespace arrows {

namespace ocv {

/// Use OpenCV to estimate a camera's pose and projection matrix from 3D
/// feature and point projection pairs.
class KWIVER_ALGO_OCV_EXPORT resection_camera
  : public vital::algo::resection_camera
{
public:
  PLUGIN_INFO( "ocv",
               "resection camera using OpenCV calibrate camera method" )

  resection_camera();
  virtual ~resection_camera();

  /// Get this algorithm's \link vital::config_block configuration block
  /// \endlink.
  vital::config_block_sptr get_configuration() const override;

  /// Set this algorithm's properties via a config block.
  void set_configuration( vital::config_block_sptr config ) override;

  /// Check that the algorithm's configuration config_block is valid.
  bool check_configuration( vital::config_block_sptr config ) const override;

  kwiver::vital::camera_perspective_sptr
  resection(
    std::vector< kwiver::vital::vector_2d > const& image_points,
    std::vector< kwiver::vital::vector_3d > const& world_points,
    kwiver::vital::camera_intrinsics_sptr initial_calibration,
    std::vector< bool >* inliers ) const override;

  using vital::algo::resection_camera::resection;

private:
  class priv;

  std::unique_ptr< priv > const d_;
};

} // namespace ocv

} // namespace arrows

} // namespace kwiver

#endif
