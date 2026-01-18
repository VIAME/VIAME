/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief OpenCV stereo disparity map computation algorithm
 *
 * This algorithm computes stereo disparity maps using OpenCV's
 * StereoBM or StereoSGBM algorithms. It supports:
 *   - Pre-rectified images (default) or internal rectification with calibration
 *   - BM (Block Matching) or SGBM (Semi-Global Block Matching) algorithms
 *   - Optional WLS (Weighted Least Squares) disparity filtering
 *   - Various output formats (raw disparity, scaled uint16, float32)
 */

#ifndef VIAME_OPENCV_COMPUTE_STEREO_DISPARITY_H
#define VIAME_OPENCV_COMPUTE_STEREO_DISPARITY_H

#include "viame_opencv_export.h"

#include <vital/algo/compute_stereo_depth_map.h>
#include "viame_algorithm_plugin_interface.h"

namespace viame {

class VIAME_OPENCV_EXPORT compute_stereo_disparity
  : public kwiver::vital::algo::compute_stereo_depth_map
{
public:
  VIAME_ALGORITHM_PLUGIN_INTERFACE( compute_stereo_disparity )
  PLUGIN_INFO( "ocv_stereo_disparity",
               "OpenCV stereo disparity map computation using BM or SGBM" )

  compute_stereo_disparity();
  virtual ~compute_stereo_disparity();

  virtual kwiver::vital::config_block_sptr get_configuration() const;

  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Compute stereo disparity map from left and right images
  ///
  /// \param left_image Left stereo image (grayscale or color)
  /// \param right_image Right stereo image (grayscale or color)
  /// \returns Disparity map image. Format depends on output_format config:
  ///          - "raw": CV_16S with disparity * 16 (OpenCV native format)
  ///          - "float32": CV_32F with disparity in pixels
  ///          - "uint16_scaled": CV_16U with disparity * 256 (for external algorithms)
  virtual kwiver::vital::image_container_sptr
  compute( kwiver::vital::image_container_sptr left_image,
           kwiver::vital::image_container_sptr right_image ) const;

private:

  class priv;
  const std::unique_ptr< priv > d;
};

}

#endif // VIAME_OPENCV_COMPUTE_STEREO_DISPARITY_H
