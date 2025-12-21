/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

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

#ifndef VIAME_OPENCV_STEREO_DISPARITY_MAP_H
#define VIAME_OPENCV_STEREO_DISPARITY_MAP_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/compute_stereo_depth_map.h>

namespace viame {

class VIAME_OPENCV_EXPORT stereo_disparity_map
  : public kwiver::vital::algo::compute_stereo_depth_map
{
public:
  PLUGIN_INFO( "ocv",
               "OpenCV stereo disparity map computation using BM or SGBM" )

  stereo_disparity_map();
  virtual ~stereo_disparity_map();

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

#endif // VIAME_OPENCV_STEREO_DISPARITY_MAP_H
