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
#include <vital/plugin_management/pluggable_macro_magic.h>

#include "calibrate_stereo_cameras.h"

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ximgproc.hpp>

namespace viame {

class VIAME_OPENCV_EXPORT compute_stereo_disparity
  : public kwiver::vital::algo::compute_stereo_depth_map
{
public:
  PLUGGABLE_IMPL( compute_stereo_disparity,
                  compute_stereo_depth_map,
                  "ocv_stereo_disparity",
                  "OpenCV stereo disparity map computation using BM or SGBM",
    PARAM_DEFAULT( algorithm, std::string,
                   "Stereo matching algorithm: 'BM' (Block Matching) or 'SGBM' (Semi-Global Block Matching). "
                   "SGBM generally produces better results but is slower.", "SGBM" )
    PARAM_DEFAULT( min_disparity, int,
                   "Minimum possible disparity value. Normally 0, but can be negative for "
                   "cameras with convergent optical axes.", 0 )
    PARAM_DEFAULT( num_disparities, int,
                   "Maximum disparity minus minimum disparity. Must be divisible by 16. "
                   "Larger values allow matching objects closer to the camera.", 128 )
    PARAM_DEFAULT( sad_window_size, int,
                   "SAD (Sum of Absolute Differences) window size for BM algorithm. Must be odd, typically 5-21.", 21 )
    PARAM_DEFAULT( block_size, int,
                   "Block size for SGBM algorithm. Must be odd, typically 3-11.", 5 )
    PARAM_DEFAULT( speckle_window_size, int,
                   "Maximum size of smooth disparity regions to consider for speckle filtering. "
                   "Set to 0 to disable speckle filtering.", 100 )
    PARAM_DEFAULT( speckle_range, int,
                   "Maximum disparity variation within each connected component for speckle filtering.", 32 )
    PARAM_DEFAULT( output_format, std::string,
                   "Output disparity format: "
                   "'raw' (CV_16S with disparity * 16, OpenCV native), "
                   "'float32' (CV_32F with disparity in pixels), "
                   "'uint16_scaled' (CV_16U with disparity * 256, compatible with external algorithms).", "raw" )
    PARAM_DEFAULT( disparity_as_alpha, bool,
                   "If true, returns the rectified left image with disparity as the alpha (4th) channel. "
                   "The output will be a 4-channel BGRA image where the alpha channel contains the 8-bit disparity.", false )
    PARAM_DEFAULT( invert_disparity_alpha, bool,
                   "If true and disparity_as_alpha is enabled, inverts the disparity values in the alpha channel. "
                   "Invalid (zero) disparity pixels are set to white before inversion.", false )
    PARAM_DEFAULT( use_wls_filter, bool,
                   "Apply Weighted Least Squares (WLS) filtering to smooth the disparity map while "
                   "preserving edges. Requires computing disparity for both left and right images.", false )
    PARAM_DEFAULT( wls_lambda, double,
                   "WLS filter regularization parameter. Larger values produce smoother disparity maps.", 8000.0 )
    PARAM_DEFAULT( wls_sigma, double,
                   "WLS filter sigma parameter for color similarity weighting.", 1.5 )
    PARAM_DEFAULT( calibration_file, std::string,
                   "Path to stereo calibration file (OpenCV YAML/XML format). If specified, images will be "
                   "rectified before computing disparity. Leave empty if input images are already rectified "
                   "(e.g., when called from measurement_utilities which handles its own rectification).", "" )
  )

  virtual ~compute_stereo_disparity() = default;

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
  // Rectification state (computed lazily from calibration)
  bool m_rectify_images{ false };
  mutable bool m_rectification_computed{ false };
  mutable cv::Mat m_rectification_map_left_x;
  mutable cv::Mat m_rectification_map_left_y;
  mutable cv::Mat m_rectification_map_right_x;
  mutable cv::Mat m_rectification_map_right_y;

  // Calibration data (loaded if calibration_file is set)
  calibrate_stereo_cameras_result m_calibration;
  calibrate_stereo_cameras m_calibrator;

  // Stereo matchers
  cv::Ptr<cv::StereoMatcher> m_left_matcher;
  cv::Ptr<cv::StereoMatcher> m_right_matcher;
  cv::Ptr<cv::ximgproc::DisparityWLSFilter> m_wls_filter;

  // Helper methods
  void create_matchers();
  void load_calibration();
  void compute_rectification_maps( const cv::Size& img_size ) const;
};

}

#endif // VIAME_OPENCV_COMPUTE_STEREO_DISPARITY_H
