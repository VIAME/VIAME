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
 *   - Disparity or metric depth output
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
  PLUGGABLE_IMPL_NAMED(
    compute_stereo_disparity, "ocv_stereo_disparity",
    "OpenCV stereo disparity map computation using BM or SGBM",
    PARAM_DEFAULT( algorithm, std::string,
                   "Stereo matching algorithm: 'BM' (Block Matching) or 'SGBM' (Semi-Global Block Matching). "
                   "SGBM generally produces better results but is slower.", "SGBM" ),
    PARAM_DEFAULT( min_disparity, int,
                   "Minimum possible disparity value. Normally 0, but can be negative for "
                   "cameras with convergent optical axes.", 0 ),
    PARAM_DEFAULT( num_disparities, int,
                   "Maximum disparity minus minimum disparity. Must be divisible by 16. "
                   "Larger values allow matching objects closer to the camera.", 128 ),
    PARAM_DEFAULT( sad_window_size, int,
                   "SAD (Sum of Absolute Differences) window size for BM algorithm. Must be odd, typically 5-21.", 21 ),
    PARAM_DEFAULT( block_size, int,
                   "Block size for SGBM algorithm. Must be odd, typically 3-11.", 5 ),
    PARAM_DEFAULT( speckle_window_size, int,
                   "Maximum size of smooth disparity regions to consider for speckle filtering. "
                   "Set to 0 to disable speckle filtering.", 100 ),
    PARAM_DEFAULT( speckle_range, int,
                   "Maximum disparity variation within each connected component for speckle filtering.", 32 ),
    PARAM_DEFAULT( use_wls_filter, bool,
                   "Apply Weighted Least Squares (WLS) filtering to smooth the disparity map while "
                   "preserving edges. Requires computing disparity for both left and right images.", false ),
    PARAM_DEFAULT( wls_lambda, double,
                   "WLS filter regularization parameter. Larger values produce smoother disparity maps.", 8000.0 ),
    PARAM_DEFAULT( wls_sigma, double,
                   "WLS filter sigma parameter for color similarity weighting.", 1.5 ),
    PARAM_DEFAULT( calibration_file, std::string,
                   "Path to stereo calibration file (OpenCV YAML/XML format). If specified, images will be "
                   "rectified before computing disparity. Leave empty if input images are already rectified "
                   "(e.g., when called from measurement_utilities which handles its own rectification).", "" ),
    PARAM_DEFAULT( compute_depth, bool,
                   "If true, computes depth Z = (fx * baseline) / disparity. "
                   "If false, computes disparity. Depth requires a valid calibration file.", false ),
    PARAM_DEFAULT( export_as_alpha, bool,
                   "If true, outputs the original left color image with the computed map (depth/disparity) "
                   "in the 4th (Alpha) channel. If false, outputs a 1-channel image of the map.", false ),
    PARAM_DEFAULT( output_rectified, bool,
                   "If true, the output map (and color image if export_as_alpha is true) will be kept "
                   "in the rectified coordinate space. If false, they will be mapped back to the original image.", true ),
    PARAM_DEFAULT( output_format, std::string,
                   "Output format: 'float32' (best for TIFF), 'uint16_scaled' (best for 16-bit PNG), or 'raw'.", "raw" ),
    PARAM_DEFAULT( uint16_scale_factor, double,
                   "Multiplier used ONLY when output_format is 'uint16_scaled'. "
                   "e.g., if depth is in meters, a factor of 1000 converts it to millimeters "
                   "to save as integers in PNG.", 256.0 )
  )

  virtual ~compute_stereo_disparity() = default;

  virtual bool check_configuration(
    kwiver::vital::config_block_sptr config ) const override;

  virtual void post_set_configuration();

  /// Compute stereo disparity (or depth) map from left and right images
  ///
  /// \param left_image Left stereo image (grayscale or color)
  /// \param right_image Right stereo image (grayscale or color)
  /// \returns Disparity or depth map image. Format depends on output_format:
  ///          - "raw": CV_16S with disparity * 16 (OpenCV native format)
  ///          - "float32": CV_32F in pixels (disparity) or scene units (depth)
  ///          - "uint16_scaled": CV_16U scaled by uint16_scale_factor
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
  mutable cv::Mat m_unrectification_map_x;
  mutable cv::Mat m_unrectification_map_y;

  // Calibration data (loaded if calibration_file is set). Mutable because
  // single-file formats carry no rectification transforms, so those are filled
  // in lazily once the first image gives us an image size.
  mutable calibrate_stereo_cameras_result m_calibration;
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
