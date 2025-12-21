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

#include "ocv_stereo_disparity_map.h"
#include "ocv_stereo_calibration.h"

#include <vital/vital_config.h>
#include <vital/types/image_container.h>
#include <vital/exceptions.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ximgproc.hpp>

#include <arrows/ocv/image_container.h>

namespace kv = kwiver::vital;

namespace viame {

class ocv_stereo_disparity_map::priv
{
public:
  // Algorithm selection
  std::string algorithm{ "SGBM" };

  // SGBM/BM parameters
  int min_disparity{ 0 };
  int num_disparities{ 128 };
  int sad_window_size{ 21 };
  int block_size{ 5 };
  int speckle_window_size{ 100 };
  int speckle_range{ 32 };

  // Output format: "raw", "float32", or "uint16_scaled"
  std::string output_format{ "uint16_scaled" };

  // Alpha channel output options
  bool disparity_as_alpha{ false };
  bool invert_disparity_alpha{ false };

  // WLS filtering
  bool use_wls_filter{ false };
  double wls_lambda{ 8000.0 };
  double wls_sigma{ 1.5 };

  // Rectification (optional - if calibration_file is set)
  std::string calibration_file;
  bool rectify_images{ false };
  mutable bool rectification_computed{ false };
  mutable cv::Mat rectification_map_left_x;
  mutable cv::Mat rectification_map_left_y;
  mutable cv::Mat rectification_map_right_x;
  mutable cv::Mat rectification_map_right_y;

  // Calibration data (loaded if calibration_file is set)
  stereo_calibration_result calibration;
  stereo_calibration calibrator;

  // Stereo matchers
  cv::Ptr<cv::StereoMatcher> left_matcher;
  cv::Ptr<cv::StereoMatcher> right_matcher;
  cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

  kv::logger_handle_t logger;

  void create_matchers()
  {
    if( algorithm == "BM" )
    {
      left_matcher = cv::StereoBM::create( num_disparities, sad_window_size );
      left_matcher->setMinDisparity( min_disparity );
      left_matcher->setSpeckleWindowSize( speckle_window_size );
      left_matcher->setSpeckleRange( speckle_range );
    }
    else if( algorithm == "SGBM" )
    {
      int p1 = 8 * block_size * block_size;
      int p2 = 32 * block_size * block_size;
      left_matcher = cv::StereoSGBM::create(
        min_disparity, num_disparities, block_size,
        p1, p2,
        1,    // disp12MaxDiff
        0,    // preFilterCap
        10,   // uniquenessRatio
        speckle_window_size,
        speckle_range,
        cv::StereoSGBM::MODE_SGBM_3WAY );
    }
    else
    {
      throw std::runtime_error( "Invalid algorithm type: " + algorithm );
    }

    if( use_wls_filter )
    {
      wls_filter = cv::ximgproc::createDisparityWLSFilter( left_matcher );
      wls_filter->setLambda( wls_lambda );
      wls_filter->setSigmaColor( wls_sigma );
      right_matcher = cv::ximgproc::createRightMatcher( left_matcher );
    }
    else
    {
      wls_filter.release();
      right_matcher.release();
    }
  }

  void load_calibration()
  {
    if( calibration_file.empty() )
    {
      rectify_images = false;
      return;
    }

    if( !calibrator.load_calibration_opencv( calibration_file, calibration ) )
    {
      VITAL_THROW( kv::invalid_data,
        "Failed to load calibration from: " + calibration_file );
    }
    rectify_images = true;
    rectification_computed = false;
  }

  void compute_rectification_maps( const cv::Size& img_size ) const
  {
    if( rectification_computed )
    {
      return;
    }

    cv::initUndistortRectifyMap(
      calibration.left.camera_matrix, calibration.left.dist_coeffs,
      calibration.R1, calibration.P1,
      img_size, CV_16SC2,
      rectification_map_left_x, rectification_map_left_y );

    cv::initUndistortRectifyMap(
      calibration.right.camera_matrix, calibration.right.dist_coeffs,
      calibration.R2, calibration.P2,
      img_size, CV_16SC2,
      rectification_map_right_x, rectification_map_right_y );

    rectification_computed = true;
  }
};


ocv_stereo_disparity_map::ocv_stereo_disparity_map()
: d( new priv() )
{
  attach_logger( "viame.opencv.ocv_stereo_disparity_map" );
  d->logger = logger();
  d->calibrator.set_logger( d->logger );
}


ocv_stereo_disparity_map::~ocv_stereo_disparity_map()
{
}


// ---------------------------------------------------------------------------------------
kv::config_block_sptr
ocv_stereo_disparity_map
::get_configuration() const
{
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "algorithm", d->algorithm,
    "Stereo matching algorithm: 'BM' (Block Matching) or 'SGBM' (Semi-Global Block Matching). "
    "SGBM generally produces better results but is slower." );

  config->set_value( "min_disparity", d->min_disparity,
    "Minimum possible disparity value. Normally 0, but can be negative for "
    "cameras with convergent optical axes." );

  config->set_value( "num_disparities", d->num_disparities,
    "Maximum disparity minus minimum disparity. Must be divisible by 16. "
    "Larger values allow matching objects closer to the camera." );

  config->set_value( "sad_window_size", d->sad_window_size,
    "SAD (Sum of Absolute Differences) window size for BM algorithm. Must be odd, typically 5-21." );

  config->set_value( "block_size", d->block_size,
    "Block size for SGBM algorithm. Must be odd, typically 3-11." );

  config->set_value( "speckle_window_size", d->speckle_window_size,
    "Maximum size of smooth disparity regions to consider for speckle filtering. "
    "Set to 0 to disable speckle filtering." );

  config->set_value( "speckle_range", d->speckle_range,
    "Maximum disparity variation within each connected component for speckle filtering." );

  config->set_value( "output_format", d->output_format,
    "Output disparity format: "
    "'raw' (CV_16S with disparity * 16, OpenCV native), "
    "'float32' (CV_32F with disparity in pixels), "
    "'uint16_scaled' (CV_16U with disparity * 256, compatible with external algorithms)." );

  config->set_value( "disparity_as_alpha", d->disparity_as_alpha,
    "If true, returns the rectified left image with disparity as the alpha (4th) channel. "
    "The output will be a 4-channel BGRA image where the alpha channel contains the 8-bit disparity." );

  config->set_value( "invert_disparity_alpha", d->invert_disparity_alpha,
    "If true and disparity_as_alpha is enabled, inverts the disparity values in the alpha channel. "
    "Invalid (zero) disparity pixels are set to white before inversion." );

  config->set_value( "use_wls_filter", d->use_wls_filter,
    "Apply Weighted Least Squares (WLS) filtering to smooth the disparity map while "
    "preserving edges. Requires computing disparity for both left and right images." );

  config->set_value( "wls_lambda", d->wls_lambda,
    "WLS filter regularization parameter. Larger values produce smoother disparity maps." );

  config->set_value( "wls_sigma", d->wls_sigma,
    "WLS filter sigma parameter for color similarity weighting." );

  config->set_value( "calibration_file", d->calibration_file,
    "Path to stereo calibration file (OpenCV YAML/XML format). If specified, images will be "
    "rectified before computing disparity. Leave empty if input images are already rectified." );

  return config;
}

// ---------------------------------------------------------------------------------------
void ocv_stereo_disparity_map
::set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->algorithm = config->get_value< std::string >( "algorithm" );
  d->min_disparity = config->get_value< int >( "min_disparity" );
  d->num_disparities = config->get_value< int >( "num_disparities" );
  d->sad_window_size = config->get_value< int >( "sad_window_size" );
  d->block_size = config->get_value< int >( "block_size" );
  d->speckle_window_size = config->get_value< int >( "speckle_window_size" );
  d->speckle_range = config->get_value< int >( "speckle_range" );
  d->output_format = config->get_value< std::string >( "output_format" );
  d->disparity_as_alpha = config->get_value< bool >( "disparity_as_alpha" );
  d->invert_disparity_alpha = config->get_value< bool >( "invert_disparity_alpha" );
  d->use_wls_filter = config->get_value< bool >( "use_wls_filter" );
  d->wls_lambda = config->get_value< double >( "wls_lambda" );
  d->wls_sigma = config->get_value< double >( "wls_sigma" );
  d->calibration_file = config->get_value< std::string >( "calibration_file" );

  // Ensure num_disparities is divisible by 16
  if( d->num_disparities % 16 != 0 )
  {
    d->num_disparities = ( ( d->num_disparities / 16 ) + 1 ) * 16;
    LOG_WARN( d->logger, "num_disparities adjusted to " << d->num_disparities
              << " (must be divisible by 16)" );
  }

  // Ensure block_size is odd
  if( d->block_size % 2 == 0 )
  {
    d->block_size++;
    LOG_WARN( d->logger, "block_size adjusted to " << d->block_size << " (must be odd)" );
  }

  // Load calibration if specified
  d->load_calibration();

  // Create stereo matchers
  d->create_matchers();
}


// ---------------------------------------------------------------------------------------
bool ocv_stereo_disparity_map
::check_configuration( kv::config_block_sptr config ) const
{
  std::string algorithm = config->get_value< std::string >( "algorithm" );
  if( algorithm != "BM" && algorithm != "SGBM" )
  {
    LOG_ERROR( logger(), "Invalid algorithm: " << algorithm << ". Must be 'BM' or 'SGBM'." );
    return false;
  }

  std::string output_format = config->get_value< std::string >( "output_format" );
  if( output_format != "raw" && output_format != "float32" && output_format != "uint16_scaled" )
  {
    LOG_ERROR( logger(), "Invalid output_format: " << output_format
               << ". Must be 'raw', 'float32', or 'uint16_scaled'." );
    return false;
  }

  return true;
}


// ---------------------------------------------------------------------------------------
kv::image_container_sptr ocv_stereo_disparity_map
::compute( kv::image_container_sptr left_image,
           kv::image_container_sptr right_image ) const
{
  if( !left_image || !right_image )
  {
    LOG_WARN( d->logger, "Null input image(s)" );
    return kv::image_container_sptr();
  }

  if( left_image->get_image().size() != right_image->get_image().size() )
  {
    LOG_WARN( d->logger, "Inconsistent left/right image sizes" );
    return kv::image_container_sptr();
  }

  // Convert to OpenCV format
  cv::Mat ocv_left = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv_right = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Convert to grayscale
  cv::Mat left_gray = stereo_calibration::to_grayscale( ocv_left );
  cv::Mat right_gray = stereo_calibration::to_grayscale( ocv_right );

  // Rectify if calibration is loaded
  cv::Mat left_rect, right_rect;
  cv::Mat left_color_rect;  // For disparity_as_alpha mode
  if( d->rectify_images )
  {
    d->compute_rectification_maps( left_gray.size() );
    cv::remap( left_gray, left_rect, d->rectification_map_left_x,
               d->rectification_map_left_y, cv::INTER_LINEAR );
    cv::remap( right_gray, right_rect, d->rectification_map_right_x,
               d->rectification_map_right_y, cv::INTER_LINEAR );

    // Also rectify color image if we need it for alpha channel output
    if( d->disparity_as_alpha )
    {
      cv::remap( ocv_left, left_color_rect, d->rectification_map_left_x,
                 d->rectification_map_left_y, cv::INTER_LINEAR );
    }
  }
  else
  {
    left_rect = left_gray;
    right_rect = right_gray;
    if( d->disparity_as_alpha )
    {
      left_color_rect = ocv_left;
    }
  }

  // Compute disparity
  cv::Mat left_disparity;
  d->left_matcher->compute( left_rect, right_rect, left_disparity );

  // Apply WLS filter if enabled
  if( d->use_wls_filter && d->right_matcher && d->wls_filter )
  {
    cv::Mat right_disparity, filtered_disparity;
    d->right_matcher->compute( right_rect, left_rect, right_disparity );
    d->wls_filter->filter( left_disparity, left_rect, filtered_disparity,
                           right_disparity, cv::Rect(), right_rect );
    left_disparity = filtered_disparity;
  }

  // Convert to requested output format
  cv::Mat output;
  if( d->output_format == "raw" )
  {
    // Raw OpenCV format: CV_16S with disparity * 16
    output = left_disparity;
  }
  else if( d->output_format == "float32" )
  {
    // Float format: CV_32F with disparity in pixels
    left_disparity.convertTo( output, CV_32F, 1.0 / 16.0 );
  }
  else // uint16_scaled
  {
    // Scaled uint16: CV_16U with disparity * 256
    // Convert from fixed-point (*16) to scaled (*256) = multiply by 16
    cv::Mat float_disp;
    left_disparity.convertTo( float_disp, CV_32F, 1.0 / 16.0 );

    // Clamp negative values and scale
    cv::Mat scaled;
    float_disp.setTo( 0, float_disp < 0 );
    float_disp.convertTo( scaled, CV_32F, 256.0 );
    scaled.setTo( 65535, scaled > 65535 );
    scaled.convertTo( output, CV_16U );
  }

  // If disparity_as_alpha is enabled, combine with left color image
  if( d->disparity_as_alpha && !left_color_rect.empty() )
  {
    // Convert left image to BGRA
    cv::Mat left_bgra;
    if( left_color_rect.channels() == 1 )
    {
      cv::cvtColor( left_color_rect, left_bgra, cv::COLOR_GRAY2BGRA );
    }
    else if( left_color_rect.channels() == 3 )
    {
      cv::cvtColor( left_color_rect, left_bgra, cv::COLOR_BGR2BGRA );
    }
    else if( left_color_rect.channels() == 4 )
    {
      left_bgra = left_color_rect;
    }
    else
    {
      LOG_WARN( d->logger, "Unexpected number of channels in left image" );
      left_bgra = left_color_rect;
    }

    // Convert disparity to 8-bit for alpha channel
    cv::Mat disp_8bit;
    cv::Mat float_disp;
    left_disparity.convertTo( float_disp, CV_32F, 1.0 / 16.0 );
    float_disp.setTo( 0, float_disp < 0 );
    float_disp.convertTo( disp_8bit, CV_8U );

    // Invert if requested
    if( d->invert_disparity_alpha )
    {
      // Set zero (invalid) pixels to white before inversion
      cv::Mat mask;
      cv::inRange( disp_8bit, cv::Scalar( 0 ), cv::Scalar( 0 ), mask );
      disp_8bit.setTo( cv::Scalar( 255 ), mask );

      // Invert
      cv::bitwise_not( disp_8bit, disp_8bit );
    }

    // Set disparity as alpha channel
    std::vector<cv::Mat> channels( 4 );
    cv::split( left_bgra, channels );
    channels[3] = disp_8bit;
    cv::merge( channels, output );
  }

  return kv::image_container_sptr(
    new kwiver::arrows::ocv::image_container(
      output, kwiver::arrows::ocv::image_container::BGR_COLOR ) );
}

} //end namespace viame
