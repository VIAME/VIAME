/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "compute_stereo_disparity.h"

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

// ---------------------------------------------------------------------------------------
void
compute_stereo_disparity
::post_set_configuration()
{
  // Ensure num_disparities is divisible by 16
  if( c_num_disparities % 16 != 0 )
  {
    c_num_disparities = ( ( c_num_disparities / 16 ) + 1 ) * 16;
    LOG_WARN( logger(), "num_disparities adjusted to " << c_num_disparities
              << " (must be divisible by 16)" );
  }

  // Ensure block_size is odd
  if( c_block_size % 2 == 0 )
  {
    c_block_size++;
    LOG_WARN( logger(), "block_size adjusted to " << c_block_size << " (must be odd)" );
  }

  // Load calibration if specified
  load_calibration();

  // Create stereo matchers
  create_matchers();
}


// ---------------------------------------------------------------------------------------
void
compute_stereo_disparity
::create_matchers()
{
  if( c_algorithm == "BM" )
  {
    m_left_matcher = cv::StereoBM::create( c_num_disparities, c_sad_window_size );
    m_left_matcher->setMinDisparity( c_min_disparity );
    m_left_matcher->setSpeckleWindowSize( c_speckle_window_size );
    m_left_matcher->setSpeckleRange( c_speckle_range );
  }
  else if( c_algorithm == "SGBM" )
  {
    int p1 = 8 * c_block_size * c_block_size;
    int p2 = 32 * c_block_size * c_block_size;
    m_left_matcher = cv::StereoSGBM::create(
      c_min_disparity, c_num_disparities, c_block_size,
      p1, p2,
      1,    // disp12MaxDiff
      0,    // preFilterCap
      10,   // uniquenessRatio
      c_speckle_window_size,
      c_speckle_range,
      cv::StereoSGBM::MODE_SGBM_3WAY );
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type: " + c_algorithm );
  }

  if( c_use_wls_filter )
  {
    m_wls_filter = cv::ximgproc::createDisparityWLSFilter( m_left_matcher );
    m_wls_filter->setLambda( c_wls_lambda );
    m_wls_filter->setSigmaColor( c_wls_sigma );
    m_right_matcher = cv::ximgproc::createRightMatcher( m_left_matcher );
  }
  else
  {
    m_wls_filter.release();
    m_right_matcher.release();
  }
}


// ---------------------------------------------------------------------------------------
void
compute_stereo_disparity
::load_calibration()
{
  if( c_calibration_file.empty() )
  {
    m_rectify_images = false;
    return;
  }

  if( !m_calibrator.load_calibration_opencv( c_calibration_file, m_calibration ) )
  {
    VITAL_THROW( kv::invalid_data,
      "Failed to load calibration from: " + c_calibration_file );
  }
  m_rectify_images = true;
  m_rectification_computed = false;
}


// ---------------------------------------------------------------------------------------
void
compute_stereo_disparity
::compute_rectification_maps( const cv::Size& img_size ) const
{
  if( m_rectification_computed )
  {
    return;
  }

  cv::initUndistortRectifyMap(
    m_calibration.left.camera_matrix, m_calibration.left.dist_coeffs,
    m_calibration.R1, m_calibration.P1,
    img_size, CV_16SC2,
    m_rectification_map_left_x, m_rectification_map_left_y );

  cv::initUndistortRectifyMap(
    m_calibration.right.camera_matrix, m_calibration.right.dist_coeffs,
    m_calibration.R2, m_calibration.P2,
    img_size, CV_16SC2,
    m_rectification_map_right_x, m_rectification_map_right_y );

  m_rectification_computed = true;
}


// ---------------------------------------------------------------------------------------
bool compute_stereo_disparity
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
kv::image_container_sptr compute_stereo_disparity
::compute( kv::image_container_sptr left_image,
           kv::image_container_sptr right_image ) const
{
  if( !left_image || !right_image )
  {
    LOG_WARN( logger(), "Null input image(s)" );
    return kv::image_container_sptr();
  }

  if( left_image->get_image().size() != right_image->get_image().size() )
  {
    LOG_WARN( logger(), "Inconsistent left/right image sizes" );
    return kv::image_container_sptr();
  }

  // Convert to OpenCV format
  cv::Mat ocv_left = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv_right = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Convert to grayscale
  cv::Mat left_gray = calibrate_stereo_cameras::to_grayscale( ocv_left );
  cv::Mat right_gray = calibrate_stereo_cameras::to_grayscale( ocv_right );

  // Rectify if calibration is loaded
  cv::Mat left_rect, right_rect;
  cv::Mat left_color_rect;  // For disparity_as_alpha mode
  if( m_rectify_images )
  {
    compute_rectification_maps( left_gray.size() );
    cv::remap( left_gray, left_rect, m_rectification_map_left_x,
               m_rectification_map_left_y, cv::INTER_LINEAR );
    cv::remap( right_gray, right_rect, m_rectification_map_right_x,
               m_rectification_map_right_y, cv::INTER_LINEAR );

    // Also rectify color image if we need it for alpha channel output
    if( c_disparity_as_alpha )
    {
      cv::remap( ocv_left, left_color_rect, m_rectification_map_left_x,
                 m_rectification_map_left_y, cv::INTER_LINEAR );
    }
  }
  else
  {
    left_rect = left_gray;
    right_rect = right_gray;
    if( c_disparity_as_alpha )
    {
      left_color_rect = ocv_left;
    }
  }

  // Compute disparity
  cv::Mat left_disparity;
  m_left_matcher->compute( left_rect, right_rect, left_disparity );

  // Apply WLS filter if enabled
  if( c_use_wls_filter && m_right_matcher && m_wls_filter )
  {
    cv::Mat right_disparity, filtered_disparity;
    m_right_matcher->compute( right_rect, left_rect, right_disparity );
    m_wls_filter->filter( left_disparity, left_rect, filtered_disparity,
                           right_disparity, cv::Rect(), right_rect );
    left_disparity = filtered_disparity;
  }

  // Convert to requested output format
  cv::Mat output;
  if( c_output_format == "raw" )
  {
    // Raw OpenCV format: CV_16S with disparity * 16
    output = left_disparity;
  }
  else if( c_output_format == "float32" )
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
  if( c_disparity_as_alpha && !left_color_rect.empty() )
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
      LOG_WARN( logger(), "Unexpected number of channels in left image" );
      left_bgra = left_color_rect;
    }

    // Convert disparity to 8-bit for alpha channel
    cv::Mat disp_8bit;
    cv::Mat float_disp;
    left_disparity.convertTo( float_disp, CV_32F, 1.0 / 16.0 );
    float_disp.setTo( 0, float_disp < 0 );
    float_disp.convertTo( disp_8bit, CV_8U );

    // Invert if requested
    if( c_invert_disparity_alpha )
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
