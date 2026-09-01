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

#include <vector>

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

  m_calibrator.set_logger( logger() );

  // Load calibration if specified
  load_calibration();

  // Create stereo matchers
  create_matchers();
}


// ---------------------------------------------------------------------------------------
bool
compute_stereo_disparity
::check_configuration( kv::config_block_sptr config ) const
{
  std::string algorithm = config->get_value< std::string >( "algorithm", c_algorithm );
  if( algorithm != "BM" && algorithm != "SGBM" )
  {
    LOG_ERROR( logger(), "Invalid algorithm: " << algorithm << ". Must be 'BM' or 'SGBM'." );
    return false;
  }

  std::string output_format =
    config->get_value< std::string >( "output_format", c_output_format );
  if( output_format != "raw" && output_format != "float32" && output_format != "uint16_scaled" )
  {
    LOG_ERROR( logger(), "Invalid output_format: " << output_format
               << ". Must be 'raw', 'float32', or 'uint16_scaled'." );
    return false;
  }

  return true;
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

  if( !m_calibrator.load_calibration( c_calibration_file, m_calibration ) )
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

  if( !calibrate_stereo_cameras::ensure_rectification( m_calibration, img_size ) )
  {
    VITAL_THROW( kv::invalid_data,
      "Calibration lacks the rectification transforms needed for " +
      std::to_string( img_size.width ) + "x" +
      std::to_string( img_size.height ) + " images: " + c_calibration_file );
  }

  cv::initUndistortRectifyMap(
    m_calibration.left.camera_matrix, m_calibration.left.dist_coeffs,
    m_calibration.R1, m_calibration.P1,
    img_size, CV_32FC1,
    m_rectification_map_left_x, m_rectification_map_left_y );

  cv::initUndistortRectifyMap(
    m_calibration.right.camera_matrix, m_calibration.right.dist_coeffs,
    m_calibration.R2, m_calibration.P2,
    img_size, CV_32FC1,
    m_rectification_map_right_x, m_rectification_map_right_y );

  cv::Mat original_grid( img_size.height * img_size.width, 1, CV_32FC2 );
  for( int y = 0; y < img_size.height; y++ )
  {
    for( int x = 0; x < img_size.width; x++ )
    {
      original_grid.at< cv::Vec2f >( y * img_size.width + x, 0 ) = cv::Vec2f( x, y );
    }
  }

  cv::Mat rectified_grid;
  cv::undistortPoints( original_grid, rectified_grid,
                       m_calibration.left.camera_matrix,
                       m_calibration.left.dist_coeffs,
                       m_calibration.R1,
                       m_calibration.P1 );

  rectified_grid = rectified_grid.reshape( 2, img_size.height );

  cv::Mat maps[2];
  cv::split( rectified_grid, maps );
  m_unrectification_map_x = maps[0];
  m_unrectification_map_y = maps[1];

  m_rectification_computed = true;
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
  cv::Mat left_color_rect;  // For export_as_alpha mode
  if( m_rectify_images )
  {
    compute_rectification_maps( left_gray.size() );
    cv::remap( left_gray, left_rect, m_rectification_map_left_x,
               m_rectification_map_left_y, cv::INTER_LINEAR );
    cv::remap( right_gray, right_rect, m_rectification_map_right_x,
               m_rectification_map_right_y, cv::INTER_LINEAR );

    // Also rectify color image if we need it for alpha channel output
    if( c_export_as_alpha )
    {
      cv::remap( ocv_left, left_color_rect, m_rectification_map_left_x,
                 m_rectification_map_left_y, cv::INTER_LINEAR );
    }
  }
  else
  {
    left_rect = left_gray;
    right_rect = right_gray;
    if( c_export_as_alpha )
    {
      left_color_rect = ocv_left;
    }
  }

  cv::Mat left_disparity_raw;
  m_left_matcher->compute( left_rect, right_rect, left_disparity_raw );

  if( c_use_wls_filter && m_right_matcher && m_wls_filter )
  {
    cv::Mat right_disparity_raw, filtered_disparity;
    m_right_matcher->compute( right_rect, left_rect, right_disparity_raw );

    m_wls_filter->filter( left_disparity_raw, left_rect, filtered_disparity,
                          right_disparity_raw, cv::Rect(), right_rect );

    left_disparity_raw = filtered_disparity;
  }

  cv::Mat float_map;
  left_disparity_raw.convertTo( float_map, CV_32F, 1.0 / 16.0 );
  float_map.setTo( 0, float_map < 0 );

  if( c_compute_depth )
  {
    if( m_calibration.P1.empty() )
    {
      VITAL_THROW( kv::invalid_data, "Cannot compute depth: calibration data missing." );
    }

    double fx = m_calibration.P1.at<double>( 0, 0 );
    double baseline =
      -m_calibration.P2.at<double>( 0, 3 ) / m_calibration.P2.at<double>( 0, 0 );
    double depth_scale = fx * baseline;

    cv::Mat depth_map = cv::Mat::zeros( float_map.size(), CV_32F );
    cv::Mat mask = float_map > 0;
    cv::divide( depth_scale, float_map, depth_map );
    depth_map.setTo( 0, ~mask );

    float_map = depth_map;
  }

  cv::Mat aligned_map;
  if( m_rectify_images && !c_output_rectified )
  {
    cv::remap( float_map, aligned_map,
               m_unrectification_map_x, m_unrectification_map_y,
               cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar( 0 ) );
  }
  else
  {
    aligned_map = float_map;
  }

  cv::Mat formatted_map;
  if( c_output_format == "float32" )
  {
    formatted_map = aligned_map;
  }
  else if( c_output_format == "uint16_scaled" )
  {
    aligned_map.convertTo( formatted_map, CV_16U, c_uint16_scale_factor );
  }
  else // "raw"
  {
    if( c_compute_depth )
    {
      formatted_map = aligned_map;
    }
    else
    {
      aligned_map.convertTo( formatted_map, CV_16S, 16.0 );
    }
  }

  cv::Mat output;
  if( c_export_as_alpha )
  {
    cv::Mat color_base;
    if( m_rectify_images && c_output_rectified )
    {
      color_base = left_color_rect;
    }
    else
    {
      color_base = ocv_left;
    }

    cv::Mat color_converted;
    if( formatted_map.depth() == CV_32F )
    {
      color_base.convertTo( color_converted, CV_32F, 1.0 / 255.0 );
    }
    else if( formatted_map.depth() == CV_16U )
    {
      color_base.convertTo( color_converted, CV_16U, 257.0 );
    }
    else
    {
      color_base.convertTo( color_converted, formatted_map.depth() );
    }

    cv::cvtColor( color_converted, output, cv::COLOR_BGR2BGRA );
    std::vector< cv::Mat > channels;
    cv::split( output, channels );
    channels[3] = formatted_map;
    cv::merge( channels, output );
  }
  else
  {
    output = formatted_map;
  }

  if( output.channels() == 1 )
  {
    return kv::image_container_sptr(
      new kwiver::arrows::ocv::image_container(
        output, kwiver::arrows::ocv::image_container::OTHER_COLOR ) );
  }

  return kv::image_container_sptr(
    new kwiver::arrows::ocv::image_container(
      output, kwiver::arrows::ocv::image_container::BGR_COLOR ) );
}

} //end namespace viame
