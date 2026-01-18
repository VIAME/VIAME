/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "apply_color_correction.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>

#include <arrows/ocv/image_container.h>

#include <cmath>

namespace viame {

using namespace kwiver;

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::apply_gamma_correction( cv::Mat& image )
{
  double gamma = c_gamma;

  if( c_gamma_auto )
  {
    gamma = estimate_auto_gamma( image );
  }

  if( std::abs( gamma - 1.0 ) < 0.001 )
  {
    return; // No correction needed
  }

  // Build lookup table for efficiency
  cv::Mat lut( 1, 256, CV_8U );
  uchar* ptr = lut.ptr();

  double inv_gamma = 1.0 / gamma;
  for( int i = 0; i < 256; i++ )
  {
    ptr[i] = cv::saturate_cast< uchar >( std::pow( i / 255.0, inv_gamma ) * 255.0 );
  }

  // Convert to 8-bit if necessary for LUT
  cv::Mat temp;
  if( image.depth() != CV_8U )
  {
    cv::normalize( image, temp, 255, 0, cv::NORM_MINMAX );
    temp.convertTo( temp, CV_8U );
  }
  else
  {
    temp = image;
  }

  cv::LUT( temp, lut, image );
}

// -----------------------------------------------------------------------------------------------
double
apply_color_correction
::estimate_auto_gamma( const cv::Mat& image )
{
  cv::Mat gray;

  if( image.channels() == 3 )
  {
#if CV_MAJOR_VERSION < 4
    cv::cvtColor( image, gray, CV_BGR2GRAY );
#else
    cv::cvtColor( image, gray, cv::COLOR_BGR2GRAY );
#endif
  }
  else
  {
    gray = image;
  }

  // Convert to double for calculations
  cv::Mat gray_dbl;
  gray.convertTo( gray_dbl, CV_64F, 1.0 / 255.0 );

  double mean_val = cv::mean( gray_dbl )[0];

  // Target mean of 0.5 (middle gray)
  // gamma = log(0.5) / log(mean_val)
  if( mean_val > 0.001 && std::abs( mean_val - 0.5 ) > 0.01 )
  {
    double gamma = std::log( 0.5 ) / std::log( mean_val );
    // Clamp to reasonable range
    return std::max( 0.1, std::min( 5.0, gamma ) );
  }

  return 1.0;
}

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::apply_gray_world_balance( cv::Mat& image )
{
  if( image.channels() != 3 )
  {
    return; // Only works on color images
  }

  // Create mask to exclude saturated pixels
  cv::Mat gray;
#if CV_MAJOR_VERSION < 4
  cv::cvtColor( image, gray, CV_BGR2GRAY );
#else
  cv::cvtColor( image, gray, cv::COLOR_BGR2GRAY );
#endif

  double max_val = ( image.depth() == CV_8U ) ? 255.0 : 1.0;
  cv::Mat mask = gray < ( c_gray_world_sat_threshold * max_val );

  // Calculate mean of each channel (excluding saturated pixels)
  cv::Scalar channel_means = cv::mean( image, mask );

  // Calculate average illumination
  double avg_illumination = ( channel_means[0] + channel_means[1] + channel_means[2] ) / 3.0;

  if( avg_illumination < 0.001 )
  {
    return; // Avoid division by zero
  }

  // Calculate scaling factors
  double scale_b = avg_illumination / ( channel_means[0] + 0.001 );
  double scale_g = avg_illumination / ( channel_means[1] + 0.001 );
  double scale_r = avg_illumination / ( channel_means[2] + 0.001 );

  // Apply scaling
  std::vector< cv::Mat > channels( 3 );
  cv::split( image, channels );

  channels[0] = channels[0] * scale_b;
  channels[1] = channels[1] * scale_g;
  channels[2] = channels[2] * scale_r;

  cv::merge( channels, image );
}

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::load_depth_map()
{
  if( !c_depth_map_path.empty() && m_depth_map.empty() )
  {
    m_depth_map = cv::imread( c_depth_map_path, cv::IMREAD_ANYDEPTH | cv::IMREAD_GRAYSCALE );
    if( !m_depth_map.empty() )
    {
      // Normalize to 0-1 range
      cv::normalize( m_depth_map, m_depth_map, 0, 1, cv::NORM_MINMAX, CV_32F );
    }
  }
}

// -----------------------------------------------------------------------------------------------
cv::Mat
apply_color_correction
::estimate_relative_depth( const cv::Mat& image )
{
  if( image.channels() != 3 )
  {
    // Return uniform depth for grayscale
    return cv::Mat::ones( image.rows, image.cols, CV_32F ) * 0.5;
  }

  std::vector< cv::Mat > channels( 3 );
  cv::split( image, channels );

  cv::Mat blue_f, red_f;
  channels[0].convertTo( blue_f, CV_32F );
  channels[2].convertTo( red_f, CV_32F );

  // Estimate relative depth from blue/red ratio
  // In underwater images, red attenuates faster, so blue/red ratio indicates depth
  cv::Mat relative_depth;
  cv::divide( blue_f, red_f + 1.0, relative_depth );

  // Apply median filter to reduce noise
  cv::Mat depth_smoothed;
  relative_depth.convertTo( depth_smoothed, CV_32F );

  // Normalize to 0-1 range
  cv::normalize( depth_smoothed, depth_smoothed, 0, 1, cv::NORM_MINMAX );

  return depth_smoothed;
}

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::set_water_type_presets()
{
  if( c_water_type == "oceanic" )
  {
    c_red_attenuation = 0.5;
    c_green_attenuation = 0.3;
    c_blue_attenuation = 0.1;
  }
  else if( c_water_type == "coastal" )
  {
    c_red_attenuation = 0.6;
    c_green_attenuation = 0.4;
    c_blue_attenuation = 0.2;
  }
  else if( c_water_type == "turbid" )
  {
    c_red_attenuation = 0.7;
    c_green_attenuation = 0.5;
    c_blue_attenuation = 0.3;
  }
}

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::apply_backscatter_removal( cv::Mat& image )
{
  if( image.channels() != 3 )
  {
    return;
  }

  // Estimate backscatter as minimum values per channel in local regions
  cv::Mat backscatter;
  int kernel_size = std::max( image.rows, image.cols ) / 20;
  kernel_size = std::max( 5, kernel_size | 1 ); // Ensure odd and at least 5

  // Use morphological erosion to estimate backscatter
  cv::Mat kernel = cv::getStructuringElement( cv::MORPH_RECT,
    cv::Size( kernel_size, kernel_size ) );

  cv::Mat image_f;
  image.convertTo( image_f, CV_32F );
  cv::erode( image_f, backscatter, kernel );

  // Apply Gaussian blur to smooth the backscatter estimate
  cv::GaussianBlur( backscatter, backscatter, cv::Size( 0, 0 ), kernel_size / 2.0 );

  // Subtract backscatter and rescale
  cv::Mat result = image_f - backscatter * 0.5;
  cv::normalize( result, result, 0, 255, cv::NORM_MINMAX );
  result.convertTo( image, CV_8U );
}

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::apply_underwater_simple( cv::Mat& image )
{
  if( image.channels() != 3 )
  {
    return;
  }

  // Get depth map
  cv::Mat depth;
  load_depth_map();

  if( !m_depth_map.empty() )
  {
    // Resize depth map to match image if needed
    if( m_depth_map.rows != image.rows || m_depth_map.cols != image.cols )
    {
      cv::resize( m_depth_map, depth, image.size() );
    }
    else
    {
      depth = m_depth_map.clone();
    }
  }
  else if( c_use_auto_depth )
  {
    depth = estimate_relative_depth( image );
  }
  else
  {
    // No depth information, use uniform depth of 0.5
    depth = cv::Mat::ones( image.rows, image.cols, CV_32F ) * 0.5;
  }

  // Apply backscatter removal first
  if( c_backscatter_removal )
  {
    apply_backscatter_removal( image );
  }

  // Split into channels
  std::vector< cv::Mat > channels( 3 );
  cv::split( image, channels );

  // Compensate each channel based on depth and attenuation coefficient
  // Formula: corrected = original * exp(attenuation * depth)
  double attenuations[3] = { c_blue_attenuation, c_green_attenuation, c_red_attenuation };

  for( int i = 0; i < 3; i++ )
  {
    cv::Mat correction_factor;
    cv::exp( depth * attenuations[i], correction_factor );

    cv::Mat channel_f;
    channels[i].convertTo( channel_f, CV_32F );
    channel_f = channel_f.mul( correction_factor );

    // Clip to valid range
    cv::threshold( channel_f, channel_f, 255, 255, cv::THRESH_TRUNC );
    channel_f.convertTo( channels[i], CV_8U );
  }

  cv::merge( channels, image );

  // Normalize overall brightness
  cv::normalize( image, image, 0, 255, cv::NORM_MINMAX );
}

// -----------------------------------------------------------------------------------------------
void
apply_color_correction
::apply_underwater_fusion( cv::Mat& image )
{
  if( image.channels() != 3 )
  {
    return;
  }

  // Method 1: White balance corrected version
  cv::Mat wb_corrected;
  image.copyTo( wb_corrected );
  apply_gray_world_balance( wb_corrected );

  // Method 2: CLAHE enhanced version
  cv::Mat clahe_enhanced;
  cv::Mat lab_image;

#if CV_MAJOR_VERSION < 4
  cv::cvtColor( image, lab_image, CV_BGR2Lab );
#else
  cv::cvtColor( image, lab_image, cv::COLOR_BGR2Lab );
#endif

  std::vector< cv::Mat > lab_planes( 3 );
  cv::split( lab_image, lab_planes );

  cv::Ptr< cv::CLAHE > clahe = cv::createCLAHE();
  clahe->setClipLimit( 2.0 );
  clahe->setTilesGridSize( cv::Size( 8, 8 ) );

  cv::Mat tmp;
  clahe->apply( lab_planes[0], tmp );
  tmp.copyTo( lab_planes[0] );

  cv::merge( lab_planes, lab_image );

#if CV_MAJOR_VERSION < 4
  cv::cvtColor( lab_image, clahe_enhanced, CV_Lab2BGR );
#else
  cv::cvtColor( lab_image, clahe_enhanced, cv::COLOR_Lab2BGR );
#endif

  // Method 3: Gamma corrected version for shadow recovery
  cv::Mat gamma_corrected;
  image.copyTo( gamma_corrected );
  double saved_gamma = c_gamma;
  c_gamma = 0.7;
  apply_gamma_correction( gamma_corrected );
  c_gamma = saved_gamma;

  // Fuse the results using weighted averaging
  // Weights can be based on local contrast/entropy, but here we use simple averaging
  cv::Mat result;
  cv::addWeighted( wb_corrected, 0.4, clahe_enhanced, 0.4, 0, result );
  cv::addWeighted( result, 1.0, gamma_corrected, 0.2, 0, result );

  // Apply final color cast correction
  apply_gray_world_balance( result );

  result.copyTo( image );
}

// =================================================================================================

// -------------------------------------------------------------------------------------------------
bool
apply_color_correction
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  bool valid = true;

  double gamma = config->get_value< double >( "gamma" );
  if( gamma <= 0.0 )
  {
    LOG_ERROR( logger(), "Gamma value must be positive" );
    valid = false;
  }

  std::string method = config->get_value< std::string >( "underwater_method" );
  if( method != "simple" && method != "fusion" )
  {
    LOG_ERROR( logger(), "Invalid underwater_method: " << method
      << ". Must be 'simple' or 'fusion'" );
    valid = false;
  }

  std::string water_type = config->get_value< std::string >( "water_type" );
  if( water_type != "oceanic" && water_type != "coastal" && water_type != "turbid" )
  {
    LOG_ERROR( logger(), "Invalid water_type: " << water_type
      << ". Must be 'oceanic', 'coastal', or 'turbid'" );
    valid = false;
  }

  return valid;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr
apply_color_correction
::filter( kwiver::vital::image_container_sptr image_data )
{
  cv::Mat input_ocv =
    arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
      kwiver::arrows::ocv::image_container::BGR_COLOR );

  cv::Mat output_ocv;
  input_ocv.copyTo( output_ocv );

  // Ensure 8-bit for processing
  if( output_ocv.depth() != CV_8U )
  {
    cv::normalize( output_ocv, output_ocv, 255, 0, cv::NORM_MINMAX );
    output_ocv.convertTo( output_ocv, CV_8U );
  }

  // Apply gamma correction first (if enabled)
  if( c_apply_gamma )
  {
    apply_gamma_correction( output_ocv );
  }

  // Apply gray world white balance (if enabled)
  if( c_apply_gray_world )
  {
    apply_gray_world_balance( output_ocv );
  }

  // Apply underwater correction (if enabled)
  if( c_apply_underwater )
  {
    if( c_underwater_method == "fusion" )
    {
      apply_underwater_fusion( output_ocv );
    }
    else
    {
      apply_underwater_simple( output_ocv );
    }
  }

  kwiver::vital::image_container_sptr output(
    new arrows::ocv::image_container( output_ocv,
      kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  return output;
}


} // end namespace
