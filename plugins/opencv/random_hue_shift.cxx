/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "random_hue_shift.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <arrows/ocv/image_container.h>

#include <cmath>

namespace viame {

using namespace kwiver;

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class random_hue_shift::priv
{
public:

  priv()
    : m_trigger_percent( 0.50 )
    , m_hue_range( 0.0 )
    , m_sat_range( 0.0 )
    , m_int_range( 0.0 )
    , m_rgb_shift_range( 0.0 )
  {}

  ~priv() {}

  double m_trigger_percent;

  double m_hue_range;
  double m_sat_range;
  double m_int_range;

  double m_rgb_shift_range;
};

// =================================================================================================

random_hue_shift
::random_hue_shift()
  : d( new priv )
{
  attach_logger( "viame.opencv.random_hue_shift" );
}


random_hue_shift
::~random_hue_shift()
{
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
random_hue_shift
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "trigger_percent", d->m_trigger_percent, "Trigger for other operations" );

  config->set_value( "hue_range", d->m_hue_range, "Hue random adjustment range" );
  config->set_value( "sat_range", d->m_sat_range, "Saturation random adjustment range" );
  config->set_value( "int_range", d->m_int_range, "Intensity random adjustment range" );

  config->set_value( "rgb_shift_range", d->m_rgb_shift_range, "Random color shift range" );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
random_hue_shift
::set_configuration( kwiver::vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_trigger_percent = config->get_value< double >( "trigger_percent" );

  d->m_hue_range = config->get_value< double >( "hue_range" );
  d->m_sat_range = config->get_value< double >( "sat_range" );
  d->m_int_range = config->get_value< double >( "int_range" );

  d->m_rgb_shift_range = config->get_value< double >( "rgb_shift_range" );
}


// -------------------------------------------------------------------------------------------------
bool
random_hue_shift
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr
random_hue_shift
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( rand() / ( RAND_MAX + 1.0 ) >= d->m_trigger_percent )
  {
    return image_data;
  }

  cv::Mat input_ocv =
    arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
      kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Shift Hue
  cv::Mat hsv_image, output_ocv;

#if CV_MAJOR_VERSION < 4
  cv::cvtColor( input_ocv, hsv_image, CV_BGR2HSV );
#else
  cv::cvtColor( input_ocv, hsv_image, cv::COLOR_BGR2HSV );
#endif

  double hue_shift = d->m_hue_range * ( rand() / ( RAND_MAX + 1.0 ) ) - ( d->m_hue_range / 2.0 );

  for( auto i = 0; i < input_ocv.cols; ++i )
  {
    for( auto j = 0; j < input_ocv.rows; ++j )
    {
      double new_value = hue_shift + hsv_image.at<cv::Vec3b>(j,i)[0];

      if( new_value > 180 )
      {
        hsv_image.at<cv::Vec3b>(j,i)[0] = new_value - 180;
      }
      else if( new_value < 0 )
      {
        hsv_image.at<cv::Vec3b>(j,i)[0] = new_value + 180;
      }
      else
      {
        hsv_image.at<cv::Vec3b>(j,i)[0] = new_value;
      }
    }
  }

  if( d->m_sat_range )
  {
    double sat_shift = d->m_sat_range * ( rand() / ( RAND_MAX + 1.0 ) ) - ( d->m_sat_range / 2.0 );

    for( auto i = 0; i < input_ocv.cols; ++i )
    {
      for( auto j = 0; j < input_ocv.rows; ++j )
      {
        double new_value = sat_shift + hsv_image.at<cv::Vec3b>(j,i)[1];

        if( new_value > 255.0 )
        {
          hsv_image.at<cv::Vec3b>(j,i)[1] = 255;
        }
        else if( new_value < 0.0 )
        {
          hsv_image.at<cv::Vec3b>(j,i)[1] = 0;
        }
        else
        {
          hsv_image.at<cv::Vec3b>(j,i)[1] = new_value;
        }
      }
    }
  }

  if( d->m_int_range )
  {
    double int_shift = d->m_int_range * ( rand() / ( RAND_MAX + 1.0 ) ) - ( d->m_int_range / 2.0 );

    for( auto i = 0; i < input_ocv.cols; ++i )
    {
      for( auto j = 0; j < input_ocv.rows; ++j )
      {
        double new_value = int_shift + hsv_image.at<cv::Vec3b>(j,i)[2];

        if( new_value > 255.0 )
        {
          hsv_image.at<cv::Vec3b>(j,i)[2] = 255;
        }
        else if( new_value < 0.0 )
        {
          hsv_image.at<cv::Vec3b>(j,i)[2] = 0;
        }
        else
        {
          hsv_image.at<cv::Vec3b>(j,i)[2] = new_value;
        }
      }
    }
  }

#if CV_MAJOR_VERSION < 4
  cv::cvtColor( hsv_image, output_ocv, CV_HSV2BGR );
#else
  cv::cvtColor( hsv_image, output_ocv, cv::COLOR_HSV2BGR );
#endif

  if( d->m_rgb_shift_range )
  {
    double r_shift = d->m_rgb_shift_range * ( rand() / ( RAND_MAX + 1.0 ) )
                       - ( d->m_rgb_shift_range / 2.0 );
    double g_shift = d->m_rgb_shift_range * ( rand() / ( RAND_MAX + 1.0 ) )
                       - ( d->m_rgb_shift_range / 2.0 );
    double b_shift = d->m_rgb_shift_range * ( rand() / ( RAND_MAX + 1.0 ) )
                       - ( d->m_rgb_shift_range / 2.0 );

    for( auto i = 0; i < output_ocv.cols; ++i )
    {
      for( auto j = 0; j < output_ocv.rows; ++j )
      {
        output_ocv.at<cv::Vec3b>(j,i)[0] =
          std::max( std::min( r_shift + output_ocv.at<cv::Vec3b>(j,i)[0], 255.0 ), 0.0 );
        output_ocv.at<cv::Vec3b>(j,i)[1] =
          std::max( std::min( g_shift + output_ocv.at<cv::Vec3b>(j,i)[1], 255.0 ), 0.0 );
        output_ocv.at<cv::Vec3b>(j,i)[2] =
          std::max( std::min( b_shift + output_ocv.at<cv::Vec3b>(j,i)[2], 255.0 ), 0.0 );
      }
    }
  }

  kwiver::vital::image_container_sptr output(
    new arrows::ocv::image_container( output_ocv,
      kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  return output;
}


} // end namespace
