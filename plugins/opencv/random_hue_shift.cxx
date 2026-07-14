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
  if( rand() / ( RAND_MAX + 1.0 ) >= c_trigger_percent )
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

  double hue_shift = c_hue_range * ( rand() / ( RAND_MAX + 1.0 ) ) - ( c_hue_range / 2.0 );

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

  if( c_sat_range )
  {
    double sat_shift = c_sat_range * ( rand() / ( RAND_MAX + 1.0 ) ) - ( c_sat_range / 2.0 );

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

  if( c_int_range )
  {
    double int_shift = c_int_range * ( rand() / ( RAND_MAX + 1.0 ) ) - ( c_int_range / 2.0 );

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

  if( c_rgb_shift_range )
  {
    double r_shift = c_rgb_shift_range * ( rand() / ( RAND_MAX + 1.0 ) )
                       - ( c_rgb_shift_range / 2.0 );
    double g_shift = c_rgb_shift_range * ( rand() / ( RAND_MAX + 1.0 ) )
                       - ( c_rgb_shift_range / 2.0 );
    double b_shift = c_rgb_shift_range * ( rand() / ( RAND_MAX + 1.0 ) )
                       - ( c_rgb_shift_range / 2.0 );

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
