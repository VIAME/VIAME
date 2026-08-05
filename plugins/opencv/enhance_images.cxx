/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "enhance_images.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/photo/photo.hpp>

#include <arrows/ocv/image_container.h>

#include <cmath>

namespace viame {

using namespace kwiver;

// -------------------------------------------------------------------------------------------------
bool
enhance_images
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr
enhance_images
::filter( kwiver::vital::image_container_sptr image_data )
{
  cv::Mat input_ocv =
    arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
      kwiver::arrows::ocv::image_container::BGR_COLOR );

  cv::Mat output_ocv;

  input_ocv.copyTo( output_ocv );

  if( c_apply_smoothing )
  {
    cv::medianBlur( output_ocv, output_ocv, c_smoothing_kernel );
  }

  if( c_apply_denoising && input_ocv.depth() == CV_8U )
  {
    cv::fastNlMeansDenoisingColored( output_ocv, output_ocv,
      c_denoise_coeff, c_denoise_coeff,
      c_denoise_kernel, c_denoise_kernel * 3 );
  }

  if( c_auto_balance && output_ocv.channels() == 3 )
  {
    cv::Scalar img_sum = sum( output_ocv );
    cv::Scalar illum = img_sum / cv::Scalar( output_ocv.rows * output_ocv.cols );

    std::vector< cv::Mat > rgb_channels( 3 );

    cv::split( output_ocv, rgb_channels );

    cv::Mat red = rgb_channels[2];
    cv::Mat green = rgb_channels[1];
    cv::Mat blue = rgb_channels[0];

    double scale = ( illum( 0 ) + illum( 1 ) + illum( 2 ) ) / 3;

    red = red * scale / illum( 2 );
    green = green * scale / illum( 1 );
    blue = blue * scale / illum( 0 );

    rgb_channels[2] = red;
    rgb_channels[1] = green;
    rgb_channels[0] = blue;

    cv::merge( rgb_channels, output_ocv );
  }

  if( c_force_8bit && output_ocv.depth() != CV_8U )
  {
    cv::normalize( output_ocv, output_ocv, 255, 0, cv::NORM_MINMAX );
    output_ocv.convertTo( output_ocv, CV_8U );
  }

  if( c_apply_denoising && input_ocv.depth() != CV_8U )
  {
    if( output_ocv.depth() != CV_8U )
    {
      throw std::runtime_error( "Unable to perform denoising on not 8-bit imagery" );
    }

    cv::fastNlMeansDenoisingColored( output_ocv, output_ocv,
      c_denoise_coeff, c_denoise_coeff,
      c_denoise_kernel, c_denoise_kernel * 3 );
  }

  if( c_apply_clahe )
  {
    cv::Mat lab_image;

    if( output_ocv.depth() != CV_8U && output_ocv.depth() != CV_32F )
    {
      output_ocv.convertTo( output_ocv, CV_32F );
    }

    if( output_ocv.channels() == 3 )
    {
#if CV_MAJOR_VERSION < 4
      cv::cvtColor( output_ocv, lab_image, CV_BGR2Lab );
#else
      cv::cvtColor( output_ocv, lab_image, cv::COLOR_BGR2Lab );
#endif
    }
    else
    {
      lab_image = output_ocv;
    }

    std::vector< cv::Mat > lab_planes( output_ocv.channels() );
    cv::split( lab_image, lab_planes );

    cv::Ptr< cv::CLAHE > clahe = cv::createCLAHE();
    clahe->setClipLimit( c_clip_limit );

    if( output_ocv.depth() == CV_32F )
    {
      double min, max;
      cv::Mat tmp1, tmp2;
      cv::minMaxLoc( lab_planes[0], &min, &max );
      double scale1 = ( max > 0.0 ? 255.0 / ( max - min ) : 1.0 );
      double shift1 = -( min * scale1 );
      double scale2 = ( max > 0.0 ?  max / 255.0 : 1.0 );
      lab_planes[0].convertTo( tmp1, CV_8U, scale1, shift1 );
      clahe->apply( tmp1, tmp2 );
      tmp2.convertTo( lab_planes[0], CV_32F, ( 1.0 / scale2 ) );
    }
    else
    {
      cv::Mat tmp;
      clahe->apply( lab_planes[0], tmp );
      tmp.copyTo( lab_planes[0] );
    }

    if( output_ocv.channels() != 1 )
    {
      cv::merge( lab_planes, lab_image );
#if CV_MAJOR_VERSION < 4
      cv::cvtColor( lab_image, output_ocv, CV_Lab2BGR );
#else
      cv::cvtColor( lab_image, output_ocv, cv::COLOR_Lab2BGR );
#endif
    }
    else
    {
      lab_planes[0].copyTo( output_ocv );
    }
  }

  if( c_saturation != 1.0 && output_ocv.channels() == 3 )
  {
    cv::Mat hsv_image;
#if CV_MAJOR_VERSION < 4
    cv::cvtColor( output_ocv, hsv_image, CV_BGR2HSV );
#else
    cv::cvtColor( output_ocv, hsv_image, cv::COLOR_BGR2HSV );
#endif

    std::vector< cv::Mat > hsv_channels( 3 );

    cv::split( hsv_image, hsv_channels );

    cv::Mat hue = hsv_channels[0];
    cv::Mat sat = hsv_channels[1];
    cv::Mat val = hsv_channels[2];

    sat *= c_saturation;

    cv::merge( hsv_channels, output_ocv );
#if CV_MAJOR_VERSION < 4
    cv::cvtColor( output_ocv, output_ocv, CV_HSV2BGR );
#else
    cv::cvtColor( output_ocv, output_ocv, cv::COLOR_HSV2BGR );
#endif
  }

  if( c_apply_sharpening )
  {
    cv::Mat tmp;
    cv::GaussianBlur( output_ocv, tmp, cv::Size( 0, 0 ), c_sharpening_kernel );
    cv::addWeighted( output_ocv, 1.0 + c_sharpening_weight, tmp,
                     0.0 - c_sharpening_weight, 0, output_ocv );
  }

  kwiver::vital::image_container_sptr output(
    new arrows::ocv::image_container( output_ocv,
      kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  return output;
}


} // end namespace
