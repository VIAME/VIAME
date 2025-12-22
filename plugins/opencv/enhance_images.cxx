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

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class enhance_images::priv
{
public:

  priv()
    : m_apply_smoothing( false )
    , m_smoothing_kernel( 3 )
    , m_apply_denoising( false )
    , m_denoise_kernel( 3 )
    , m_denoise_coeff( 3 )
    , m_force_8bit( false )
    , m_auto_balance( false )
    , m_apply_clahe( false )
    , m_clip_limit( 4 )
    , m_saturation( 1.0 )
    , m_apply_sharpening( false )
    , m_sharpening_kernel( 3 )
    , m_sharpening_weight( 0.5 )
  {}

  ~priv() {}

  bool m_apply_smoothing;
  unsigned m_smoothing_kernel;
  bool m_apply_denoising;
  unsigned m_denoise_kernel;
  unsigned m_denoise_coeff;
  bool m_force_8bit;
  bool m_auto_balance;
  bool m_apply_clahe;
  unsigned m_clip_limit;
  float m_saturation;
  bool m_apply_sharpening;
  unsigned m_sharpening_kernel;
  double m_sharpening_weight;
};

// =================================================================================================

enhance_images
::enhance_images()
  : d( new priv )
{
  attach_logger( "viame.opencv.enhance_images" );
}


enhance_images
::~enhance_images()
{
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
enhance_images
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "apply_smoothing", d->m_apply_smoothing, "Apply smoothing to the input" );
  config->set_value( "smoothing_kernel", d->m_smoothing_kernel, "Smoothing kernel size" );
  config->set_value( "apply_denoising", d->m_apply_denoising, "Apply denoising to the input" );
  config->set_value( "denoise_kernel", d->m_denoise_kernel, "Denoising kernel size" );
  config->set_value( "denoise_coeff", d->m_denoise_coeff, "Denoising coefficient" );
  config->set_value( "force_8bit", d->m_force_8bit, "Force output to be 8 bit" );
  config->set_value( "auto_balance", d->m_auto_balance, "Perform automatic white balancing" );
  config->set_value( "apply_clahe", d->m_apply_clahe, "Apply CLAHE illumination normalization" );
  config->set_value( "clip_limit", d->m_clip_limit, "Clip limit used during hist normalization" );
  config->set_value( "saturation", d->m_saturation, "Saturation scale factor" );
  config->set_value( "apply_sharpening", d->m_apply_sharpening, "Apply sharpening to the input" );
  config->set_value( "sharpening_kernel", d->m_sharpening_kernel, "Sharpening kernel size" );
  config->set_value( "sharpening_weight", d->m_sharpening_weight, "Sharpening weight [0.0,1.0]" );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
enhance_images
::set_configuration( kwiver::vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_apply_smoothing = config->get_value< bool >( "apply_smoothing" );
  d->m_smoothing_kernel = config->get_value< unsigned >( "smoothing_kernel" );
  d->m_apply_denoising = config->get_value< bool >( "apply_denoising" );
  d->m_denoise_kernel = config->get_value< unsigned >( "denoise_kernel" );
  d->m_denoise_coeff = config->get_value< unsigned >( "denoise_coeff" );
  d->m_force_8bit = config->get_value< bool >( "force_8bit" );
  d->m_auto_balance = config->get_value< bool >( "auto_balance" );
  d->m_apply_clahe = config->get_value< bool >( "apply_clahe" );
  d->m_clip_limit = config->get_value< unsigned >( "clip_limit" );
  d->m_saturation = config->get_value< float >( "saturation" );
  d->m_apply_sharpening = config->get_value< bool >( "apply_sharpening" );
  d->m_sharpening_kernel = config->get_value< unsigned >( "sharpening_kernel" );
  d->m_sharpening_weight = config->get_value< double >( "sharpening_weight" );
}


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

  if( d->m_apply_smoothing )
  {
    cv::medianBlur( output_ocv, output_ocv, d->m_smoothing_kernel );
  }

  if( d->m_apply_denoising && input_ocv.depth() == CV_8U )
  {
    cv::fastNlMeansDenoisingColored( output_ocv, output_ocv,
      d->m_denoise_coeff, d->m_denoise_coeff,
      d->m_denoise_kernel, d->m_denoise_kernel * 3 );
  }

  if( d->m_auto_balance && output_ocv.channels() == 3 )
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

  if( d->m_force_8bit && output_ocv.depth() != CV_8U )
  {
    cv::normalize( output_ocv, output_ocv, 255, 0, cv::NORM_MINMAX );
    output_ocv.convertTo( output_ocv, CV_8U );
  }

  if( d->m_apply_denoising && input_ocv.depth() != CV_8U )
  {
    if( output_ocv.depth() != CV_8U )
    {
      throw std::runtime_error( "Unable to perform denoising on not 8-bit imagery" );
    }

    cv::fastNlMeansDenoisingColored( output_ocv, output_ocv,
      d->m_denoise_coeff, d->m_denoise_coeff,
      d->m_denoise_kernel, d->m_denoise_kernel * 3 );
  }

  if( d->m_apply_clahe )
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
    clahe->setClipLimit( d->m_clip_limit );

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

  if( d->m_saturation != 1.0 && output_ocv.channels() == 3 )
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

    sat *= d->m_saturation;

    cv::merge( hsv_channels, output_ocv );
#if CV_MAJOR_VERSION < 4
    cv::cvtColor( output_ocv, output_ocv, CV_HSV2BGR );
#else
    cv::cvtColor( output_ocv, output_ocv, cv::COLOR_HSV2BGR );
#endif
  }

  if( d->m_apply_sharpening )
  {
    cv::Mat tmp;
    cv::GaussianBlur( output_ocv, tmp, cv::Size( 0, 0 ), d->m_sharpening_kernel );
    cv::addWeighted( output_ocv, 1.0 + d->m_sharpening_weight, tmp,
                     0.0 - d->m_sharpening_weight, 0, output_ocv );
  }

  kwiver::vital::image_container_sptr output(
    new arrows::ocv::image_container( output_ocv,
      kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  return output;
}


} // end namespace
