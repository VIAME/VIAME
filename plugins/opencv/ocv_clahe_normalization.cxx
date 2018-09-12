/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "ocv_clahe_normalization.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <arrows/ocv/image_container.h>

#include <cmath>

namespace viame {

using namespace kwiver;

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class ocv_clahe_normalization::priv
{
public:

  priv()
    : m_force_8bit( false )
    , m_auto_balance( false )
    , m_clip_limit( 4 )
  {}

  ~priv() {}

  bool m_force_8bit;
  bool m_auto_balance;
  unsigned m_clip_limit;
};

// =================================================================================================

ocv_clahe_normalization
::ocv_clahe_normalization()
  : d( new priv )
{
  attach_logger( "viame.opencv.ocv_clahe_normalization" );
}


ocv_clahe_normalization
::~ocv_clahe_normalization()
{
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
ocv_clahe_normalization
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "force_8bit", d->m_force_8bit, "Force output to be 8 bit" );
  config->set_value( "auto_balance", d->m_auto_balance, "Perform automatic white balancing" );
  config->set_value( "clip_limit", d->m_clip_limit, "Clip limit used during hist normalization" );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
ocv_clahe_normalization
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_force_8bit = config->get_value< bool >( "force_8bit" );
  d->m_auto_balance = config->get_value< bool >( "auto_balance" );
  d->m_clip_limit = config->get_value< unsigned >( "clip_limit" );
}


// -------------------------------------------------------------------------------------------------
bool
ocv_clahe_normalization
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr
ocv_clahe_normalization
::filter( kwiver::vital::image_container_sptr image_data )
{
  cv::Mat input_ocv = arrows::ocv::image_container::vital_to_ocv( image_data->get_image() );
  cv::Mat lab_image, output_ocv;

  if( d->m_auto_balance )
  {
    cv::Scalar img_sum = sum( input_ocv );
    cv::Scalar illum = img_sum / cv::Scalar( input_ocv.rows * input_ocv.cols );

    std::vector< cv::Mat > rgb_channels( 3 );

    cv::split( input_ocv, rgb_channels );

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
    
    cv::merge( rgb_channels, input_ocv );
  }

  cv::cvtColor( input_ocv, lab_image, CV_BGR2Lab );

  std::vector< cv::Mat > lab_planes( 3 );
  cv::split( lab_image, lab_planes );

  cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
  clahe->setClipLimit( d->m_clip_limit );

  cv::Mat tmp;
  clahe->apply( lab_planes[0], tmp );
  tmp.copyTo( lab_planes[0] );

  cv::merge( lab_planes, lab_image );

  cv::cvtColor( lab_image, output_ocv, CV_Lab2BGR );

  if( d->m_force_8bit && output_ocv.depth() != CV_8U )
  { 
    cv::normalize( output_ocv, output_ocv, 255, 0, cv::NORM_MINMAX );
    output_ocv.convertTo( output_ocv, CV_8U );
  }

  kwiver::vital::image_container_sptr output( new arrows::ocv::image_container( output_ocv ) );

  return output;
}


} // end namespace
