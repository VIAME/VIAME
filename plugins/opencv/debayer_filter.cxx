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

#include "debayer_filter.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <arrows/ocv/image_container.h>

#include <cmath>

namespace viame {

using namespace kwiver;

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class debayer_filter::priv
{
public:

  priv()
    : m_pattern( "BG" )
    , m_force_8bit( false )
    , m_is_first( true )
  {}

  ~priv() {}

  std::string m_pattern;
  bool m_force_8bit;
  bool m_is_first;
};

// =================================================================================================

debayer_filter
::debayer_filter()
  : d( new priv )
{
  attach_logger( "viame.opencv.debayer_filter" );
}


debayer_filter
::~debayer_filter()
{
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::config_block_sptr
debayer_filter
::get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "pattern", d->m_pattern, "Bayer pattern, can either be: BG, GB, RG, or GR. "
    "The two letters indicate the particular pattern type. These are components from the second "
    "row, second and third columns of the image, respectively." );

  config->set_value( "force_8bit", d->m_force_8bit, "Force output to be 8 bit" );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
debayer_filter
::set_configuration( kwiver::vital::config_block_sptr config )
{
  d->m_pattern = config->get_value< std::string >( "pattern" );
  d->m_force_8bit = config->get_value< bool >( "force_8bit" );
}


// -------------------------------------------------------------------------------------------------
bool
debayer_filter
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( !( d->m_pattern == "BG" ||
         d->m_pattern == "GB" ||
         d->m_pattern == "RG" ||
         d->m_pattern == "GR" ) )
  {
    LOG_ERROR( logger(), "Invalid pattern " << d->m_pattern );
    return false;
  }

  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::image_container_sptr
debayer_filter
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( image_data->depth() != 1 )
  {
    if( d->m_is_first )
    {
      LOG_WARN( logger(), "Not running debayering on multi-channel input" );
      d->m_is_first = false;
    }

    return image_data;
  }

  cv::Mat input_ocv =
    arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
      kwiver::arrows::ocv::image_container::BGR_COLOR );

  cv::Mat output_ocv;

  if( d->m_pattern == "BG" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerBG2BGR );
  }
  else if( d->m_pattern == "GB" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerGB2BGR );
  }
  else if( d->m_pattern == "RG" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerRG2BGR );
  }
  else if( d->m_pattern == "GR" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerGR2BGR );
  }

  if( d->m_force_8bit && output_ocv.depth() != CV_8U )
  { 
    cv::normalize( output_ocv, output_ocv, 255, 0, cv::NORM_MINMAX );
    output_ocv.convertTo( output_ocv, CV_8U );
  }

  kwiver::vital::image_container_sptr output(
    new arrows::ocv::image_container( output_ocv,
      kwiver::arrows::ocv::image_container::BGR_COLOR ) );

  return output;
}


} // end namespace
