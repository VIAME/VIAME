/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "debayer_filter.h"

#include <opencv2/imgproc/imgproc.hpp>

#include <arrows/ocv/image_container.h>

#include <cmath>

namespace viame {

using namespace kwiver;

// -------------------------------------------------------------------------------------------------
bool
debayer_filter
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( !( c_pattern == "BG" ||
         c_pattern == "GB" ||
         c_pattern == "RG" ||
         c_pattern == "GR" ) )
  {
    LOG_ERROR( logger(), "Invalid pattern " << c_pattern );
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
    if( m_is_first )
    {
      LOG_WARN( logger(), "Not running debayering on multi-channel input" );
      m_is_first = false;
    }

    return image_data;
  }

  cv::Mat input_ocv =
    arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
      kwiver::arrows::ocv::image_container::BGR_COLOR );

  cv::Mat output_ocv;

  if( c_pattern == "BG" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerBG2BGR );
  }
  else if( c_pattern == "GB" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerGB2BGR );
  }
  else if( c_pattern == "RG" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerRG2BGR );
  }
  else if( c_pattern == "GR" )
  {
    cv::cvtColor( input_ocv, output_ocv, cv::COLOR_BayerGR2BGR );
  }

  if( c_force_8bit && output_ocv.depth() != CV_8U )
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
