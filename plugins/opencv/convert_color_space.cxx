/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "convert_color_space.h"

#include <vital/exceptions.h>
#include <vital/types/color_space.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame {

namespace
{

typedef int cv_convert_code;

static const cv_convert_code CV_Invalid = -1;

/// \brief Is there an opencv conversion method between these 2 spaces?
cv_convert_code lookup_cv_conversion_code(
  kwiver::vital::color_space space1, kwiver::vital::color_space space2 )
{
  switch( space1 )
  {
    case kwiver::vital::RGB:
      switch( space2 )
      {
        case kwiver::vital::XYZ:
          return cv::COLOR_RGB2XYZ;
        case kwiver::vital::YCrCb:
          return cv::COLOR_RGB2YCrCb;
        case kwiver::vital::HSV:
          return cv::COLOR_RGB2HSV;
        case kwiver::vital::HLS:
          return cv::COLOR_RGB2HLS;
        case kwiver::vital::Lab:
          return cv::COLOR_RGB2Lab;
        case kwiver::vital::Luv:
          return cv::COLOR_RGB2Luv;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::BGR:
      switch( space2 )
      {
        case kwiver::vital::XYZ:
          return cv::COLOR_BGR2XYZ;
        case kwiver::vital::YCrCb:
          return cv::COLOR_BGR2YCrCb;
        case kwiver::vital::HSV:
          return cv::COLOR_BGR2HSV;
        case kwiver::vital::HLS:
          return cv::COLOR_BGR2HLS;
        case kwiver::vital::Lab:
          return cv::COLOR_BGR2Lab;
        case kwiver::vital::Luv:
          return cv::COLOR_BGR2Luv;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::HSV:
      switch( space2 )
      {
        case kwiver::vital::RGB:
          return cv::COLOR_HSV2RGB;
        case kwiver::vital::BGR:
          return cv::COLOR_HSV2BGR;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::HLS:
      switch( space2 )
      {
        case kwiver::vital::RGB:
          return cv::COLOR_HLS2RGB;
        case kwiver::vital::BGR:
          return cv::COLOR_HLS2BGR;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::XYZ:
      switch( space2 )
      {
        case kwiver::vital::RGB:
          return cv::COLOR_XYZ2RGB;
        case kwiver::vital::BGR:
          return cv::COLOR_XYZ2BGR;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::Lab:
      switch( space2 )
      {
        case kwiver::vital::RGB:
          return cv::COLOR_Lab2RGB;
        case kwiver::vital::BGR:
          return cv::COLOR_Lab2BGR;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::Luv:
      switch( space2 )
      {
        case kwiver::vital::RGB:
          return cv::COLOR_Luv2RGB;
        case kwiver::vital::BGR:
          return cv::COLOR_Luv2BGR;
        default:
          return CV_Invalid;
      }
    case kwiver::vital::YCrCb:
      switch( space2 )
      {
        case kwiver::vital::RGB:
          return cv::COLOR_YCrCb2RGB;
        case kwiver::vital::BGR:
          return cv::COLOR_YCrCb2BGR;
        default:
          return CV_Invalid;
      }
    default:
      return CV_Invalid;
  }
  return CV_Invalid;
}

} // end anonymous namespace

// --------------------------------------------------------------------------------------
/// Private implementation class
class convert_color_space::priv
{
public:

  priv()
   : input_color_space( kwiver::vital::RGB )
   , output_color_space( kwiver::vital::HLS )
   , conversion_code( cv::COLOR_RGB2HLS )
  {
  }

  ~priv()
  {
  }

  kwiver::vital::color_space input_color_space;
  kwiver::vital::color_space output_color_space;

  cv_convert_code conversion_code;
};

// --------------------------------------------------------------------------------------
convert_color_space
::convert_color_space()
: d( new priv() )
{
  attach_logger( "viame.opencv.convert_color_space" );
}

convert_color_space
::~convert_color_space()
{
}

kwiver::vital::config_block_sptr
convert_color_space
::get_configuration() const
{
  // get base config from base class
  kwiver::vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "input_color_space",
    kwiver::vital::color_space_to_string( d->input_color_space ),
    "Input color space." );
  config->set_value( "output_color_space",
    kwiver::vital::color_space_to_string( d->output_color_space ),
    "Output color space." );

  return config;
}

void
convert_color_space
::set_configuration( kwiver::vital::config_block_sptr in_config )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing a
  // get_value() call.
  kwiver::vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for conversion
  d->input_color_space = kwiver::vital::string_to_color_space(
    config->get_value< std::string >( "input_color_space" ) );
  d->output_color_space = kwiver::vital::string_to_color_space(
    config->get_value< std::string >( "output_color_space" ) );

  d->conversion_code = lookup_cv_conversion_code(
    d->input_color_space, d->output_color_space );

  if( d->conversion_code == CV_Invalid )
  {
    throw kwiver::vital::algorithm_configuration_exception(
      "convert_color_space", this->impl_name(),
      "No conversion available between specified color spaces" );
  }
}

bool
convert_color_space
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( kwiver::vital::string_to_color_space(
    config->get_value< std::string >( "input_color_space" ) ) == kwiver::vital::INVALID_CS )
  {
    throw kwiver::vital::algorithm_configuration_exception(
      "convert_color_space", this->impl_name(),
      "Invalid input color space specified: " +
      config->get_value< std::string >( "input_color_space" ) );
  }
  if( kwiver::vital::string_to_color_space(
    config->get_value< std::string >( "output_color_space" ) ) == kwiver::vital::INVALID_CS )
  {
    throw kwiver::vital::algorithm_configuration_exception(
      "convert_color_space", this->impl_name(),
      "Invalid output color space specified: " +
      config->get_value< std::string >( "output_color_space" ) );
  }

  return true;
}

// Perform color conversion
kwiver::vital::image_container_sptr
convert_color_space
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  cv::Mat cv_output, cv_input =
    kwiver::arrows::ocv::image_container::vital_to_ocv(
      image_data->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR );

  cv::cvtColor( cv_input, cv_output, d->conversion_code );

  return kwiver::vital::image_container_sptr(
    new kwiver::arrows::ocv::image_container( cv_output,
      kwiver::arrows::ocv::image_container::RGB_COLOR ) );
}

} // end namespace viame
