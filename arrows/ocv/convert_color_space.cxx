/*ckwg +5
 * Copyright 2019 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "convert_color_space.h"

#include <vital/exceptions.h>
#include <vital/types/color_space.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace kwiver {
namespace arrows {
namespace ocv {

namespace
{

typedef int cv_convert_code;

static const cv_convert_code CV_Invalid = -1;

/// \brief Is there an opencv conversion method between these 2 spaces?
cv_convert_code lookup_cv_conversion_code(
  vital::color_space space1, vital::color_space space2 )
{
  switch( space1 )
  {
    case vital::RGB:
      switch( space2 )
      {
        case vital::XYZ:
          return CV_RGB2XYZ;
        case vital::YCrCb:
          return CV_RGB2YCrCb;
        case vital::HSV:
          return CV_RGB2HSV;
        case vital::HLS:
          return CV_RGB2HLS;
        case vital::Lab:
          return CV_RGB2Lab;
        case vital::Luv:
          return CV_RGB2Luv;
        default:
          return CV_Invalid;
      }
    case vital::BGR:
      switch( space2 )
      {
        case vital::XYZ:
          return CV_BGR2XYZ;
        case vital::YCrCb:
          return CV_BGR2YCrCb;
        case vital::HSV:
          return CV_BGR2HSV;
        case vital::HLS:
          return CV_BGR2HLS;
        case vital::Lab:
          return CV_BGR2Lab;
        case vital::Luv:
          return CV_BGR2Luv;
        default:
          return CV_Invalid;
      }
    case vital::HSV:
      switch( space2 )
      {
        case vital::RGB:
          return CV_HSV2RGB;
        case vital::BGR:
          return CV_HSV2BGR;
        default:
          return CV_Invalid;
      }
    case vital::HLS:
      switch( space2 )
      {
        case vital::RGB:
          return CV_HLS2RGB;
        case vital::BGR:
          return CV_HLS2BGR;
        default:
          return CV_Invalid;
      }
    case vital::XYZ:
      switch( space2 )
      {
        case vital::RGB:
          return CV_XYZ2RGB;
        case vital::BGR:
          return CV_XYZ2BGR;
        default:
          return CV_Invalid;
      }
    case vital::Lab:
      switch( space2 )
      {
        case vital::RGB:
          return CV_Lab2RGB;
        case vital::BGR:
          return CV_Lab2BGR;
        default:
          return CV_Invalid;
      }
    case vital::Luv:
      switch( space2 )
      {
        case vital::RGB:
          return CV_Luv2RGB;
        case vital::BGR:
          return CV_Luv2BGR;
        default:
          return CV_Invalid;
      }
    case vital::YCrCb:
      switch( space2 )
      {
        case vital::RGB:
          return CV_YCrCb2RGB;
        case vital::BGR:
          return CV_YCrCb2BGR;
        default:
          return CV_Invalid;
      }
    default:
      return CV_Invalid;
  }
  return CV_Invalid;
}

} // end anonoymous namespace

// --------------------------------------------------------------------------------------
/// Private implementation class
class convert_color_space::priv
{
public:

  priv()
   : input_color_space( vital::RGB )
   , output_color_space( vital::HLS )
   , conversion_code( CV_RGB2HLS )
  {
  }

  ~priv()
  {
  }

  vital::color_space input_color_space;
  vital::color_space output_color_space;

  cv_convert_code conversion_code;
};

// --------------------------------------------------------------------------------------
convert_color_space
::convert_color_space()
: d( new priv() )
{
  attach_logger( "arrows.ocv.convert_color_space" );
}

convert_color_space
::~convert_color_space()
{
}

vital::config_block_sptr
convert_color_space
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  config->set_value( "input_color_space",
    vital::color_space_to_string( d->input_color_space ),
    "Input color space." );
  config->set_value( "output_color_space",
    vital::color_space_to_string( d->output_color_space ),
    "Output color space." );

  return config;
}

void
convert_color_space
::set_configuration( vital::config_block_sptr in_config )
{
  // Starting with our generated vital::config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing a
  // get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( in_config );

  // Settings for conversion
  d->input_color_space = vital::string_to_color_space(
    config->get_value< std::string >( "input_color_space" ) );
  d->output_color_space = vital::string_to_color_space(
    config->get_value< std::string >( "output_color_space" ) );

  d->conversion_code = lookup_cv_conversion_code(
    d->input_color_space, d->output_color_space );

  if( d->conversion_code == CV_Invalid )
  {
    throw vital::algorithm_configuration_exception(
      type_name(), impl_name(),
      "No conversion available between specified color spaces" );
  }
}

bool
convert_color_space
::check_configuration( vital::config_block_sptr config ) const
{
  if( vital::string_to_color_space(
    config->get_value< std::string >( "input_color_space" ) ) == vital::INVALID_CS )
  {
    throw vital::algorithm_configuration_exception(
      type_name(), impl_name(),
      "Invalid input color space specified: " +
      config->get_value< std::string >( "input_color_space" ) );
  }
  if( vital::string_to_color_space(
    config->get_value< std::string >( "output_color_space" ) ) == vital::INVALID_CS )
  {
    throw vital::algorithm_configuration_exception(
      type_name(), impl_name(),
      "Invalid output color space specified: " +
      config->get_value< std::string >( "output_color_space" ) );
  }
      
  return true;
}

// Perform stitch operation
kwiver::vital::image_container_sptr
convert_color_space
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  cv::Mat cv_output, cv_input =
    ocv::image_container::vital_to_ocv(
      image_data->get_image(), ocv::image_container::RGB_COLOR );

  cv::cvtColor( cv_input, cv_output, d->conversion_code );

  return vital::image_container_sptr(
    new ocv::image_container( cv_output, ocv::image_container::RGB_COLOR ) );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
