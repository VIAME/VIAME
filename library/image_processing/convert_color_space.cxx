/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of colour space conversion filter
 *
 * `cv::cvtColor` until P7-T04b; `image_ops::color` since. The bridge was
 * asked for an `RGB_COLOR` mat both ways, so the vital image's planes went
 * into `cvtColor` and came back out untouched -- which is why an `input` of
 * `bgr` means "these planes hold BGR" rather than anything the bridge did.
 *
 * **Three of the C++'s colour spaces are gone: XYZ, YCrCb and Luv.** No
 * shipped pipeline or config asks for any of them -- the only use of this
 * filter in the tree is `train_aug_intensity_hue_motion.pipe`, which asks
 * for rgb to hls, and that is also the registered default. Carrying three
 * more conversion pairs that nothing selects is the "extra" this branch has
 * been removing everywhere else. Asking for one is a configuration error
 * naming the space, as an unavailable pair already was.
 */

#include "convert_color_space.h"

#include <image_ops/color.h>
#include <image_ops/dispatch.h>

#include <viame/algorithm_framework/exceptions/algorithm.h>
#include <viame/core_types/color_space.h>
#include <viame/core_types/image_container.h>

#include <string>

namespace io = viame::image_ops;

namespace viame {

namespace {

namespace kv = kwiver::vital;

// ----------------------------------------------------------------------------
/// Whether a space is one of the two plane orders rather than a conversion.
bool
is_channel_order( kv::color_space space )
{
  return space == kv::RGB || space == kv::BGR;
}

// ----------------------------------------------------------------------------
/// Whether this build converts to and from \p space.
bool
is_supported( kv::color_space space )
{
  return is_channel_order( space ) || space == kv::HSV ||
         space == kv::HLS || space == kv::Lab;
}

// ----------------------------------------------------------------------------
/// The planes in the other order, which is what BGR and RGB differ by.
template < typename T >
kwiver::vital::image_of< T >
swap_channels( kwiver::vital::image_of< T > const& image )
{
  return io::swap_rb( image );
}

// ----------------------------------------------------------------------------
template < typename T >
kwiver::vital::image_of< T >
convert( kwiver::vital::image_of< T > const& image,
         kv::color_space from, kv::color_space to )
{
  // One of the two ends is always a channel order: the C++ had no pair that
  // converted one colour space straight into another, and nothing asks for
  // one.
  if( is_channel_order( from ) )
  {
    auto const rgb = ( from == kv::BGR ) ? swap_channels( image ) : image;

    switch( to )
    {
      case kv::HSV: return io::rgb_to_hsv( rgb );
      case kv::HLS: return io::rgb_to_hls( rgb );
      case kv::Lab: return io::rgb_to_lab( rgb );
      default: break;
    }
  }
  else if( is_channel_order( to ) )
  {
    kwiver::vital::image_of< T > rgb;

    switch( from )
    {
      case kv::HSV: rgb = io::hsv_to_rgb( image ); break;
      case kv::HLS: rgb = io::hls_to_rgb( image ); break;
      case kv::Lab: rgb = io::lab_to_rgb( image ); break;
      default: break;
    }

    if( rgb.size() )
    {
      return ( to == kv::BGR ) ? swap_channels( rgb ) : rgb;
    }
  }

  throw kv::algorithm_configuration_exception(
    "convert_color_space", "ocv_convert_color",
    "No conversion available between specified color spaces" );
}

} // namespace

// ----------------------------------------------------------------------------
void
convert_color_space
::initialize()
{
  resolve_conversion_code();
}

// ----------------------------------------------------------------------------
void
convert_color_space
::set_configuration_internal(
  [[maybe_unused]] kwiver::vital::config_block_sptr config )
{
  resolve_conversion_code();
}

// ----------------------------------------------------------------------------
void
convert_color_space
::resolve_conversion_code()
{
  auto const input = kwiver::vital::string_to_color_space(
    c_input_color_space );
  auto const output = kwiver::vital::string_to_color_space(
    c_output_color_space );

  auto const usable =
    is_supported( input ) && is_supported( output ) &&
    ( is_channel_order( input ) != is_channel_order( output ) );

  if( !usable )
  {
    throw kwiver::vital::algorithm_configuration_exception(
      "convert_color_space", this->impl_name(),
      "No conversion available between specified color spaces" );
  }
}

// ----------------------------------------------------------------------------
bool
convert_color_space
::check_configuration( kwiver::vital::config_block_sptr config ) const
{
  if( kwiver::vital::string_to_color_space(
    config->get_value< std::string >( "input_color_space" ) ) ==
      kwiver::vital::INVALID_CS )
  {
    throw kwiver::vital::algorithm_configuration_exception(
      "convert_color_space", this->impl_name(),
      "Invalid input color space specified: " +
      config->get_value< std::string >( "input_color_space" ) );
  }
  if( kwiver::vital::string_to_color_space(
    config->get_value< std::string >( "output_color_space" ) ) ==
      kwiver::vital::INVALID_CS )
  {
    throw kwiver::vital::algorithm_configuration_exception(
      "convert_color_space", this->impl_name(),
      "Invalid output color space specified: " +
      config->get_value< std::string >( "output_color_space" ) );
  }

  return true;
}

// ----------------------------------------------------------------------------
// Perform color conversion
kwiver::vital::image_container_sptr
convert_color_space
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( !image_data )
  {
    return kwiver::vital::image_container_sptr();
  }

  auto const input = kwiver::vital::string_to_color_space(
    c_input_color_space );
  auto const output = kwiver::vital::string_to_color_space(
    c_output_color_space );

  auto const converted = io::dispatch_pixel_type(
    image_data->get_image(),
    [ & ]( auto const& typed ) -> kwiver::vital::image
    {
      return kwiver::vital::image( convert( typed, input, output ) );
    } );

  return std::make_shared< kwiver::vital::simple_image_container >(
    converted );
}

} // end namespace viame
