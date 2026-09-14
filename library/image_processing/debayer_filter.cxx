/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of debayer filter
 *
 * `cv::cvtColor`'s Bayer conversions until P7-T04b; `image_ops::demosaic`
 * since. The pattern letters mean what they say here -- see the note on
 * `bayer_pattern`, which is where OpenCV's reversed spelling is written
 * down.
 */

#include "debayer_filter.h"

#include <image_ops/color.h>
#include <image_ops/dispatch.h>
#include <image_ops/histogram.h>
#include <image_ops/pixel.h>

#include <viame/core_types/image_container.h>

#include <stdexcept>
#include <type_traits>

namespace io = viame::image_ops;

namespace viame {

namespace {

// ----------------------------------------------------------------------------
/// The config letter, as the mosaic it actually decodes.
///
/// **Reversed, and deliberately.** The filter called
/// `cv::cvtColor( ..., COLOR_Bayer<letter>2BGR )`, and OpenCV's Bayer
/// constants name the pattern the other way round from everyone else: the
/// constant that correctly decodes a blue-at-(0, 0) mosaic is the `RG` one.
/// So `pattern: BG` has always been decoding the mosaic as though red were
/// at (0, 0), and the picture comes out with **red and blue swapped**.
///
/// The recording settles it rather than the reasoning: `ocv_debayer`'s
/// recorded output on the BG fixture matches the source image only once its
/// channels are reversed. Reproduced rather than corrected, because every
/// model VIAME ships was trained on images this produced;
/// `design/lite-findings.md` records it.
io::bayer_pattern
pattern_of( std::string const& name )
{
  if( name == "BG" ) { return io::bayer_pattern::RG; }
  if( name == "GB" ) { return io::bayer_pattern::GR; }
  if( name == "RG" ) { return io::bayer_pattern::BG; }
  if( name == "GR" ) { return io::bayer_pattern::GB; }

  throw std::invalid_argument( "unknown Bayer pattern '" + name + "'" );
}

} // namespace

// ----------------------------------------------------------------------------
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

// ----------------------------------------------------------------------------
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

  auto const pattern = pattern_of( c_pattern );
  auto const force_8bit = c_force_8bit;

  auto const out = io::dispatch_pixel_type(
    image_data->get_image(),
    [ & ]( auto const& typed ) -> kwiver::vital::image
    {
      using pixel_t = std::decay_t< decltype( typed( 0, 0, 0 ) ) >;

      auto colour = io::demosaic( typed, pattern );

      if( !force_8bit || sizeof( pixel_t ) == 1 )
      {
        return kwiver::vital::image( colour );
      }

      // `cv::normalize( ..., 255, 0, NORM_MINMAX )` then a convert to 8U:
      // the range is stretched to fill 0..255 across **every** plane
      // together, and then rounded rather than truncated.
      auto const stretched = io::normalize_min_max( colour, 0.0, 255.0 );

      kwiver::vital::image_of< uint8_t > bytes(
        stretched.width(), stretched.height(), stretched.depth() );

      for( size_t plane = 0; plane < stretched.depth(); ++plane )
      {
        for( size_t j = 0; j < stretched.height(); ++j )
        {
          for( size_t i = 0; i < stretched.width(); ++i )
          {
            bytes( i, j, plane ) = io::saturate_pixel< uint8_t >(
              static_cast< double >( stretched( i, j, plane ) ) );
          }
        }
      }

      return kwiver::vital::image( bytes );
    } );

  return std::make_shared< kwiver::vital::simple_image_container >( out );
}

} // end namespace
