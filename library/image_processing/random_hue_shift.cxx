/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of random hue shift filter
 *
 * `cv::cvtColor` until P7-T04b; `image_ops::color` since. Four things are
 * reproduced rather than tidied, because this is a training augmentation and
 * a model was trained on what it produced:
 *
 * * the draws come from `rand()`, in this order and only when the
 *   corresponding range is non-zero, so a seeded run gives the same sequence
 *   it always did;
 * * every assignment lands in a `uchar`, which **truncates** rather than
 *   rounds -- a shifted value of 42.9 is 42;
 * * the hue wrap is `> 180`, not `>= 180`;
 * * the three colour shifts are drawn r, g, b and applied to channels 0, 1,
 *   2 of a **BGR** mat, so the first draw shifts blue. It makes no
 *   difference to the distribution and every difference to a seeded
 *   sequence.
 *
 * `ocv_random_hue_shift` throws on a single channel image about half the
 * time -- whenever its trigger fires -- which is finding 1.10 and is why the
 * recording pins the refusal without asserting it.
 */

#include "random_hue_shift.h"

#include <image_ops/color.h>
#include <image_ops/dispatch.h>

#include <viame/core_types/image_container.h>

#include <algorithm>
#include <cstdlib>

namespace io = viame::image_ops;

namespace viame {

namespace {

// ----------------------------------------------------------------------------
/// One draw of `rand()`, as the C++ took it.
double
draw()
{
  return rand() / ( RAND_MAX + 1.0 );
}

/// A shift in `[-range/2, range/2)`.
double
shift( double range )
{
  return range * draw() - ( range / 2.0 );
}

/// A double into a `uchar`, which is what every assignment here did:
/// truncation toward zero, and no saturation -- the callers clamp first.
uint8_t
to_byte( double value )
{
  return static_cast< uint8_t >( value );
}

} // namespace

// ----------------------------------------------------------------------------
bool
random_hue_shift
::check_configuration(
  [[maybe_unused]] kwiver::vital::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
kwiver::vital::image_container_sptr
random_hue_shift
::filter( kwiver::vital::image_container_sptr image_data )
{
  if( draw() >= c_trigger_percent )
  {
    return image_data;
  }

  // Eight bit only, as before: the C++ read every pixel through
  // `cv::Vec3b`, so anything else was already reading the wrong bytes.
  auto const source = kwiver::vital::image_of< uint8_t >(
    image_data->get_image() );

  auto hsv = io::rgb_to_hsv( source );

  auto const hue_shift = shift( c_hue_range );

  for( size_t i = 0; i < hsv.width(); ++i )
  {
    for( size_t j = 0; j < hsv.height(); ++j )
    {
      auto const value = hue_shift + hsv( i, j, 0 );

      hsv( i, j, 0 ) = to_byte( value > 180.0 ? value - 180.0
                                              : ( value < 0.0 ? value + 180.0
                                                              : value ) );
    }
  }

  if( c_sat_range )
  {
    auto const sat_shift = shift( c_sat_range );

    for( size_t i = 0; i < hsv.width(); ++i )
    {
      for( size_t j = 0; j < hsv.height(); ++j )
      {
        auto const value = sat_shift + hsv( i, j, 1 );
        hsv( i, j, 1 ) = to_byte( std::max( std::min( value, 255.0 ), 0.0 ) );
      }
    }
  }

  if( c_int_range )
  {
    auto const int_shift = shift( c_int_range );

    for( size_t i = 0; i < hsv.width(); ++i )
    {
      for( size_t j = 0; j < hsv.height(); ++j )
      {
        auto const value = int_shift + hsv( i, j, 2 );
        hsv( i, j, 2 ) = to_byte( std::max( std::min( value, 255.0 ), 0.0 ) );
      }
    }
  }

  auto out = io::hsv_to_rgb( hsv );

  if( c_rgb_shift_range )
  {
    // Drawn in the order the C++ drew them and applied to the BGR mat's
    // channels 0, 1 and 2 -- so the first lands on blue, which here is
    // plane 2.
    double const shifts[ 3 ] = { shift( c_rgb_shift_range ),
                                 shift( c_rgb_shift_range ),
                                 shift( c_rgb_shift_range ) };

    for( size_t channel = 0; channel < 3; ++channel )
    {
      auto const plane = 2 - channel;

      for( size_t i = 0; i < out.width(); ++i )
      {
        for( size_t j = 0; j < out.height(); ++j )
        {
          auto const value = shifts[ channel ] + out( i, j, plane );
          out( i, j, plane ) =
            to_byte( std::max( std::min( value, 255.0 ), 0.0 ) );
        }
      }
    }
  }

  return std::make_shared< kwiver::vital::simple_image_container >(
    kwiver::vital::image( out ) );
}

} // end namespace
