/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_CONVERT_H
#define VIAME_IMAGE_OPS_CONVERT_H

#include <image_ops/channels.h>
#include <image_ops/pixel.h>
#include <image_ops/statistics.h>

#include <vital/types/image.h>

#include <cstddef>
#include <limits>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Convert every pixel to \p Out with a plain cast.
template < typename Out, typename In >
kwiver::vital::image_of< Out >
cast( kwiver::vital::image_of< In > const& image )
{
  kwiver::vital::image_of< Out > result( image.width(), image.height(),
                                         image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        result( i, j, plane ) =
          cast_pixel< Out >( image( i, j, plane ) );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Multiply by \p factor, round, and saturate at the top of \p Out.
///
/// Only the top is saturated: the cutoff is computed as
/// `max(Out) / factor` in the *input* type, so for an unsigned input
/// nothing can come out below zero anyway, and for a signed one the
/// negative side casts the way `round_pixel` does. That asymmetry is VXL's
/// `scale_image` and pipelines depend on the clamp at the top.
template < typename Out, typename In >
kwiver::vital::image_of< Out >
scale( kwiver::vital::image_of< In > const& image, double factor )
{
  kwiver::vital::image_of< Out > result( image.width(), image.height(),
                                         image.depth() );

  constexpr Out max_output = std::numeric_limits< Out >::max();
  auto const max_input =
    static_cast< In >( pixel_max< Out >() / factor );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const value = image( i, j, plane );

        result( i, j, plane ) =
          ( value <= max_input )
          ? static_cast< Out >( static_cast< double >( value ) * factor + 0.5 )
          : max_output;
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Stretch the range between two percentiles across the whole of \p Out.
///
/// Values below the lower percentile become zero and values above the upper
/// one become the type maximum. When the two percentiles coincide, so there
/// is no range to stretch, the input range is mapped to the output range
/// instead, which keeps a flat image from becoming noise.
///
/// \param lower, upper   Fractions in [0, 1].
/// \param sampling_points How many samples the percentiles are taken from.
template < typename Out, typename In >
kwiver::vital::image_of< Out >
percentile_stretch( kwiver::vital::image_of< In > const& image,
                    double lower, double upper,
                    size_t sampling_points,
                    bool ignore_extremes = true )
{
  auto const bounds =
    percentiles( image, std::vector< double >{ lower, upper },
                 sampling_points, ignore_extremes );

  auto const lower_bound = bounds.empty() ? In{ 0 } : bounds[ 0 ];
  auto const upper_bound = bounds.size() < 2 ? In{ 0 } : bounds[ 1 ];

  constexpr Out max_output = std::numeric_limits< Out >::max();

  double factor;

  if( upper_bound > lower_bound )
  {
    factor = ( pixel_max< Out >() + 0.5 ) /
             static_cast< double >( upper_bound - lower_bound );
  }
  else
  {
    factor = pixel_max< Out >() / pixel_max< In >();
  }

  kwiver::vital::image_of< Out > result( image.width(), image.height(),
                                         image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const value = image( i, j, plane );

        if( value < lower_bound )
        {
          result( i, j, plane ) = Out{ 0 };
        }
        else if( value > upper_bound )
        {
          result( i, j, plane ) = max_output;
        }
        else
        {
          result( i, j, plane ) = static_cast< Out >(
            static_cast< double >( value - lower_bound ) * factor );
        }
      }
    }
  }

  return result;
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_CONVERT_H
