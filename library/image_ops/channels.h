/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_OPS_CHANNELS_H
#define VIAME_IMAGE_OPS_CHANNELS_H

#include <image_ops/pixel.h>

#include <viame/core_types/image.h>

#include <cstddef>

namespace viame {
namespace image_ops {

/// Luminance weights for RGB, from ITU-R BT.709, as VXL uses.
constexpr double red_weight = 0.2125;
constexpr double green_weight = 0.7154;
constexpr double blue_weight = 0.0721;

// ----------------------------------------------------------------------------
/// Weighted luminance of the first three planes.
template < typename T >
kwiver::vital::image_of< T >
planes_to_grey( kwiver::vital::image_of< T > const& image )
{
  kwiver::vital::image_of< T > result( image.width(), image.height(), 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const luminance =
        static_cast< double >( image( i, j, 0 ) ) * red_weight +
        static_cast< double >( image( i, j, 1 ) ) * green_weight +
        static_cast< double >( image( i, j, 2 ) ) * blue_weight;

      result( i, j, 0 ) = round_pixel< T >( luminance );
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Unweighted mean of every plane.
///
/// The accumulation is in the pixel type, matching
/// `vil_math_mean_over_planes`, so a narrow type wraps around rather than
/// widening: two uint8_t planes of 100 and 200 sum to 44, not 300, and
/// average to 22. Callers that want the obvious answer for three planes get
/// it from `planes_to_grey`, which accumulates in double.
template < typename T >
kwiver::vital::image_of< T >
mean_over_planes( kwiver::vital::image_of< T > const& image )
{
  kwiver::vital::image_of< T > result( image.width(), image.height(), 1 );

  auto const depth = image.depth();

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      T total = T{ 0 };

      for( size_t plane = 0; plane < depth; ++plane )
      {
        total = static_cast< T >( total + image( i, j, plane ) );
      }

      result( i, j, 0 ) = static_cast< T >( total / depth );
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Reduce an image to one plane.
///
/// Three planes are treated as RGB and take the luminance weights; any other
/// count is averaged flat. This is VXL's `combine_channels`.
template < typename T >
kwiver::vital::image_of< T >
combine_channels( kwiver::vital::image_of< T > const& image )
{
  if( image.depth() == 3 )
  {
    return planes_to_grey( image );
  }

  return mean_over_planes( image );
}

// ----------------------------------------------------------------------------
/// Grow an image to three planes, repeating plane 0 where there is nothing.
///
/// A four plane image keeps its first three; the alpha is dropped rather
/// than composited, which is what VXL's `force_three_channels` does.
template < typename T >
kwiver::vital::image_of< T >
force_three_channels( kwiver::vital::image_of< T > const& image )
{
  if( image.depth() == 3 )
  {
    return image;
  }

  kwiver::vital::image_of< T > result( image.width(), image.height(), 3 );

  auto const depth = image.depth();

  for( size_t plane = 0; plane < 3; ++plane )
  {
    auto const source = ( plane < depth ) ? plane : size_t{ 0 };

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        result( i, j, plane ) = image( i, j, source );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// Copy \p plane of \p image into every plane of a \p depth plane image.
template < typename T >
kwiver::vital::image_of< T >
broadcast_plane( kwiver::vital::image_of< T > const& image, size_t plane,
                 size_t depth )
{
  kwiver::vital::image_of< T > result( image.width(), image.height(), depth );

  for( size_t target = 0; target < depth; ++target )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        result( i, j, target ) = image( i, j, plane );
      }
    }
  }

  return result;
}

} // namespace image_ops
} // namespace viame

#endif // VIAME_IMAGE_OPS_CHANNELS_H
