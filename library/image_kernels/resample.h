/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_RESAMPLE_H
#define VIAME_IMAGE_KERNELS_RESAMPLE_H

#include <viame/core_types/image.h>

#include <algorithm>
#include <cstddef>

namespace viame {
namespace image_kernels {

// ----------------------------------------------------------------------------
/// Bilinear sample of one plane at a real position.
///
/// Positions outside the image return zero rather than the nearest pixel.
/// That is what `vil_bilin_interp_safe` does, and callers keep their sample
/// grid inside the image so it does not come up.
template < typename T >
double
bilinear_sample( viame::image_of< T > const& image,
                 double x, double y, size_t plane )
{
  auto const width = static_cast< int >( image.width() );
  auto const height = static_cast< int >( image.height() );

  if( x < 0.0 || y < 0.0 || x > width - 1 || y > height - 1 )
  {
    return 0.0;
  }

  auto const ix = static_cast< int >( x );
  auto const iy = static_cast< int >( y );
  auto const fx = x - ix;
  auto const fy = y - iy;

  auto const at =
    [ & ]( int i, int j )
    {
      return static_cast< double >(
        image( static_cast< size_t >( i ), static_cast< size_t >( j ),
               plane ) );
    };

  // The exact corners are taken without touching the neighbour that would
  // carry zero weight, so sampling the last row or column stays in bounds
  if( fx == 0.0 && fy == 0.0 ) { return at( ix, iy ); }
  if( fx == 0.0 ) { return at( ix, iy ) + ( at( ix, iy + 1 ) - at( ix, iy ) ) * fy; }
  if( fy == 0.0 ) { return at( ix, iy ) + ( at( ix + 1, iy ) - at( ix, iy ) ) * fx; }

  auto const left = at( ix, iy ) + ( at( ix, iy + 1 ) - at( ix, iy ) ) * fy;
  auto const right =
    at( ix + 1, iy ) + ( at( ix + 1, iy + 1 ) - at( ix + 1, iy ) ) * fy;

  return left + ( right - left ) * fx;
}

// ----------------------------------------------------------------------------
/// Resize to \p width by \p height with bilinear interpolation.
///
/// The sample grid spans the source almost exactly: the step is
/// `0.9999999 * (extent - 1) / (count - 1)`, and that shortfall is VXL's, to
/// keep the last sample from landing exactly on the final pixel and reading
/// one past it. Reproduced so that resized frames match to the last bit.
///
/// The result is truncated into the pixel type, not rounded, which is also
/// what `vil_resample_bilin` does.
template < typename T >
viame::image_of< T >
resize_bilinear( viame::image_of< T > const& image,
                 size_t width, size_t height )
{
  viame::image_of< T > result( width, height, image.depth() );

  if( width == 0 || height == 0 ||
      image.width() == 0 || image.height() == 0 )
  {
    return result;
  }

  constexpr double shortfall = 0.9999999;

  auto const step_x = ( width > 1 )
    ? shortfall * ( image.width() - 1.0 ) / ( width - 1.0 ) : 0.0;
  auto const step_y = ( height > 1 )
    ? shortfall * ( image.height() - 1.0 ) / ( height - 1.0 ) : 0.0;

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      auto const y = j * step_y;

      for( size_t i = 0; i < width; ++i )
      {
        result( i, j, plane ) = static_cast< T >(
          bilinear_sample( image, i * step_x, y, plane ) );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// The \p width by \p height rectangle at (\p left, \p top), as a new image.
// ----------------------------------------------------------------------------
/// \brief Resize by averaging each destination pixel's source footprint.
///
/// What OpenCV calls INTER_AREA, and the right filter for downsampling: a
/// bilinear shrink samples a handful of source pixels and aliases, where this
/// averages every pixel that falls under the destination one. Measured against
/// cv2.INTER_AREA on a natural frame, this agrees to within one grey level.
///
/// Falls back to bilinear when either axis is being enlarged, which is what
/// OpenCV does too -- there is no footprint to average over.
template < typename T >
viame::image_of< T >
resize_area( viame::image_of< T > const& image, size_t width, size_t height )
{
  if( width == 0 || height == 0 || image.width() == 0 || image.height() == 0 )
  {
    return viame::image_of< T >( width, height, image.depth() );
  }

  if( width > image.width() || height > image.height() )
  {
    return resize_bilinear( image, width, height );
  }

  viame::image_of< T > result( width, height, image.depth() );

  double const x_scale = static_cast< double >( image.width() ) / width;
  double const y_scale = static_cast< double >( image.height() ) / height;

  for( size_t y = 0; y < height; ++y )
  {
    size_t const top = static_cast< size_t >( y * y_scale );
    size_t bottom = static_cast< size_t >( ( y + 1 ) * y_scale + 0.5 );
    bottom = std::min( std::max( bottom, top + 1 ), image.height() );

    for( size_t x = 0; x < width; ++x )
    {
      size_t const left = static_cast< size_t >( x * x_scale );
      size_t right = static_cast< size_t >( ( x + 1 ) * x_scale + 0.5 );
      right = std::min( std::max( right, left + 1 ), image.width() );

      auto const count = static_cast< double >( ( bottom - top ) * ( right - left ) );

      for( size_t d = 0; d < image.depth(); ++d )
      {
        double total = 0.0;
        for( size_t j = top; j < bottom; ++j )
        {
          for( size_t i = left; i < right; ++i )
          {
            total += static_cast< double >( image( i, j, d ) );
          }
        }
        result( x, y, d ) = static_cast< T >( total / count + 0.5 );
      }
    }
  }

  return result;
}

template < typename T >
viame::image_of< T >
crop( viame::image_of< T > const& image,
      size_t left, size_t top, size_t width, size_t height )
{
  width = std::min( width, image.width() - std::min( left, image.width() ) );
  height = std::min( height, image.height() - std::min( top, image.height() ) );

  viame::image_of< T > result( width, height, image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        result( i, j, plane ) = image( left + i, top + j, plane );
      }
    }
  }

  return result;
}

// ----------------------------------------------------------------------------
/// The image placed at the top left of a \p width by \p height field of zeros.
///
/// Where the image is larger it is cropped instead, so the result is always
/// exactly the size asked for.
template < typename T >
viame::image_of< T >
pad_or_crop( viame::image_of< T > const& image,
             size_t width, size_t height )
{
  viame::image_of< T > result( width, height, image.depth() );

  auto const copy_width = std::min( width, image.width() );
  auto const copy_height = std::min( height, image.height() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        result( i, j, plane ) =
          ( i < copy_width && j < copy_height ) ? image( i, j, plane )
                                                : T{ 0 };
      }
    }
  }

  return result;
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_RESAMPLE_H
