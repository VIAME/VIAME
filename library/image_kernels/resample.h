/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#ifndef VIAME_IMAGE_KERNELS_RESAMPLE_H
#define VIAME_IMAGE_KERNELS_RESAMPLE_H

#include <viame/core_types/image.h>
#include <viame/image_kernels/warp.h>
#include <viame/image_kernels/letterbox_plan.h>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <limits>

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
/// Resize using OpenCV's INTER_AREA grid and fractional pixel coverage.
/// The VXL bilinear grid above remains separate for its C++ callers.
template < typename T >
viame::image_of< T >
resize_area( viame::image_of< T > const& image, size_t width, size_t height )
{
  return resize( image, width, height, interpolation::AREA );
}

/// The width by height rectangle at (left, top), as a new image.
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

// Classifier preprocessing, sharing its coefficient plan with CUDA.
template <typename T>
viame::image_of< T >
resize_letterbox( viame::image_of< T > const& input, int width, int height )
{
  if( width < 1 || height < 1 || input.width() == 0 || input.height() == 0 )
    throw std::invalid_argument("letterbox dimensions must be positive");
  auto const plan = detail::make_letterbox_plan(input.width(), input.height(), width, height);
  auto const channels = input.depth();
  viame::image_of< T > scaled;
  if( plan.area || (plan.width == int(input.width()) && plan.height == int(input.height())) )
    scaled = resize_area(input, plan.width, plan.height);
  else
  {
    scaled = viame::image_of< T >(plan.width, plan.height, channels);
    using accumulator = std::conditional_t<std::is_same_v<T,uint8_t>,int32_t,float>;
    viame::image_of< accumulator > horizontal(plan.width, input.height(), channels);
    for( size_t y = 0; y < input.height(); ++y )
      for( int x = 0; x < plan.width; ++x )
        for( size_t c = 0; c < channels; ++c )
        {
          accumulator value = 0;
          for( int k = plan.x.offsets[x]; k < plan.x.offsets[x+1]; ++k )
          {
            auto const& tap = plan.x.entries[k];
            if constexpr(std::is_same_v<T,uint8_t>)
              value += int(input(tap.index,y,c))*tap.fixed;
            else
              value += float(input(tap.index,y,c))*tap.weight;
          }
          horizontal(x,y,c) = value;
        }
    for( int y = 0; y < plan.height; ++y )
      for( int x = 0; x < plan.width; ++x )
        for( size_t c = 0; c < channels; ++c )
        {
          if constexpr(std::is_same_v<T,uint8_t>)
          {
            int64_t value = 0;
            for( int k = plan.y.offsets[y]; k < plan.y.offsets[y+1]; ++k )
              value += int64_t(horizontal(x,plan.y.entries[k].index,c))*plan.y.entries[k].fixed;
            scaled(x,y,c) = static_cast<T>(std::clamp<int64_t>((value+(1<<21))>>22,0,255));
          }
          else
          {
            float value = 0;
            for( int k = plan.y.offsets[y]; k < plan.y.offsets[y+1]; ++k )
              value += horizontal(x,plan.y.entries[k].index,c)*plan.y.entries[k].weight;
            if constexpr(std::is_integral_v<T>)
              scaled(x,y,c) = static_cast<T>(std::clamp<long>(std::lrint(value),0,std::numeric_limits<T>::max()));
            else
              scaled(x,y,c) = value;
          }
        }
  }
  viame::image_of< T > result(width,height,channels);
  for( int y = 0; y < height; ++y )
    for( int x = 0; x < width; ++x )
      for( size_t c = 0; c < channels; ++c )
        result(x,y,c) = x >= plan.left && x < plan.left+plan.width &&
                         y >= plan.top && y < plan.top+plan.height
          ? scaled(x-plan.left,y-plan.top,c) : 0;
  return result;
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_RESAMPLE_H
