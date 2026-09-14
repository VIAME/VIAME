/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of image warping onto another image
 *
 * `cv::warpPerspective` until P7-T04b; `image_ops::warp_perspective` since.
 * The bridge was asked for a `BGR_COLOR` mat on the way in and the result
 * was wrapped as one on the way out, so the two swaps cancelled and the warp
 * always ran on the vital image's own plane order.
 *
 * The two paths are the C++'s. With an alpha mask the warped image and the
 * destination are blended in float by the warped mask; without one, a
 * full-value plane is warped by nearest neighbour to say which pixels the
 * source covers, and only those are written. The second is not the same as
 * blending by a binary mask: it leaves the destination untouched where the
 * source does not reach, rather than multiplying it by one.
 */

#include "warp_image_ocv.h"

#include <image_ops/dispatch.h>
#include <image_ops/pixel.h>
#include <image_ops/warp.h>

#include <viame/core_types/image_container.h>

#include <algorithm>
#include <cmath>

namespace io = viame::image_ops;

namespace viame {

namespace kv = kwiver::vital;

namespace {

// ----------------------------------------------------------------------------
/// The value a full-scale sample of this type has, which is what an alpha
/// mask is divided by. `depth_max_value` in the C++, by pixel type rather
/// than by `cv::Mat` depth.
template < typename T >
double
full_scale()
{
  if constexpr( std::is_same< T, int8_t >::value ) { return 127.0; }
  else if constexpr( std::is_same< T, int16_t >::value ) { return 32767.0; }
  else if constexpr( std::is_integral< T >::value )
  {
    return static_cast< double >( io::pixel_max< T >() );
  }
  else { return 1.0; }
}

// ----------------------------------------------------------------------------
/// The alpha mask as a float weight in [0, 1], warped onto the destination.
kv::image_of< float >
warped_weight( kv::image const& mask, kv::matrix_3x3d const& transform,
               size_t width, size_t height )
{
  auto const weight = io::dispatch_pixel_type(
    mask,
    [ & ]( auto const& typed ) -> kv::image
    {
      using pixel_t = std::decay_t< decltype( typed( 0, 0, 0 ) ) >;

      kv::image_of< float > out( typed.width(), typed.height(), 1 );
      auto const scale = full_scale< pixel_t >();

      for( size_t j = 0; j < typed.height(); ++j )
      {
        for( size_t i = 0; i < typed.width(); ++i )
        {
          out( i, j, 0 ) =
            static_cast< float >( static_cast< double >( typed( i, j, 0 ) ) /
                                  scale );
        }
      }

      return kv::image( out );
    } );

  return io::warp_perspective( kv::image_of< float >( weight ), transform,
                               width, height );
}

} // namespace

// ----------------------------------------------------------------------------
/// Warp image
kv::image_container_sptr
warp_image_ocv
::warp( kv::image_container_sptr src_image,
        kv::image_container_sptr dst_image,
        kv::homography_sptr homography,
        kv::image_container_sptr alpha_mask ) const
{
  if( !src_image || !homography )
  {
    return dst_image;
  }

  auto const source = src_image->get_image();
  auto const transform = homography->matrix();

  auto const width = dst_image ? dst_image->width() : source.width();
  auto const height = dst_image ? dst_image->height() : source.height();

  auto const weight = alpha_mask
                      ? warped_weight( alpha_mask->get_image(), transform,
                                       width, height )
                      : kv::image_of< float >();

  auto const result = io::dispatch_pixel_type(
    source,
    [ & ]( auto const& typed ) -> kv::image
    {
      using pixel_t = std::decay_t< decltype( typed( 0, 0, 0 ) ) >;

      auto const warped = io::warp_perspective( typed, transform, width,
                                                height );

      kv::image_of< pixel_t > destination(
        width, height, warped.depth() );

      if( dst_image )
      {
        auto const given = kv::image_of< pixel_t >( dst_image->get_image() );

        for( size_t plane = 0; plane < destination.depth(); ++plane )
        {
          for( size_t j = 0; j < height; ++j )
          {
            for( size_t i = 0; i < width; ++i )
            {
              destination( i, j, plane ) =
                given( i, j, std::min( plane, given.depth() - 1 ) );
            }
          }
        }
      }
      else
      {
        // `cv::Mat::zeros` when there is no destination
        for( size_t plane = 0; plane < destination.depth(); ++plane )
        {
          for( size_t j = 0; j < height; ++j )
          {
            for( size_t i = 0; i < width; ++i )
            {
              destination( i, j, plane ) = pixel_t{};
            }
          }
        }
      }

      if( alpha_mask )
      {
        for( size_t plane = 0; plane < destination.depth(); ++plane )
        {
          for( size_t j = 0; j < height; ++j )
          {
            for( size_t i = 0; i < width; ++i )
            {
              auto const alpha =
                static_cast< double >( weight( i, j, 0 ) );

              auto const blended =
                static_cast< double >( warped( i, j, plane ) ) * alpha +
                static_cast< double >( destination( i, j, plane ) ) *
                  ( 1.0 - alpha );

              // `convertTo` back to the destination type, which rounds.
              destination( i, j, plane ) =
                io::saturate_pixel< pixel_t >( blended );
            }
          }
        }

        return kv::image( destination );
      }

      // No mask: a full-value plane warped by nearest neighbour says which
      // pixels the source reaches, and only those are written.
      kv::image_of< uint8_t > full( source.width(), source.height(), 1 );

      for( size_t j = 0; j < full.height(); ++j )
      {
        for( size_t i = 0; i < full.width(); ++i )
        {
          full( i, j, 0 ) = 255;
        }
      }

      auto const covered = io::warp_perspective(
        full, transform, width, height, io::interpolation::NEAREST );

      for( size_t plane = 0; plane < destination.depth(); ++plane )
      {
        for( size_t j = 0; j < height; ++j )
        {
          for( size_t i = 0; i < width; ++i )
          {
            if( covered( i, j, 0 ) )
            {
              destination( i, j, plane ) = warped( i, j, plane );
            }
          }
        }
      }

      return kv::image( destination );
    } );

  return std::make_shared< kv::simple_image_container >( result );
}

} // end namespace viame
