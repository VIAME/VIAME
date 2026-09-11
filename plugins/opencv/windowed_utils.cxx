/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Chipping an image up for a windowed detector
 *
 * `cv::Mat` and `cv::resize` until P7-T04b; `vital::image` and
 * `image_ops::resize` since. Only the image handling changed -- the chip
 * geometry, the edge flags and the scale bookkeeping are as they were, and
 * `tests/golden/opencv`'s seven `ocv_windowed` cases hold them to it.
 */

#include "windowed_utils.h"

#include <image_ops/dispatch.h>
#include <image_ops/resample.h>
#include <image_ops/warp.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace kv = kwiver::vital;
namespace io = viame::image_ops;

namespace viame {

namespace {

// ----------------------------------------------------------------------------
/// `cv::resize` with a scale factor and no explicit size.
///
/// OpenCV computes the destination as `saturate_cast< int >( extent * scale )`,
/// which **rounds**, and interpolates bilinearly. Both matter: flooring gives
/// a chip one pixel narrower on about half of the scales.
kv::image
scaled( kv::image const& source, double scale )
{
  auto const width = static_cast< size_t >(
    std::lround( static_cast< double >( source.width() ) * scale ) );
  auto const height = static_cast< size_t >(
    std::lround( static_cast< double >( source.height() ) * scale ) );

  return io::dispatch_pixel_type(
    source,
    [ & ]( auto const& typed ) -> kv::image
    {
      return kv::image( io::resize( typed, width, height ) );
    } );
}

// ----------------------------------------------------------------------------
/// A zeroed image of \p width by \p height with \p source copied into its
/// top left, which is `Mat::setTo( 0 )` and a copy into a region of interest.
kv::image
padded( kv::image const& source, size_t width, size_t height )
{
  return io::dispatch_pixel_type(
    source,
    [ & ]( auto const& typed ) -> kv::image
    {
      using pixel_t = std::decay_t< decltype( typed( 0, 0, 0 ) ) >;

      kv::image_of< pixel_t > out( width, height, typed.depth() );

      for( size_t plane = 0; plane < out.depth(); ++plane )
      {
        for( size_t j = 0; j < height; ++j )
        {
          for( size_t i = 0; i < width; ++i )
          {
            out( i, j, plane ) = pixel_t{};
          }
        }
      }

      auto const copy_width = std::min( width, typed.width() );
      auto const copy_height = std::min( height, typed.height() );

      for( size_t plane = 0; plane < out.depth(); ++plane )
      {
        for( size_t j = 0; j < copy_height; ++j )
        {
          for( size_t i = 0; i < copy_width; ++i )
          {
            out( i, j, plane ) = typed( i, j, plane );
          }
        }
      }

      return kv::image( out );
    } );
}

// ----------------------------------------------------------------------------
kv::image
cropped( kv::image const& source, int x, int y, int width, int height )
{
  return io::dispatch_pixel_type(
    source,
    [ & ]( auto const& typed ) -> kv::image
    {
      return kv::image( io::crop( typed,
                                  static_cast< size_t >( x ),
                                  static_cast< size_t >( y ),
                                  static_cast< size_t >( width ),
                                  static_cast< size_t >( height ) ) );
    } );
}

} // namespace

// ----------------------------------------------------------------------------
kv::image
crop_region( const kv::image& image, const image_rect& rect )
{
  return cropped( image, rect.x, rect.y, rect.width, rect.height );
}

// ----------------------------------------------------------------------------
double
scale_image_maintaining_ar( const kv::image& src, kv::image& dst,
                            int width, int height, bool pad )
{
  double scale = 1.0;

  if( static_cast< int >( src.height() ) == height &&
      static_cast< int >( src.width() ) == width )
  {
    dst = src;
    return scale;
  }

  auto const original_height = static_cast< double >( src.height() );
  auto const original_width = static_cast< double >( src.width() );

  if( original_height > height )
  {
    scale = height / original_height;
  }
  if( original_width > width )
  {
    scale = std::min( scale, width / original_width );
  }

  auto const resized = scaled( src, scale );

  if( pad )
  {
    dst = padded( resized, static_cast< size_t >( width ),
                  static_cast< size_t >( height ) );
  }
  else
  {
    dst = resized;
  }

  return scale;
}

// ----------------------------------------------------------------------------
double
format_image( const kv::image& src, kv::image& dst, rescale_option option,
              double scale_factor, int width, int height, bool pad )
{
  double scale = 1.0;

  if( option == MAINTAIN_AR )
  {
    scale = scale_image_maintaining_ar( src, dst, width, height, pad );
  }
  else if( option == CHIP || option == SCALE ||
           option == CHIP_AND_ORIGINAL )
  {
    if( scale_factor == 1.0 )
    {
      dst = src;
    }
    else
    {
      dst = scaled( src, scale_factor );
      scale = scale_factor;
    }
  }
  else
  {
    rescale_option_converter conv;
    throw std::runtime_error(
      "Invalid resize option: " + conv.to_string( option ) );
  }

  return scale;
}

// ----------------------------------------------------------------------------
// `prepare_image_regions` was here.
//
// Upstream added `plugins/core/windowed_utils` -- an OpenCV-free windowed
// detector, refiner and trainer registered as `windowed` -- and it declares
// a `prepare_image_regions` with the **same signature in the same
// namespace** as this one. Two definitions of one symbol in two libraries:
// the loader binds whichever it sees first and every caller in both gets
// that one, which is `viame::enhance_images` again (finding 1.10).
//
// There is one definition now, core's, and it is the one the seven
// `ocv_windowed` golden cases have been reproducing all along -- exactly,
// including `black_pad` once the padding was fixed. The declaration comes in
// through `../core/windowed_utils.h`, which this header already includes.
//
// The other three helpers here do not collide: core's `scale_image_maintaining_ar`
// and `format_image` return the image and take the scale as an out parameter,
// and its `crop_image` aliases where `crop_region` copies, which the chip
// writer depends on. Collapsing the rest of the `ocv_windowed` and `windowed`
// duplication belongs to the restructuring, with the trainer recorded first.

} // end namespace viame
