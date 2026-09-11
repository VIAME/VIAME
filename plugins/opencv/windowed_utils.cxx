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
void
prepare_image_regions(
  const kv::image& image,
  const window_settings& settings,
  std::vector< kv::image >& regions_to_process,
  std::vector< windowed_region_prop >& region_properties )
{
  regions_to_process.clear();
  region_properties.clear();

  auto const rows = static_cast< int >( image.height() );
  auto const cols = static_cast< int >( image.width() );

  rescale_option mode = settings.mode;

  if( mode == ADAPTIVE )
  {
    if( ( rows * cols ) >= settings.chip_adaptive_thresh )
    {
      mode = CHIP_AND_ORIGINAL;
    }
    else if( settings.original_to_chip_size )
    {
      mode = MAINTAIN_AR;
    }
    else
    {
      mode = DISABLED;
    }
  }

  kv::image resized_image;
  double scale_factor = 1.0;

  if( mode != DISABLED )
  {
    scale_factor = format_image( image, resized_image,
      ( mode == ORIGINAL_AND_RESIZED ? SCALE : mode ),
      settings.scale, settings.chip_width, settings.chip_height );
  }
  else
  {
    resized_image = image;
  }

  image_rect original_dims( 0, 0, cols, rows );

  if( mode == ORIGINAL_AND_RESIZED )
  {
    kv::image scaled_original;

    if( rows <= settings.chip_height && cols <= settings.chip_width )
    {
      regions_to_process.push_back( image );
      region_properties.push_back(
        windowed_region_prop( original_dims, 1.0 ) );
    }
    else
    {
      if( ( rows * cols ) >= settings.chip_adaptive_thresh )
      {
        regions_to_process.push_back( resized_image );
        region_properties.push_back(
          windowed_region_prop( original_dims, 1.0 / scale_factor ) );
      }

      double scaled_original_scale = scale_image_maintaining_ar( image,
        scaled_original, settings.chip_width, settings.chip_height,
        settings.black_pad );

      regions_to_process.push_back( scaled_original );
      region_properties.push_back(
        windowed_region_prop( original_dims, 1.0 / scaled_original_scale ) );
    }
  }
  else if( mode != CHIP && mode != CHIP_AND_ORIGINAL )
  {
    regions_to_process.push_back( resized_image );
    region_properties.push_back(
      windowed_region_prop( original_dims, 1.0 / scale_factor ) );
  }
  else
  {
    auto const resized_cols = static_cast< int >( resized_image.width() );
    auto const resized_rows = static_cast< int >( resized_image.height() );

    // Chip up scaled image
    for( int li = 0;
         li < resized_cols - settings.chip_width + settings.chip_step_width;
         li += settings.chip_step_width )
    {
      int ti = std::min( li + settings.chip_width, resized_cols );

      for( int lj = 0;
           lj < resized_rows - settings.chip_height + settings.chip_step_height;
           lj += settings.chip_step_height )
      {
        int tj = std::min( lj + settings.chip_height, resized_rows );

        if( tj - lj < 0 || ti - li < 0 )
        {
          continue;
        }

        image_rect original_roi(
          static_cast< int >( li / scale_factor ),
          static_cast< int >( lj / scale_factor ),
          static_cast< int >( ( ti - li ) / scale_factor ),
          static_cast< int >( ( tj - lj ) / scale_factor ) );

        auto const cropped_chip =
          cropped( resized_image, li, lj, ti - li, tj - lj );

        kv::image scaled_crop;

        double scaled_crop_scale = scale_image_maintaining_ar(
          cropped_chip, scaled_crop, settings.chip_width,
          settings.chip_height, settings.black_pad );

        regions_to_process.push_back( scaled_crop );

        region_properties.push_back(
          windowed_region_prop( original_roi,
            settings.chip_edge_filter,
            ( li + settings.chip_step_width ) >=
              ( resized_cols - settings.chip_width +
                settings.chip_step_width ),
            ( lj + settings.chip_step_height ) >=
              ( resized_rows - settings.chip_height +
                settings.chip_step_height ),
            1.0 / scaled_crop_scale,
            li, lj,
            1.0 / scale_factor ) );
      }
    }

    // Extract full sized image chip if enabled
    if( mode == CHIP_AND_ORIGINAL )
    {
      kv::image scaled_original;

      if( settings.original_to_chip_size )
      {
        double scaled_original_scale = scale_image_maintaining_ar( image,
          scaled_original, settings.chip_width, settings.chip_height,
          settings.black_pad );

        regions_to_process.push_back( scaled_original );
        region_properties.push_back(
          windowed_region_prop( original_dims,
                                1.0 / scaled_original_scale ) );
      }
      else
      {
        regions_to_process.push_back( image );
        region_properties.push_back(
          windowed_region_prop( original_dims, 1.0 ) );
      }
    }
  }
}

} // end namespace viame
