/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Histograms, and the equalisations built on them
///
/// What `cv::calcHist`, `cv::minMaxLoc`, `cv::normalize`, `cv::equalizeHist`
/// and `cv::createCLAHE` did. CLAHE is the one that matters: `ocv_enhancer`
/// is selected by thirteen shipped pipelines and its `apply_clahe` path is
/// what most of them are for.
///
/// `stretch.h` already has the range stretches VXL's callers were tuned
/// against, and `statistics.h` the sampled percentiles; neither is a
/// histogram. What is here is the counted kind, which is what an
/// equalisation needs.

#ifndef VIAME_IMAGE_OPS_HISTOGRAM_H
#define VIAME_IMAGE_OPS_HISTOGRAM_H

#include <image_ops/filter.h>
#include <image_ops/pixel.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Where an extreme value is, as `cv::minMaxLoc` reports it.
struct extremum
{
  double value = 0.0;
  size_t i = 0;
  size_t j = 0;
  size_t plane = 0;
};

/// The smallest and largest value in \p image, and where each first occurs.
///
/// First in the order this walks -- plane, then row, then column -- which is
/// what `cv::minMaxLoc` reports for a single plane image and what any
/// tie-breaking rule has to fix on to be repeatable.
template < typename T >
void
min_max( kwiver::vital::image_of< T > const& image, extremum& lowest,
         extremum& highest )
{
  if( image.width() == 0 || image.height() == 0 || image.depth() == 0 )
  {
    throw std::invalid_argument( "min_max: the image has no pixels" );
  }

  lowest = extremum{ std::numeric_limits< double >::infinity(), 0, 0, 0 };
  highest = extremum{ -std::numeric_limits< double >::infinity(), 0, 0, 0 };

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const value = static_cast< double >( image( i, j, plane ) );

        if( value < lowest.value )
        {
          lowest = extremum{ value, i, j, plane };
        }

        if( value > highest.value )
        {
          highest = extremum{ value, i, j, plane };
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
/// The count in each of \p bins between \p low and \p high, over one plane.
///
/// Half-open bins, as `cv::calcHist` uses: a value equal to \p high falls in
/// the last bin rather than off the end, and everything outside the range is
/// dropped.
template < typename T >
std::vector< size_t >
histogram( kwiver::vital::image_of< T > const& image, size_t bins,
           double low, double high, size_t plane = 0 )
{
  if( bins == 0 )
  {
    throw std::invalid_argument( "histogram: no bins" );
  }

  if( !( high > low ) )
  {
    throw std::invalid_argument( "histogram: the range is empty" );
  }

  if( plane >= image.depth() )
  {
    throw std::invalid_argument( "histogram: no such plane" );
  }

  std::vector< size_t > counts( bins, 0 );

  auto const scale = static_cast< double >( bins ) / ( high - low );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const value = static_cast< double >( image( i, j, plane ) );

      if( value < low || value > high )
      {
        continue;
      }

      auto index = static_cast< size_t >( ( value - low ) * scale );

      if( index >= bins )
      {
        index = bins - 1;
      }

      ++counts[ index ];
    }
  }

  return counts;
}

/// The histogram of an integer plane over the whole range of its type.
///
/// The common case: 256 bins for a byte, one value each, which is what an
/// equalisation wants and what `cv::equalizeHist` builds.
template < typename T >
std::vector< size_t >
histogram_full( kwiver::vital::image_of< T > const& image, size_t plane = 0 )
{
  static_assert( std::is_integral< T >::value,
                 "histogram_full is for integer pixels; give a range" );

  auto const bins = static_cast< size_t >( pixel_max< T >() ) + 1;
  return histogram( image, bins, 0.0, pixel_max< T >() + 1.0, plane );
}

// ----------------------------------------------------------------------------
/// Rescale so the values span [\p low, \p high], which is `cv::NORM_MINMAX`.
///
/// A flat image has no range to stretch and comes back at \p low, which is
/// what OpenCV does with it.
template < typename T >
kwiver::vital::image_of< T >
normalize_min_max( kwiver::vital::image_of< T > const& image, double low,
                   double high )
{
  extremum lowest;
  extremum highest;
  min_max( image, lowest, highest );

  auto const span = highest.value - lowest.value;

  kwiver::vital::image_of< T > out( image.width(), image.height(),
                                    image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const value = static_cast< double >( image( i, j, plane ) );

        out( i, j, plane ) = saturate_pixel< T >(
          ( span > 0.0 )
            ? low + ( value - lowest.value ) * ( high - low ) / span
            : low );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Flatten the histogram of one plane, which is `cv::equalizeHist`.
///
/// The mapping is OpenCV's, and it is not the textbook one. The textbook
/// scales the cumulative count by `(levels - 1) / total`; OpenCV divides by
/// `total - count_of_the_first_occupied_bin` and rounds, so that the darkest
/// occupied value maps to zero exactly. On an image with a large flat
/// background the two differ across the whole range, not at the ends.
template < typename T >
kwiver::vital::image_of< T >
equalize( kwiver::vital::image_of< T > const& image )
{
  static_assert( std::is_integral< T >::value,
                 "equalize is for integer pixels" );

  auto const levels = static_cast< size_t >( pixel_max< T >() ) + 1;

  kwiver::vital::image_of< T > out( image.width(), image.height(),
                                    image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    auto const counts = histogram_full( image, plane );

    // The first occupied bin, whose count OpenCV takes out of the total
    size_t first = 0;
    while( first < levels && counts[ first ] == 0 )
    {
      ++first;
    }

    auto const total = image.width() * image.height();

    if( first == levels || counts[ first ] == total )
    {
      // Empty, or every pixel the same value: nothing to flatten
      for( size_t j = 0; j < image.height(); ++j )
      {
        for( size_t i = 0; i < image.width(); ++i )
        {
          out( i, j, plane ) = image( i, j, plane );
        }
      }
      continue;
    }

    auto const scale = static_cast< double >( levels - 1 ) /
                       static_cast< double >( total - counts[ first ] );

    std::vector< T > mapping( levels, 0 );
    size_t running = 0;

    for( size_t level = first; level < levels; ++level )
    {
      running += counts[ level ];

      // The first occupied level maps to zero: its own count is the part
      // taken out of both the running sum and the total
      auto const above =
        static_cast< double >( running - counts[ first ] );

      // Half to even, as OpenCV's saturate_cast is: a cumulative count
      // scaled to a byte lands on an exact half constantly
      mapping[ level ] = saturate_pixel_even< T >( above * scale );
    }

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        out( i, j, plane ) =
          mapping[ static_cast< size_t >( image( i, j, plane ) ) ];
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Contrast limited adaptive histogram equalisation, `cv::createCLAHE`.
///
/// The image is divided into \p tiles_x by \p tiles_y tiles, each is
/// equalised on its own histogram with the counts above \p clip_limit
/// redistributed evenly, and the per-tile mappings are interpolated
/// bilinearly between tile centres so the tile edges do not show.
///
/// The clip limit is OpenCV's: a multiple of the average bin count rather
/// than an absolute count, so the same setting means the same thing whatever
/// the tile size. `ocv_enhancer` passes 3 and 20 in the shipped pipelines.
///
/// @param image one plane, integer
/// @param clip_limit multiples of the average bin count; zero or less means
///        no clipping
/// @param tiles_x how many tiles across
/// @param tiles_y how many tiles down
template < typename T >
kwiver::vital::image_of< T >
clahe( kwiver::vital::image_of< T > const& image, double clip_limit = 40.0,
       size_t tiles_x = 8, size_t tiles_y = 8 )
{
  static_assert( std::is_integral< T >::value,
                 "clahe is for integer pixels" );

  if( image.depth() != 1 )
  {
    throw std::invalid_argument( "clahe takes a single plane" );
  }

  if( tiles_x == 0 || tiles_y == 0 )
  {
    throw std::invalid_argument( "clahe: no tiles" );
  }

  auto const width = image.width();
  auto const height = image.height();

  if( width == 0 || height == 0 )
  {
    throw std::invalid_argument( "clahe: the image has no pixels" );
  }

  auto const levels = static_cast< size_t >( pixel_max< T >() ) + 1;

  // OpenCV pads the image up to a whole number of tiles rather than letting
  // the last tile be a different size, so every tile has the same area and
  // the clip limit means the same thing in each.
  //
  // Its padding rule is reproduced exactly, oddities included: it pads when
  // *either* dimension fails to divide, and then pads *both* by
  // `tiles - (extent % tiles)` -- which for a dimension that already divides
  // is a whole extra tile. A 16 by 12 image with an 8 by 8 grid therefore
  // becomes 24 by 16 and the tiles are 3 by 2 rather than 2 by 2. Rounding
  // up instead, which is the obvious thing, gives a different answer on any
  // size that does not divide evenly -- which real frames rarely do.
  auto const padded_w = ( width % tiles_x == 0 && height % tiles_y == 0 )
    ? width
    : width + tiles_x - ( width % tiles_x );
  auto const padded_h = ( width % tiles_x == 0 && height % tiles_y == 0 )
    ? height
    : height + tiles_y - ( height % tiles_y );

  auto const tile_w = padded_w / tiles_x;
  auto const tile_h = padded_h / tiles_y;
  auto const tile_area = tile_w * tile_h;

  auto const limit = ( clip_limit > 0.0 )
    ? std::max< size_t >(
        1, static_cast< size_t >( clip_limit *
                                  static_cast< double >( tile_area ) /
                                  static_cast< double >( levels ) ) )
    : 0;

  // One mapping per tile
  std::vector< std::vector< T > > mappings( tiles_x * tiles_y );

  for( size_t ty = 0; ty < tiles_y; ++ty )
  {
    for( size_t tx = 0; tx < tiles_x; ++tx )
    {
      std::vector< size_t > counts( levels, 0 );

      for( size_t j = 0; j < tile_h; ++j )
      {
        for( size_t i = 0; i < tile_w; ++i )
        {
          // The padding is BORDER_REFLECT_101, which is what
          // `copyMakeBorder` gets from OpenCV's CLAHE
          auto const value = sample_with_border(
            image, static_cast< long >( tx * tile_w + i ),
            static_cast< long >( ty * tile_h + j ), 0,
            border_mode::REFLECT_101 );

          ++counts[ static_cast< size_t >( value ) ];
        }
      }

      if( limit > 0 )
      {
        // Clip, and give what was clipped back evenly. OpenCV does one
        // redistribution pass and then tops up the remainder by walking the
        // bins, rather than iterating to a fixed point.
        size_t clipped = 0;

        for( auto& count : counts )
        {
          if( count > limit )
          {
            clipped += count - limit;
            count = limit;
          }
        }

        auto const share = clipped / levels;
        auto const remainder = clipped - share * levels;

        for( auto& count : counts )
        {
          count += share;
        }

        if( remainder > 0 )
        {
          auto const step = std::max< size_t >( 1, levels / remainder );
          size_t given = 0;

          for( size_t level = 0; level < levels && given < remainder;
               level += step )
          {
            ++counts[ level ];
            ++given;
          }
        }
      }

      std::vector< T > mapping( levels, 0 );
      size_t running = 0;

      auto const scale = static_cast< double >( levels - 1 ) /
                         static_cast< double >( tile_area );

      for( size_t level = 0; level < levels; ++level )
      {
        running += counts[ level ];
        mapping[ level ] = saturate_pixel_even< T >(
          static_cast< double >( running ) * scale );
      }

      mappings[ ty * tiles_x + tx ] = std::move( mapping );
    }
  }

  // Interpolate between the four surrounding tile centres
  kwiver::vital::image_of< T > out( width, height, 1 );

  auto const half_w = static_cast< double >( tile_w ) / 2.0;
  auto const half_h = static_cast< double >( tile_h ) / 2.0;

  for( size_t j = 0; j < height; ++j )
  {
    auto const ty = ( static_cast< double >( j ) - half_h ) /
                    static_cast< double >( tile_h );
    auto const top = static_cast< long >( std::floor( ty ) );
    auto const fy = ty - static_cast< double >( top );

    auto const y0 = static_cast< size_t >(
      std::max( 0L, std::min( static_cast< long >( tiles_y ) - 1, top ) ) );
    auto const y1 = static_cast< size_t >(
      std::max( 0L, std::min( static_cast< long >( tiles_y ) - 1,
                              top + 1 ) ) );

    for( size_t i = 0; i < width; ++i )
    {
      auto const tx = ( static_cast< double >( i ) - half_w ) /
                      static_cast< double >( tile_w );
      auto const left = static_cast< long >( std::floor( tx ) );
      auto const fx = tx - static_cast< double >( left );

      auto const x0 = static_cast< size_t >(
        std::max( 0L, std::min( static_cast< long >( tiles_x ) - 1, left ) ) );
      auto const x1 = static_cast< size_t >(
        std::max( 0L, std::min( static_cast< long >( tiles_x ) - 1,
                                left + 1 ) ) );

      auto const level = static_cast< size_t >( image( i, j, 0 ) );

      auto const a =
        static_cast< double >( mappings[ y0 * tiles_x + x0 ][ level ] );
      auto const b =
        static_cast< double >( mappings[ y0 * tiles_x + x1 ][ level ] );
      auto const c =
        static_cast< double >( mappings[ y1 * tiles_x + x0 ][ level ] );
      auto const d =
        static_cast< double >( mappings[ y1 * tiles_x + x1 ][ level ] );

      out( i, j, 0 ) = saturate_pixel< T >(
        a * ( 1.0 - fx ) * ( 1.0 - fy ) + b * fx * ( 1.0 - fy ) +
        c * ( 1.0 - fx ) * fy + d * fx * fy );
    }
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
