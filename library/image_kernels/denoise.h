/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Non-local means denoising, as `cv::fastNlMeansDenoising` does it
///
/// Every weight here is a fixed-point integer and every distance an integer
/// sum, so unlike the float filters in `filter.h` this reproduces OpenCV
/// exactly rather than to within a rounding. Three details carry it:
///
/// * the distance between two template windows is quantised before it reaches
///   the weight table -- `dist >> ceil(log2(area))` -- and the table is built
///   over those quantised distances rather than over real ones, so the weight
///   is a step function and the steps have to land in the same places;
/// * a weight below a thousandth of the fixed-point scale is forced to **zero**
///   rather than kept small, which changes which neighbours contribute at all;
/// * the border is `BORDER_DEFAULT`, which is reflect-101, and extends by
///   `search/2 + template/2` rather than by either alone.
///
/// The coloured form is not this applied three times. `cv::cvtColor` is called
/// with **`COLOR_LBGR2Lab`**, not `COLOR_BGR2Lab`: the input is taken as linear
/// light, so the sRGB transfer curve is skipped on the way in and on the way
/// out. Denoising in the sRGB-coded space instead is wrong everywhere the
/// curve is steep, which is the whole of the shadows.

#ifndef VIAME_IMAGE_KERNELS_DENOISE_H
#define VIAME_IMAGE_KERNELS_DENOISE_H

#include <image_kernels/color.h>
#include <image_kernels/filter.h>
#include <image_kernels/pixel.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

namespace detail {

/// The smallest power of two at or above \p value, as an exponent.
inline int
nearest_power_of_two( int value )
{
  auto power = 0;

  while( ( 1 << power ) < value ) { ++power; }

  return power;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// `cv::fastNlMeansDenoising` with `NORM_L2`, for one to three planes.
///
/// \p strength is OpenCV's `h`. \p patch and \p window are its
/// `templateWindowSize` and `searchWindowSize`; both are forced odd the way it
/// forces them, by halving and doubling back.
inline viame::image_of<uint8_t>
denoise_non_local_means_serial ( viame::image_of<uint8_t> const &image, double strength,
                                 int patch = 7, int window = 21 )
{
  auto const planes = static_cast< int >( image.depth() );

  if( planes < 1 || planes > 3 )
  {
    throw std::invalid_argument(
      "denoise_non_local_means takes one to three planes" );
  }

  auto const patch_half = patch / 2;
  auto const window_half = window / 2;
  auto const patch_size = patch_half * 2 + 1;
  auto const window_size = window_half * 2 + 1;
  auto const border = window_half + patch_half;

  auto const width = static_cast< int >( image.width() );
  auto const height = static_cast< int >( image.height() );

  viame::image_of< uint8_t > out(
    image.width(), image.height(), image.depth() );

  if( width == 0 || height == 0 )
  {
    return out;
  }

  // The fixed-point scale is whatever keeps the weighted sum inside an int.
  auto const ceiling =
    static_cast< int64_t >( window_size ) * window_size * 255;
  auto const scale = static_cast< int64_t >(
    std::numeric_limits< int32_t >::max() / ceiling );

  auto const area = patch_size * patch_size;
  auto const shift = detail::nearest_power_of_two( area );
  auto const step = static_cast< double >( 1 << shift ) / area;
  auto const furthest = 255 * 255 * planes;
  auto const levels = static_cast< int >( furthest / step ) + 1;

  std::vector< int64_t > weight( static_cast< size_t >( levels ) );

  for( int level = 0; level < levels; ++level )
  {
    auto const distance = level * step;
    auto value = ( strength == 0.0 )
                 ? ( distance == 0.0 ? 1.0 : 0.0 )
                 : std::exp( -distance / ( strength * strength * planes ) );

    if( std::isnan( value ) ) { value = 1.0; }

    auto found = static_cast< int64_t >(
      std::nearbyint( static_cast< double >( scale ) * value ) );

    if( static_cast< double >( found ) < 0.001 * static_cast< double >( scale ) )
    {
      found = 0;
    }

    weight[ static_cast< size_t >( level ) ] = found;
  }

  // The extended image, reflect-101 on every side.
  auto const wide = width + 2 * border;
  auto const tall = height + 2 * border;
  std::vector< int > extended(
    static_cast< size_t >( wide ) * tall * planes );

  for( int y = 0; y < tall; ++y )
  {
    auto const sy = detail::border_index( y - border, height,
                                          border_mode::REFLECT_101 );

    for( int x = 0; x < wide; ++x )
    {
      auto const sx = detail::border_index( x - border, width,
                                            border_mode::REFLECT_101 );

      for( int plane = 0; plane < planes; ++plane )
      {
        extended[ ( static_cast< size_t >( y ) * wide + x ) * planes + plane ] =
          static_cast< int >( image( static_cast< size_t >( sx ),
                                     static_cast< size_t >( sy ),
                                     static_cast< size_t >( plane ) ) );
      }
    }
  }

  auto const at = [ & ]( int y, int x, int plane ) -> int
  {
    return extended[ ( static_cast< size_t >( y ) * wide + x ) * planes +
                     plane ];
  };

  // For each search displacement, maintain the vertical patch sums and slide
  // their horizontal sum. Each squared pixel difference is evaluated only
  // twice per output row, rather than patch_size^2 times per output pixel.
  auto const pixels = static_cast< size_t >( width ) * height;
  std::vector< int64_t > estimate( pixels * planes, 0 );
  std::vector< int64_t > total( pixels, 0 );
  auto const columns = width + 2 * patch_half;
  std::vector< int > vertical( static_cast< size_t >( columns ) );

  for( int dy = -window_half; dy <= window_half; ++dy )
  {
    for( int dx = -window_half; dx <= window_half; ++dx )
    {
      auto const squared = [ & ]( int y, int x )
      {
        auto distance = 0;
        for( int plane = 0; plane < planes; ++plane )
        {
          auto const gap = at( y, x, plane ) - at( y + dy, x + dx, plane );
          distance += gap * gap;
        }
        return distance;
      };
      std::fill( vertical.begin(), vertical.end(), 0 );
      for( int y = -patch_half; y <= patch_half; ++y )
      {
        for( int x = 0; x < columns; ++x )
        {
          vertical[ x ] += squared( border + y, border - patch_half + x );
        }
      }
      for( int y = 0; y < height; ++y )
      {
        auto distance = 0;
        for( int x = 0; x < patch_size; ++x ) { distance += vertical[ x ]; }
        for( int x = 0; x < width; ++x )
        {
          auto const found = weight[ static_cast< size_t >(
            std::min( distance >> shift, levels - 1 ) ) ];
          auto const pixel = static_cast< size_t >( y ) * width + x;
          total[ pixel ] += found;
          for( int plane = 0; plane < planes; ++plane )
          {
            estimate[ pixel * planes + plane ] +=
              found * at( border + y + dy, border + x + dx, plane );
          }
          if( x + 1 < width )
          {
            distance += vertical[ x + patch_size ] - vertical[ x ];
          }
        }
        if( y + 1 < height )
        {
          for( int x = 0; x < columns; ++x )
          {
            auto const sx = border - patch_half + x;
            vertical[ x ] += squared( border + y + patch_half + 1, sx ) -
                              squared( border + y - patch_half, sx );
          }
        }
      }
    }
  }
  for( int y = 0; y < height; ++y )
  {
    for( int x = 0; x < width; ++x )
    {
      auto const pixel = static_cast< size_t >( y ) * width + x;
      for( int plane = 0; plane < planes; ++plane )
      {
        auto const value =
          ( estimate[ pixel * planes + plane ] + total[ pixel ] / 2 ) /
          total[ pixel ];
        out( x, y, plane ) = static_cast< uint8_t >( value );
      }
    }
  }

  return out;
}

// Each output stripe includes the complete patch/search halo. Only its own
// output rows are copied back, preserving the full image's border semantics.
inline viame::image_of<uint8_t>
denoise_non_local_means ( viame::image_of<uint8_t> const &image, double strength,
                          int patch = 7, int window = 21 )
{
  if ( image.height () < 128 || kernel_thread_count () == 1 || patch < 1 || window < 1 )
    return denoise_non_local_means_serial ( image, strength, patch, window );
  auto const halo = static_cast<std::size_t> ( patch / 2 + window / 2 );
  viame::image_of<uint8_t> out ( image.width (), image.height (), image.depth () );
  parallel_rows ( 0, image.height (), 64,
                  [&] ( std::size_t begin, std::size_t end )
                  {
                    auto const top = begin > halo ? begin - halo : 0;
                    auto const bottom = std::min ( image.height (), end + halo );
                    viame::image_of<uint8_t> part (
                        image.first_pixel () + static_cast<ptrdiff_t>(top) * image.h_step (), image.width (),
                        bottom - top, image.depth (), image.w_step (), image.h_step (),
                        image.d_step () );
                    auto const filtered =
                        denoise_non_local_means_serial ( part, strength, patch, window );
                    for ( auto y = begin; y < end; ++y )
                      for ( std::size_t plane = 0; plane < image.depth (); ++plane )
                        for ( std::size_t x = 0; x < image.width (); ++x )
                          out ( x, y, plane ) = filtered ( x, y - top, plane );
                  } );
  return out;
}

// ----------------------------------------------------------------------------
/// `cv::fastNlMeansDenoisingColored`: L and the chroma pair, separately.
///
/// The image goes to L*a*b* through the **linear** transfer -- OpenCV asks
/// `cvtColor` for `COLOR_LBGR2Lab` -- L is denoised at \p strength and the two
/// chroma planes together at \p colour_strength, and the result comes back the
/// same way.
inline viame::image_of< uint8_t >
denoise_non_local_means_colour( viame::image_of< uint8_t > const& image,
                                double strength, double colour_strength,
                                int patch = 7, int window = 21 )
{
  detail::require_planes( image, 3, "denoise_non_local_means_colour" );

  auto const width = image.width();
  auto const height = image.height();

  auto const lab = rgb_to_lab( image, true );

  viame::image_of< uint8_t > lightness( width, height, 1 );
  viame::image_of< uint8_t > chroma( width, height, 2 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      lightness( i, j, 0 ) = lab( i, j, 0 );
      chroma( i, j, 0 ) = lab( i, j, 1 );
      chroma( i, j, 1 ) = lab( i, j, 2 );
    }
  }

  auto const clean_lightness =
    denoise_non_local_means( lightness, strength, patch, window );
  auto const clean_chroma =
    denoise_non_local_means( chroma, colour_strength, patch, window );

  viame::image_of< uint8_t > merged( width, height, 3 );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      merged( i, j, 0 ) = clean_lightness( i, j, 0 );
      merged( i, j, 1 ) = clean_chroma( i, j, 0 );
      merged( i, j, 2 ) = clean_chroma( i, j, 1 );
    }
  }

  return lab_to_rgb( merged, true );
}

} // namespace image_kernels
} // namespace viame

#endif
