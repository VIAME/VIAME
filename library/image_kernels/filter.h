/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Convolution and the separable filters built on it
///
/// What `cv::filter2D`, `cv::GaussianBlur`, `cv::blur`, `cv::Sobel` and
/// `cv::addWeighted` did. Every one of them is a weighted sum of a
/// neighbourhood, so there is one kernel underneath and the rest are the
/// weights plus a border rule.
///
/// The border rules are OpenCV's, spelled its way, because a caller porting
/// a `cv::` call has a `BORDER_` constant in front of it and the two have to
/// line up. `REFLECT_101` is OpenCV's default and is what an unspecified
/// border means there.

#ifndef VIAME_IMAGE_KERNELS_FILTER_H
#define VIAME_IMAGE_KERNELS_FILTER_H

#include <image_kernels/pixel.h>
#include <image_kernels/gaussian_kernel.h>
#include <image_kernels/gaussian_simd.h>
#include <image_kernels/parallel.h>

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

/// Optional scratch storage for successive float Gaussian filters. One workspace
/// per concurrent caller; output images own their pixels independently of it.
struct gaussian_workspace
{
  std::vector<float> padded, across, taps;
  std::vector<std::size_t> rows;
};

// ----------------------------------------------------------------------------
/// What a filter reads where the image stops.
///
/// Named after OpenCV's `BORDER_` constants, and meaning the same thing:
///
/// * `CONSTANT`     -- a fixed value, zero unless one is given
/// * `REPLICATE`    -- the edge pixel, repeated:      aaaaa|abcde|eeeee
/// * `REFLECT`      -- mirrored including the edge:   edcba|abcde|edcba
/// * `REFLECT_101`  -- mirrored excluding the edge:   dcba|abcde|dcba
/// * `WRAP`         -- the far side, tiled:           bcdea|abcde|abcde
///
/// `REFLECT_101` is OpenCV's default and the one a `cv::` call gets when it
/// says nothing, which is why it is the default here.
enum class border_mode
{
  CONSTANT,
  REPLICATE,
  REFLECT,
  REFLECT_101,
  WRAP,
};

namespace detail {

/// A coordinate brought inside [0, extent) by the border rule.
///
/// Returns -1 for `CONSTANT`, which is the caller's signal to use the
/// constant instead of reading anything.
inline long
border_index( long at, long extent, border_mode mode )
{
  if( at >= 0 && at < extent )
  {
    return at;
  }

  if( extent <= 0 )
  {
    return -1;
  }

  switch( mode )
  {
    case border_mode::CONSTANT:
      return -1;

    case border_mode::REPLICATE:
      return std::max( 0L, std::min( extent - 1, at ) );

    case border_mode::REFLECT:
      // The edge pixel appears twice: ...b a | a b c | c b...
      while( at < 0 || at >= extent )
      {
        if( at < 0 ) { at = -at - 1; }
        if( at >= extent ) { at = 2 * extent - at - 1; }
      }
      return at;

    case border_mode::WRAP:
      // Tiled, which is `cv::BORDER_WRAP`. The remainder is made
      // non-negative first: C++'s `%` keeps the sign of the dividend, so
      // -1 % 5 is -1 rather than the 4 that is wanted here.
      at %= extent;
      return at < 0 ? at + extent : at;

    case border_mode::REFLECT_101:
    default:
      // The edge pixel appears once: ...c b | a b c | b a...
      if( extent == 1 )
      {
        return 0;
      }

      while( at < 0 || at >= extent )
      {
        if( at < 0 ) { at = -at; }
        if( at >= extent ) { at = 2 * extent - at - 2; }
      }
      return at;
  }
}

} // namespace detail

// ----------------------------------------------------------------------------
/// The pixel at (\p i, \p j) of \p plane, with the border rule applied.
template < typename T >
double
sample_with_border( viame::image_of< T > const& image, long i, long j,
                    size_t plane, border_mode mode, double constant = 0.0 )
{
  auto const x = detail::border_index(
    i, static_cast< long >( image.width() ), mode );
  auto const y = detail::border_index(
    j, static_cast< long >( image.height() ), mode );

  if( x < 0 || y < 0 )
  {
    return constant;
  }

  return static_cast< double >(
    image( static_cast< size_t >( x ), static_cast< size_t >( y ), plane ) );
}

// ----------------------------------------------------------------------------
/// A two dimensional convolution kernel, in row order.
///
/// Stored rather than passed as a raw pointer so that the width and height
/// travel with it, and because every builder below returns one.
struct kernel
{
  size_t width = 0;
  size_t height = 0;
  std::vector< double > weights;

  double
  at( size_t i, size_t j ) const
  {
    return weights[ j * width + i ];
  }

  /// The kernel's sum, which is what a normalising builder divides by.
  double
  sum() const
  {
    double total = 0.0;
    for( auto const weight : weights ) { total += weight; }
    return total;
  }
};

// ----------------------------------------------------------------------------
/// Correlate \p image with \p k, which is what `cv::filter2D` computes.
///
/// Correlation, not convolution: OpenCV's `filter2D` does not flip the
/// kernel, and neither does this. For the symmetric kernels below the two
/// are the same; for an asymmetric one -- a Sobel, say -- they differ in
/// sign, and the sign is what a gradient means.
///
/// The anchor is the kernel's centre, as OpenCV's default anchor is. The
/// result is saturated into \p T, so a gradient on an unsigned type clips at
/// zero; a caller wanting the sign asks for a signed or floating result
/// through \p Out.
///
/// @param image the image, any number of planes; each is filtered alone
/// @param k the weights
/// @param mode what to read past the edge
/// @param constant the value for `CONSTANT`
template < typename Out, typename T >
viame::image_of< Out >
filter_2d( viame::image_of< T > const& image, kernel const& k,
           border_mode mode = border_mode::REFLECT_101,
           double constant = 0.0 )
{
  if( k.width == 0 || k.height == 0 ||
      k.weights.size() != k.width * k.height )
  {
    throw std::invalid_argument( "filter_2d: the kernel has no shape" );
  }

  auto const anchor_i = static_cast< long >( k.width / 2 );
  auto const anchor_j = static_cast< long >( k.height / 2 );

  viame::image_of< Out > out( image.width(), image.height(),
                                      image.depth() );

  // Whether a pixel's whole footprint is inside the image, which for all but
  // a rim of the kernel's own radius it is. The taps are visited in the same
  // order and the zero ones skipped the same way on both paths, so the answer
  // is identical to the last bit; what the inside path drops is
  // `sample_with_border`, which dispatches on the border rule **per tap**.
  // That dispatch, not the arithmetic, is what a kernel built on this costs:
  // it was 0.219 s of the 0.245 it took to find a thousand corners in a
  // 1080p frame, and the same shape of cost showed up twice more in the
  // optical flow before it was looked for rather than guessed at.
  auto const inside_i = static_cast< long >( k.width ) - 1 - anchor_i;
  auto const inside_j = static_cast< long >( k.height ) - 1 - anchor_j;

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      auto const row_inside =
        static_cast< long >( j ) >= anchor_j &&
        static_cast< long >( j ) + inside_j < static_cast< long >( image.height() );

      for( size_t i = 0; i < image.width(); ++i )
      {
        double total = 0.0;

        auto const all_inside = row_inside &&
          static_cast< long >( i ) >= anchor_i &&
          static_cast< long >( i ) + inside_i < static_cast< long >( image.width() );

        if( all_inside )
        {
          for( size_t kj = 0; kj < k.height; ++kj )
          {
            for( size_t ki = 0; ki < k.width; ++ki )
            {
              auto const weight = k.at( ki, kj );

              if( weight == 0.0 )
              {
                continue;
              }

              total += weight *
                static_cast< double >( image(
                  static_cast< size_t >( static_cast< long >( i ) +
                    static_cast< long >( ki ) - anchor_i ),
                  static_cast< size_t >( static_cast< long >( j ) +
                    static_cast< long >( kj ) - anchor_j ),
                  plane ) );
            }
          }
        }
        else
        {
          for( size_t kj = 0; kj < k.height; ++kj )
          {
            for( size_t ki = 0; ki < k.width; ++ki )
            {
              auto const weight = k.at( ki, kj );

              if( weight == 0.0 )
              {
                continue;
              }

              total += weight * sample_with_border(
                image,
                static_cast< long >( i ) + static_cast< long >( ki ) - anchor_i,
                static_cast< long >( j ) + static_cast< long >( kj ) - anchor_j,
                plane, mode, constant );
            }
          }
        }

        out( i, j, plane ) = saturate_pixel< Out >( total );
      }
    }
  }

  return out;
}

/// `filter_2d` keeping the input's pixel type.
template < typename T >
viame::image_of< T >
filter_2d( viame::image_of< T > const& image, kernel const& k,
           border_mode mode = border_mode::REFLECT_101,
           double constant = 0.0 )
{
  return filter_2d< T, T >( image, k, mode, constant );
}

// ----------------------------------------------------------------------------
/// The outer product of two one dimensional kernels.
///
/// A separable filter applied as one two dimensional pass. Slower than two
/// passes for a large kernel and simpler for a small one, and the kernels
/// here are three to nine wide.
inline kernel
separable_kernel( std::vector< double > const& horizontal,
                  std::vector< double > const& vertical )
{
  kernel k;
  k.width = horizontal.size();
  k.height = vertical.size();
  k.weights.resize( k.width * k.height );

  for( size_t j = 0; j < k.height; ++j )
  {
    for( size_t i = 0; i < k.width; ++i )
    {
      k.weights[ j * k.width + i ] = horizontal[ i ] * vertical[ j ];
    }
  }

  return k;
}

// ----------------------------------------------------------------------------
/// Correlate \p image with the outer product of \p down and \p across, in
/// two passes.
///
/// The same answer as `filter_2d( image, separable_kernel( across, down ) )`
/// and a fraction of the work: a square pass over an N by N kernel does N^2
/// weighted samples a pixel where two passes do 2N. At N of 17 -- which is
/// what the coarsest pyramid level of the optical flow asks for -- that is
/// 289 against 34, and it is the difference between a 1080p blur taking
/// 1.9 s and a tenth of that.
///
/// It is not *guaranteed* to be bit for bit what the square pass gives, since
/// that one multiplies the two weights together before touching the pixel and
/// this one does not, and floating point multiplication is not associative.
/// Measured over three image sizes, four kernel widths and both a byte and a
/// float pixel: identical everywhere, which is why `gaussian_blur` and
/// `box_blur` below are allowed to use it. Anything else that moves onto it
/// should check the same way rather than assume.
template < typename Out, typename T >
viame::image_of< Out >
separable_filter( viame::image_of< T > const& image,
                  std::vector< double > const& across,
                  std::vector< double > const& down,
                  border_mode mode = border_mode::REFLECT_101,
                  double constant = 0.0 )
{
  if( across.empty() || down.empty() )
  {
    throw std::invalid_argument( "separable_filter: a kernel has no shape" );
  }

  auto const anchor_i = static_cast< long >( across.size() / 2 );
  auto const anchor_j = static_cast< long >( down.size() / 2 );

  auto const width = image.width();
  auto const height = image.height();

  viame::image_of< Out > out( width, height, image.depth() );

  // What a row wholly outside the image contributes, which only `CONSTANT`
  // ever has: every sample of it is the constant, so its horizontal pass is
  // the constant times the row of weights.
  double outside = 0.0;

  for( auto const weight : across ) { outside += weight * constant; }

  std::vector< double > buffer( width * height );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      auto* destination = buffer.data() + j * width;

      std::fill( destination, destination + width, 0.0 );
      auto const* source = image.first_pixel() + j * image.h_step() + plane * image.d_step();
      for( size_t k = 0; k < across.size(); ++k )
      {
        double const weight = across[k];
        if( weight == 0.0 ) { continue; }
        long const shift = static_cast< long >( k ) - anchor_i;
        size_t const first = std::min( width, static_cast< size_t >( std::max( 0L, -shift ) ) );
        size_t const last = static_cast< size_t >( std::max( static_cast< long >( first ),
          std::min( static_cast< long >( width ), static_cast< long >( width ) - shift ) ) );
        // Only the edges need border handling. Traverse pixels in the inner
        // loop so the compiler can vectorize each weighted row addition.
        for( size_t i = 0; i < first; ++i )
        {
          destination[i] += weight * sample_with_border(
            image, static_cast< long >( i ) + shift, j, plane, mode, constant );
        }
        for( size_t i = first; i < last; ++i )
        { destination[i] += weight * source[( static_cast< ptrdiff_t >( i ) + shift ) * image.w_step()]; }
        for( size_t i = last; i < width; ++i )
        {
          destination[i] += weight * sample_with_border(
            image, static_cast< long >( i ) + shift, j, plane, mode, constant );
        }
      }
    }

    std::vector< double const* > rows( down.size(), nullptr );
    std::vector< double > totals( width );

    for( size_t j = 0; j < height; ++j )
    {
      for( size_t k = 0; k < down.size(); ++k )
      {
        auto const at = detail::border_index(
          static_cast< long >( j ) + static_cast< long >( k ) - anchor_j,
          static_cast< long >( height ), mode );

        rows[ k ] = at < 0
          ? nullptr
          : buffer.data() + static_cast< size_t >( at ) * width;
      }

      std::fill( totals.begin(), totals.end(), 0.0 );
      for( size_t k = 0; k < down.size(); ++k )
      {
        double const weight = down[k];
        if( weight == 0.0 ) { continue; }
        if( rows[k] )
        {
          for( size_t i = 0; i < width; ++i )
          { totals[i] += weight * rows[k][i]; }
        }
        else
        {
          for( size_t i = 0; i < width; ++i ) { totals[i] += weight * outside; }
        }
      }
      auto* destination = out.first_pixel() + j * out.h_step() + plane * out.d_step();
      for( size_t i = 0; i < width; ++i )
      { destination[i] = saturate_pixel< Out >( totals[i] ); }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
namespace detail {

/// How wide OpenCV's fixed point Gaussian is for each pixel type.
///
/// An 8 bit image is filtered with a Q8.8 kernel, a row buffer of Q8.8 in a
/// uint16 and a column accumulator of Q16.16 in a uint32 -- `ufixedpoint16`
/// and its `WT`. A 16 bit image doubles all three: Q16.16, uint32, and
/// Q32.32 in a uint64. Nothing else has a fixed point path, and a float
/// image is filtered in float.
template < typename T > struct fixed_blur;

template <> struct fixed_blur< uint8_t >
{
  static constexpr int shift = 8;
  using row_type = uint16_t;
  using column_type = uint32_t;
};

template <> struct fixed_blur< uint16_t >
{
  static constexpr int shift = 16;
  using row_type = uint32_t;
  using column_type = uint64_t;
};

/// OpenCV's fixed point kernel, which sums to exactly `1 << shift`.
///
/// Rounding each tap independently does not: a 7-tap sigma 1.5 kernel comes to
/// 253 rather than 256, and a blur three parts in 256 dark is three counts
/// dark at the top of the range. OpenCV rounds the **running total** and takes
/// differences, so the sum is right by construction and the rounding error is
/// spread along the kernel rather than parked on one tap. Putting the
/// shortfall on the centre tap instead is close but not the same: it leaves
/// two counts on 209 pixels of a 20 by 28 frame.
///
/// OpenCV writes this as error diffusion -- `getGaussianKernelFixedPoint_ED`
/// carries the residue from tap to tap -- which is the same arithmetic:
/// diffusing the residue *is* differencing the rounded running total.
inline std::vector< int64_t >
gaussian_kernel_fixed( std::vector< double > const& line, int shift )
{
  auto const one = static_cast< int64_t >( 1 ) << shift;

  // The line arrives normalised and is not normalised again: OpenCV diffuses
  // over `kernel_bitexact` as `getGaussianKernelBitExact` left it, and a
  // second division by a sum that is 0.9999999999999999 rather than 1 moves
  // a Q16.16 tap.
  std::vector< int64_t > raw( line.size() );
  auto running = 0.0;
  int64_t placed = 0;

  for( size_t i = 0; i < line.size(); ++i )
  {
    running += line[ i ];

    auto const edge = static_cast< int64_t >(
      std::nearbyint( running * static_cast< double >( one ) ) );

    raw[ i ] = edge - placed;
    placed = edge;
  }

  return raw;
}

/// `cv::GaussianBlur`'s float path, in OpenCV's associations.
///
/// A float image is filtered in float32, and *which* float32 matters: the
/// answer depends on the order the taps are combined, and OpenCV's order is
/// not the obvious one. Three shapes, picked by kernel size, all of them
/// fused:
///
/// * **across, any size above five** (`RowVec_32f`): the centre-most tap is a
///   plain multiply and the rest accumulate in tap order,
///   `s = fma( src[k], k[k], s )`. Left to right, not symmetric pairs;
/// * **across, size three or five** (`SymmRowSmallVec_32f`): symmetric pairs,
///   innermost first -- size five is
///   `fma( s2 + s-2, k2, fma( s0, k0, ( s-1 + s1 ) * k1 ) )`, where the k1
///   term is a plain multiply and the other two are fused;
/// * **down** (`SymmColumnVec_32f`, or its size-three form): the centre tap
///   first, then **each symmetric pair summed before one fused multiply**,
///   `s = fma( row[k] + row[-k], k[k], s )`.
///
/// Reproducing all three takes the gap against cv2 from 4.6e-05 -- which
/// finding 2.44 recorded as irreproducible accumulation order -- to **zero**,
/// on every pixel the vectorised body covers. What is left is the same
/// remainder as 2.56's: the last `width % lanes` columns of each row go
/// through OpenCV's scalar loop, whose association is a third one, and they
/// differ by about 1e-07. Zero when the width is a multiple of the lane count
/// -- 8 floats with AVX2 -- and the kernels cannot chase it further without
/// baking the host's vector width in.
inline viame::image_of<float>
gaussian_blur_float_fast ( viame::image_of<float> const &image,
                           std::vector<double> const &line, border_mode mode,
                           gaussian_workspace &work )
{
  auto const width = image.width (), height = image.height (), n = line.size ();
  auto const half = n / 2, padded_width = width + 2 * half;
  viame::image_of<float> out ( width, height, image.depth () );
  if ( !width || !height || !image.depth () )
  {
    return out;
  }
  work.padded.resize ( padded_width * height );
  work.across.resize ( width * height );
  work.taps.assign ( line.begin (), line.end () );
  work.rows.resize ( height * n );
  for ( std::size_t y = 0; y < height; ++y )
    for ( std::size_t k = 0; k < n; ++k )
      work.rows[y * n + k] = border_index (
          static_cast<long> ( y ) + static_cast<long> ( k ) - static_cast<long> ( half ),
          height, mode );
  auto const grain =
      std::max<std::size_t> ( 1, 32768 / std::max<std::size_t> ( 1, width * n ) );
  for ( std::size_t plane = 0; plane < image.depth (); ++plane )
  {
    parallel_rows (
        0, height, grain,
        [&] ( std::size_t begin, std::size_t end )
        {
          for ( auto y = begin; y < end; ++y )
          {
            auto *padded = work.padded.data () + y * padded_width;
            auto const *input =
                image.first_pixel () + static_cast<ptrdiff_t>(y) * image.h_step () + static_cast<ptrdiff_t>(plane) * image.d_step ();
            for ( std::size_t x = 0; x < width; ++x )
            {
              padded[half + x] = input[static_cast<ptrdiff_t>(x) * image.w_step ()];
            }
            for ( std::size_t x = 0; x < half; ++x )
            {
              padded[x] = input[border_index ( static_cast<long> ( x ) -
                                                   static_cast<long> ( half ),
                                               width, mode ) *
                                image.w_step ()];
              padded[half + width + x] =
                  input[border_index ( width + x, width, mode ) * image.w_step ()];
            }
            gaussian_row ( padded, work.across.data () + y * width, width,
                           work.taps.data (), n );
          }
        } );
    parallel_rows (
        0, height, grain,
        [&] ( std::size_t begin, std::size_t end )
        {
          for ( auto y = begin; y < end; ++y )
          {
            // image_of owns planar storage, so each output plane is contiguous.
            auto *output = out.first_pixel () + y * out.h_step () + plane * out.d_step ();
            gaussian_column ( work.across.data (), width, work.rows.data () + y * n,
                              output, work.taps.data (), n );
          }
        } );
  }
  return out;
}

template <typename T>
viame::image_of<T>
gaussian_blur_float ( viame::image_of<T> const &image, std::vector<double> const &line,
                      border_mode mode, gaussian_workspace *workspace = nullptr )
{
  if constexpr ( std::is_same<T, float>::value )
  {
    gaussian_workspace local;
    return gaussian_blur_float_fast ( image, line, mode, workspace ? *workspace : local );
  }
  constexpr double constant = 0.0;

  auto const n = line.size();
  auto const half = static_cast< long >( n / 2 );

  auto const width = image.width();
  auto const height = image.height();
  auto const planes = image.depth();

  viame::image_of< T > out( width, height, planes );

  if( width == 0 || height == 0 || planes == 0 )
  {
    return out;
  }

  std::vector< float > tap( n );

  for( size_t k = 0; k < n; ++k )
  {
    tap[ k ] = static_cast< float >( line[ k ] );
  }

  // The centred half, as OpenCV indexes it: `centre[ 0 ]` is the middle tap.
  auto const centre = [ & ]( long offset ) -> float
  {
    return tap[ static_cast< size_t >( half + offset ) ];
  };

  std::vector< float > across( width * height );

  for( size_t plane = 0; plane < planes; ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        auto const at = [ & ]( long offset ) -> float
        {
          return static_cast< float >( sample_with_border(
            image, static_cast< long >( i ) + offset,
            static_cast< long >( j ), plane, mode, constant ) );
        };

        float sum;

        if( n == 1 )
        {
          sum = at( 0 ) * centre( 0 );
        }
        else if( n == 3 )
        {
          sum = std::fma( at( 0 ), centre( 0 ),
                          ( at( -1 ) + at( 1 ) ) * centre( 1 ) );
        }
        else if( n == 5 )
        {
          sum = std::fma( at( 2 ) + at( -2 ), centre( 2 ),
                          std::fma( at( 0 ), centre( 0 ),
                                    ( at( -1 ) + at( 1 ) ) * centre( 1 ) ) );
        }
        else
        {
          sum = at( -half ) * tap[ 0 ];

          for( size_t k = 1; k < n; ++k )
          {
            sum = std::fma( at( static_cast< long >( k ) - half ), tap[ k ],
                            sum );
          }
        }

        across[ j * width + i ] = sum;
      }
    }

    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        auto const at = [ & ]( long offset ) -> float
        {
          auto const row = border_index(
            static_cast< long >( j ) + offset,
            static_cast< long >( height ), mode );

          // Never negative: the caller keeps `CONSTANT` on the general
          // path, since OpenCV's float filter has no constant to take.
          return across[ static_cast< size_t >( row ) * width + i ];
        };

        auto sum = std::fma( at( 0 ), centre( 0 ), 0.0f );

        for( long k = 1; k <= half; ++k )
        {
          sum = std::fma( at( k ) + at( -k ), centre( k ), sum );
        }

        out( i, j, plane ) = static_cast< T >( sum );
      }
    }
  }

  return out;
}

/// `cv::GaussianBlur`'s integer path: one fixed point width across, twice
/// that down.
///
/// The row pass accumulates `kernel * pixel` in the narrow type, which is why
/// the products and the sums both saturate there -- for a byte, 256 times 255
/// only just fits a uint16. The column pass multiplies two of those into the
/// double width type and the pixel comes off the top with a half added, which
/// is `ufixedpoint16::saturate_cast` and `ufixedpoint32`'s in OpenCV's
/// `fixedpoint.inl.hpp`.
template < typename T >
viame::image_of< T >
gaussian_blur_fixed( viame::image_of< T > const& image,
                     std::vector< double > const& line, border_mode mode )
{
  using traits = fixed_blur< T >;
  using row_type = typename traits::row_type;
  using column_type = typename traits::column_type;

  constexpr int shift = traits::shift;
  constexpr auto row_ceiling =
    static_cast< column_type >( std::numeric_limits< row_type >::max() );
  constexpr auto half = static_cast< column_type >( 1 )
                        << ( 2 * shift - 1 );

  auto const raw = gaussian_kernel_fixed( line, shift );
  auto const n = raw.size();
  auto const anchor = static_cast< long >( n / 2 );

  auto const width = image.width();
  auto const height = image.height();
  auto const planes = image.depth();

  viame::image_of< T > out( width, height, planes );

  if( width == 0 || height == 0 || planes == 0 )
  {
    return out;
  }

  std::vector< row_type > across( width * height );

  for( size_t plane = 0; plane < planes; ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        column_type sum = 0;

        for( size_t k = 0; k < n; ++k )
        {
          auto const sample = sample_with_border(
            image, static_cast< long >( i ) + static_cast< long >( k ) - anchor,
            static_cast< long >( j ), plane, mode, 0.0 );

          auto const product = std::min< column_type >(
            static_cast< column_type >( raw[ k ] ) *
            static_cast< column_type >( sample ), row_ceiling );

          sum = std::min< column_type >( sum + product, row_ceiling );
        }

        across[ j * width + i ] = static_cast< row_type >( sum );
      }
    }

    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        column_type sum = 0;

        for( size_t k = 0; k < n; ++k )
        {
          auto const row = border_index(
            static_cast< long >( j ) + static_cast< long >( k ) - anchor,
            static_cast< long >( height ), mode );

          sum += static_cast< column_type >( raw[ k ] ) *
                 static_cast< column_type >( across[
                   static_cast< size_t >( row ) * width + i ] );
        }

        out( i, j, plane ) = static_cast< T >( std::min< column_type >(
          ( sum + half ) >> ( 2 * shift ),
          static_cast< column_type >( std::numeric_limits< T >::max() ) ) );
      }
    }
  }

  return out;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// A Gaussian blur, which is `cv::GaussianBlur`.
///
/// @param image the image
/// @param size the kernel width and height, odd
/// @param sigma the standard deviation, derived from the size when not given
// ----------------------------------------------------------------------------
/// `cv::GaussianBlur` on a byte image when its input is a **submatrix**.
///
/// Not a variation on `gaussian_blur`: a different computation, and this is
/// not a subtlety worth a count. `cv::GaussianBlur`'s bit-exact fixed-point
/// path is guarded by
///
///     sdepth == CV_8U && ((borderType & BORDER_ISOLATED) || !isSubmatrix())
///
/// so a caller passing a region of a larger buffer and a border rule that
/// reaches outside it falls through to `sepFilter2D` with the **float**
/// kernel instead. The two disagree by a count on about a fifth of the
/// pixels, because the fixed-point kernel is built by error diffusion with
/// its centre tap forced so the taps sum to exactly 256, and `sepFilter2D`
/// rounds each float tap on its own -- for size 7 and sigma 2 that is
/// 18, 34, 49, 55, 49, 34, 18, which sums to 257.
///
/// ORB is the caller that needs this: it blurs each pyramid level in place
/// inside one packed buffer, with `BORDER_REFLECT_101` and no
/// `BORDER_ISOLATED`, and its descriptor is a comparison of single pixels,
/// so a count decides a bit.
///
/// The accumulation is **float32** in both passes and its shape is not free.
/// `RowVec_8u32f` runs a fused multiply-add chain from zero in kernel order;
/// `SymmColumnVec_32f8u` multiplies the centre tap, then folds each
/// symmetric pair with **one** fused multiply-add of their sum. Both the
/// pairing and the fusion are load-bearing: unfused, or with the taps taken
/// in order, the result lands a single unit in the last place away, which is
/// enough to move a value that sits exactly on a half. One pixel of one
/// pyramid level was the whole of the difference when this was first written
/// the obvious way.
///
/// **`sepFilter2D` does not reproduce itself on an exact half.** Its vector
/// body rounds one to even, through `v_round`, and its scalar remainder
/// rounds one away from zero, so the answer for the last `width % lanes`
/// columns depends on the vector width -- 32 with AVX2, 16 with SSE2. This
/// takes the vector body, as `hsv_to_rgb` and the float blur do, for the
/// same reason: it is what a real frame's interior goes through, and the
/// alternative is not portable between hosts.
///
/// It only shows for a kernel whose taps are dyadic, which is what
/// `sigma == 0` gives for sizes up to nine -- 1/4, 1/2, 1/4 and its
/// relatives. Those make the whole float32 accumulation exact, so a half
/// lands on a half; for any other sigma the sum misses by some bits and the
/// rounding is never in question. ORB's blur is size 7 at sigma 2, which is
/// the second kind: no exact half turned up anywhere in its verification.
///
/// @param image the image
/// @param size the kernel width and height, odd
/// @param sigma the standard deviation, derived from the size when not given
template < typename T >
viame::image_of< T >
gaussian_blur_float_taps( viame::image_of< T > const& image, size_t size,
                          double sigma = 0.0,
                          border_mode mode = border_mode::REFLECT_101 )
{
  auto const exact = gaussian_kernel_1d( size, sigma );

  if( exact.size() % 2 == 0 )
  {
    throw std::invalid_argument(
      "gaussian_blur_float_taps: the kernel has to be odd, since the column "
      "pass folds it about its centre" );
  }

  // `getGaussianKernel( n, sigma, CV_32F )` is the same kernel narrowed to
  // float, one tap at a time.
  std::vector< float > line( exact.size() );
  for( size_t k = 0; k < exact.size(); ++k )
  { line[k] = static_cast< float >( exact[k] ); }

  auto const taps = line.size();
  auto const half = static_cast< long >( taps / 2 );
  auto const width = image.width();
  auto const height = image.height();

  viame::image_of< T > out( width, height, image.depth() );
  std::vector< float > buffer( width * height );

  // Where each tap reads, once rather than per pixel.
  std::vector< long > across( taps * width );
  std::vector< long > down( taps * height );
  for( size_t k = 0; k < taps; ++k )
  {
    for( size_t i = 0; i < width; ++i )
    {
      across[ k * width + i ] = detail::border_index(
        static_cast< long >( i ) + static_cast< long >( k ) - half,
        static_cast< long >( width ), mode );
    }
    for( size_t j = 0; j < height; ++j )
    {
      down[ k * height + j ] = detail::border_index(
        static_cast< long >( j ) + static_cast< long >( k ) - half,
        static_cast< long >( height ), mode );
    }
  }

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      auto* destination = buffer.data() + j * width;
      for( size_t i = 0; i < width; ++i )
      {
        float total = 0.0f;
        for( size_t k = 0; k < taps; ++k )
        {
          auto const at = across[ k * width + i ];
          auto const value = at < 0
            ? 0.0f
            : static_cast< float >(
                image( static_cast< size_t >( at ), j, plane ) );
          total = std::fma( line[k], value, total );
        }
        destination[i] = total;
      }
    }

    auto const row = [&]( size_t k, size_t j ) -> float const*
    {
      auto const at = down[ k * height + j ];
      return at < 0 ? nullptr
                    : buffer.data() + static_cast< size_t >( at ) * width;
    };

    for( size_t j = 0; j < height; ++j )
    {
      auto const* centre = row( static_cast< size_t >( half ), j );
      for( size_t i = 0; i < width; ++i )
      {
        float total = centre ? line[ half ] * centre[i] : 0.0f;
        for( long k = 1; k <= half; ++k )
        {
          auto const* above = row( static_cast< size_t >( half + k ), j );
          auto const* below = row( static_cast< size_t >( half - k ), j );
          auto const pair = ( above ? above[i] : 0.0f ) +
                            ( below ? below[i] : 0.0f );
          total = std::fma( line[ half + k ], pair, total );
        }
        // `saturate_cast< uchar >( float )` is `cvRound`, which is half to
        // even; rounding half away is a count out wherever the accumulation
        // lands on an exact half, and it does.
        out( i, j, plane ) = saturate_pixel_even< T >( total );
      }
    }
  }

  return out;
}

template <typename T>
viame::image_of<T> gaussian_blur ( viame::image_of<T> const &image, size_t size,
                                   double sigma = 0.0,
                                   border_mode mode = border_mode::REFLECT_101,
                                   gaussian_workspace *workspace = nullptr )
{
  auto const line = gaussian_kernel_1d( size, sigma );

  // `cv::GaussianBlur` does not filter an integer image in floating point. It
  // converts the kernel to fixed point and runs two integer passes, and the
  // answers differ: identical where the kernel is dyadic -- which the
  // sigma-derived small kernels are -- and a count apart on about a fifth of
  // the pixels for any other sigma. `detail::gaussian_blur_fixed` is that
  // path, for 8 and for 16 bit alike. The 16 bit one is easy to miss, since
  // the dispatch for it sits in a separate branch of `smooth.dispatch.cpp`
  // and float is where a reasonable reading would expect it to land: it does
  // not, and taking it for float leaves a count on 8 percent of a 16 bit
  // frame however carefully the float accumulation is ordered.
  //
  // Not for a constant border: OpenCV's fixed-point row filter simply omits
  // the taps that fall outside, which is a constant of zero and nothing else,
  // where this kernel takes the constant as an argument.
  if constexpr( std::is_same< T, uint8_t >::value ||
                std::is_same< T, uint16_t >::value )
  {
    if( mode != border_mode::CONSTANT )
    {
      return detail::gaussian_blur_fixed( image, line, mode );
    }
  }
  else if constexpr( std::is_floating_point< T >::value )
  {
    // And a float image is not filtered in double. `gaussian_blur_float`
    // says which float32 associations OpenCV uses and what it costs to get
    // them wrong.
    if( mode != border_mode::CONSTANT )
    {
      return detail::gaussian_blur_float ( image, line, mode, workspace );
    }
  }

  return separable_filter< T, T >( image, line, line, mode );
}

// ----------------------------------------------------------------------------
/// `cv::medianBlur`: the median of a \p size by \p size window.
///
/// Exact by construction rather than by measurement, which is unusual here: an
/// odd window holds an odd number of samples, so its median is a single sample
/// and there is no rounding or tie to get wrong. Any correct implementation
/// agrees with OpenCV's, whichever of its several it dispatched to.
///
/// The border replicates, which is what `medianBlur` does and does not let the
/// caller change.
template < typename T >
viame::image_of< T >
median_blur( viame::image_of< T > const& image, size_t size )
{
  if( size < 3 || size % 2 == 0 )
  {
    throw std::invalid_argument(
      "median_blur: the size has to be odd and at least 3" );
  }

  auto const width = image.width();
  auto const height = image.height();
  auto const planes = image.depth();
  auto const half = static_cast< long >( size / 2 );

  viame::image_of< T > out( width, height, planes );

  if( width == 0 || height == 0 )
  {
    return out;
  }

  std::vector< T > window( size * size );

  for( size_t plane = 0; plane < planes; ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        auto at = size_t{ 0 };

        for( long dj = -half; dj <= half; ++dj )
        {
          for( long di = -half; di <= half; ++di )
          {
            auto const y = detail::border_index(
              static_cast< long >( j ) + dj, static_cast< long >( height ),
              border_mode::REPLICATE );
            auto const x = detail::border_index(
              static_cast< long >( i ) + di, static_cast< long >( width ),
              border_mode::REPLICATE );

            window[ at++ ] = image( static_cast< size_t >( x ),
                                    static_cast< size_t >( y ), plane );
          }
        }

        auto const middle = window.begin() +
                            static_cast< ptrdiff_t >( window.size() / 2 );

        std::nth_element( window.begin(), middle, window.end() );

        out( i, j, plane ) = *middle;
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// A box blur, which is `cv::blur`: the mean of a \p size by \p size window.
template < typename T >
viame::image_of< T >
box_blur( viame::image_of< T > const& image, size_t size,
          border_mode mode = border_mode::REFLECT_101, size_t height = 0 )
{
  if( height == 0 )
  {
    height = size;
  }

  if( size == 0 || height == 0 )
  {
    throw std::invalid_argument( "box_blur: the size has to be positive" );
  }

  // Separable, and the two sides need not match: a rectangular window is a
  // mean over `size` columns and then over `height` rows.
  std::vector< double > const across(
    size, 1.0 / static_cast< double >( size ) );
  std::vector< double > const down(
    height, 1.0 / static_cast< double >( height ) );

  return separable_filter< T, T >( image, across, down, mode );
}

// ----------------------------------------------------------------------------
/// The Sobel derivative kernels, as `cv::getDerivKernels` builds them.
///
/// \p order is 1 for a first derivative and 2 for a second; \p size is 1, 3,
/// 5 or 7 and gives the smoothing across the gradient. Size 1 is the plain
/// difference with no smoothing, which is what OpenCV's `ksize=1` means.
inline std::vector< double >
sobel_kernel_1d( int order, size_t size )
{
  if( order < 0 || order > 2 )
  {
    throw std::invalid_argument( "sobel_kernel_1d: order is 0, 1 or 2" );
  }

  if( size != 1 && size != 3 && size != 5 && size != 7 )
  {
    throw std::invalid_argument( "sobel_kernel_1d: size is 1, 3, 5 or 7" );
  }

  if( size == 1 )
  {
    if( order == 0 ) { return { 1.0 }; }
    if( order == 1 ) { return { -1.0, 0.0, 1.0 }; }
    return { 1.0, -2.0, 1.0 };
  }

  // OpenCV's construction, which is not "a binomial, differenced": it is a
  // binomial of length `size - order` convolved with `order` copies of the
  // difference (-1, 1). Each convolution *lengthens* the row by one, so the
  // result is `size` long however the order and size are split.
  //
  // That is what makes a Sobel of size 3 come out (-1, 0, 1) rather than
  // (2, 0, -2), and of size 5 come out (-1, -2, 0, 2, 1). Differencing the
  // length-`size` binomial instead gives a row that is too short and scaled
  // wrong, which is the mistake this comment exists to stop.
  std::vector< double > row{ 1.0 };

  for( size_t step = 1; step + static_cast< size_t >( order ) < size; ++step )
  {
    std::vector< double > next( row.size() + 1, 0.0 );

    for( size_t i = 0; i < row.size(); ++i )
    {
      next[ i ] += row[ i ];
      next[ i + 1 ] += row[ i ];
    }

    row = next;
  }

  for( int taken = 0; taken < order; ++taken )
  {
    std::vector< double > next( row.size() + 1, 0.0 );

    for( size_t i = 0; i < next.size(); ++i )
    {
      auto const before = ( i > 0 ) ? row[ i - 1 ] : 0.0;
      auto const at = ( i < row.size() ) ? row[ i ] : 0.0;
      next[ i ] = before - at;
    }

    row = next;
  }

  return row;
}

// ----------------------------------------------------------------------------
/// A Sobel derivative, which is `cv::Sobel`.
///
/// \p Out should be signed or floating: a gradient has a sign, and an
/// unsigned result clips half of it away.
///
/// @param image the image
/// @param dx the order of the derivative across
/// @param dy the order of the derivative down
/// @param size 1, 3, 5 or 7
template < typename Out, typename T >
viame::image_of< Out >
sobel( viame::image_of< T > const& image, int dx, int dy,
       size_t size = 3, border_mode mode = border_mode::REFLECT_101 )
{
  // OpenCV builds the size-1 case as a 1 by 3 or 3 by 1, so the smoothing
  // row has to match whichever length the derivative row came out
  auto horizontal = sobel_kernel_1d( dx, size );
  auto vertical = sobel_kernel_1d( dy, size );

  if( size == 1 )
  {
    auto const wanted = std::max( horizontal.size(), vertical.size() );

    auto pad = [ wanted ]( std::vector< double >& row )
    {
      while( row.size() < wanted )
      {
        row.insert( row.begin(), 0.0 );
        if( row.size() < wanted ) { row.push_back( 0.0 ); }
      }
    };

    pad( horizontal );
    pad( vertical );
  }

  // Both template arguments, because with `Out` and `T` the same the one
  // argument overload below is an equally good match and the call is
  // ambiguous -- which only shows up on a float image
  return filter_2d< Out, T >( image, separable_kernel( horizontal, vertical ),
                              mode );
}

// ----------------------------------------------------------------------------
/// \p alpha times \p first plus \p beta times \p second plus \p gamma.
///
/// `cv::addWeighted`, which the enhancer's sharpening uses: an image plus a
/// weighted difference from its own blur.
///
/// Three details of OpenCV's, all found by fitting the model to what
/// `cv2.addWeighted` returns over every one of the 65536 byte pairs:
///
/// * it rounds **half to even**. With alpha 1.5 and beta -0.5 -- which is
///   exactly what sharpening uses -- every second byte pair lands on an exact
///   half, so rounding away from zero instead disagrees on a quarter of the
///   image. Weights of 1 and -1, or 2 and -1, produce no halves at all,
///   which is why this looked exact for as long as those were what it was
///   tested on;
/// * an integral image accumulates in **float**, not double. On 0.3 and 0.7
///   a double accumulation is wrong on 820 of the 65536 pairs;
/// * and the two products are **fused**, nested the way the vector body
///   nests them: `fma( first, alpha, fma( second, beta, gamma ) )`, one
///   rounding each rather than one per operation. Unfused float is wrong on
///   153 pairs; fused is wrong on none, at any weights tried.
template < typename T >
viame::image_of< T >
add_weighted( viame::image_of< T > const& first, double alpha,
              viame::image_of< T > const& second, double beta,
              double gamma = 0.0 )
{
  if( first.width() != second.width() || first.height() != second.height() ||
      first.depth() != second.depth() )
  {
    throw std::invalid_argument(
      "add_weighted: the two images have different shapes" );
  }

  viame::image_of< T > out( first.width(), first.height(),
                                    first.depth() );

  for( size_t plane = 0; plane < first.depth(); ++plane )
  {
    for( size_t j = 0; j < first.height(); ++j )
    {
      for( size_t i = 0; i < first.width(); ++i )
      {
        if constexpr( std::is_integral< T >::value )
        {
          auto const value = std::fma(
            static_cast< float >( first( i, j, plane ) ),
            static_cast< float >( alpha ),
            std::fma( static_cast< float >( second( i, j, plane ) ),
                      static_cast< float >( beta ),
                      static_cast< float >( gamma ) ) );

          out( i, j, plane ) =
            saturate_pixel_even< T >( static_cast< double >( value ) );
        }
        else
        {
          out( i, j, plane ) = saturate_pixel< T >(
            alpha * static_cast< double >( first( i, j, plane ) ) +
            beta * static_cast< double >( second( i, j, plane ) ) + gamma );
        }
      }
    }
  }

  return out;
}


// ----------------------------------------------------------------------------
/// Edge-preserving blur, which is `cv::bilateralFilter`.
///
/// Each output pixel is a weighted mean of its neighbourhood where the weight
/// is the product of two Gaussians: one on the distance and one on the
/// **colour** difference. That second factor is what preserves an edge --
/// across it the colour difference is large, the weight small, and the two
/// sides do not mix.
///
/// The weights are OpenCV's, table for table. \p diameter of zero derives the
/// radius from \p space_sigma, the round neighbourhood is the disk OpenCV
/// uses rather than the square it sits in, and the colour table is indexed by
/// the **sum** of the absolute per-plane differences, which is why it has one
/// entry per plane per grey level.
///
/// Three-plane images are exact against cv2 over every configuration
/// measured. A single-plane one is within a count on about half its pixels:
/// OpenCV's one-channel 8-bit body accumulates in a different order and the
/// twenty-odd float additions do not land on the same value.
template < typename T >
viame::image_of< T >
bilateral_blur( viame::image_of< T > const& image, int diameter,
                double colour_sigma, double space_sigma,
                border_mode mode = border_mode::REFLECT_101 )
{
  if( colour_sigma <= 0.0 ) { colour_sigma = 1.0; }
  if( space_sigma <= 0.0 ) { space_sigma = 1.0; }

  auto const colour_coefficient = -0.5 / ( colour_sigma * colour_sigma );
  auto const space_coefficient = -0.5 / ( space_sigma * space_sigma );

  auto radius = ( diameter <= 0 )
    ? static_cast< int >( std::nearbyint( space_sigma * 1.5 ) )
    : diameter / 2;
  radius = std::max( radius, 1 );

  auto const planes = image.depth();
  auto const levels = static_cast< size_t >( pixel_max< T >() ) + 1;

  // One entry per plane per level, because the index is the sum over planes.
  std::vector< float > colour_weight( levels * planes );

  for( size_t at = 0; at < colour_weight.size(); ++at )
  {
    colour_weight[ at ] = static_cast< float >( std::exp(
      static_cast< double >( at ) * static_cast< double >( at ) *
      colour_coefficient ) );
  }

  // The disk, not the square: OpenCV skips an offset whose distance exceeds
  // the radius, so a diameter of five is twenty-one taps and not twenty-five.
  std::vector< float > space_weight;
  std::vector< std::pair< int, int > > offsets;

  for( int dj = -radius; dj <= radius; ++dj )
  {
    for( int di = -radius; di <= radius; ++di )
    {
      auto const distance = std::sqrt( static_cast< double >( di * di + dj * dj ) );

      if( distance > static_cast< double >( radius ) ) { continue; }

      space_weight.push_back( static_cast< float >(
        std::exp( distance * distance * space_coefficient ) ) );
      offsets.emplace_back( di, dj );
    }
  }

  viame::image_of< T > out( image.width(), image.height(), planes );

  std::vector< float > total( planes );
  std::vector< double > centre( planes );
  std::vector< double > neighbour( planes );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      for( size_t plane = 0; plane < planes; ++plane )
      {
        centre[ plane ] = static_cast< double >( image( i, j, plane ) );
      }

      std::fill( total.begin(), total.end(), 0.0f );
      float weight_total = 0.0f;

      for( size_t tap = 0; tap < offsets.size(); ++tap )
      {
        auto const at_i = static_cast< long >( i ) + offsets[ tap ].first;
        auto const at_j = static_cast< long >( j ) + offsets[ tap ].second;

        long difference = 0;

        for( size_t plane = 0; plane < planes; ++plane )
        {
          neighbour[ plane ] = sample_with_border(
            image, at_i, at_j, plane, mode, 0.0 );
          difference += std::abs(
            static_cast< long >( neighbour[ plane ] ) -
            static_cast< long >( centre[ plane ] ) );
        }

        auto const weight =
          space_weight[ tap ] *
          colour_weight[ static_cast< size_t >( difference ) ];

        weight_total += weight;

        for( size_t plane = 0; plane < planes; ++plane )
        {
          total[ plane ] +=
            weight * static_cast< float >( neighbour[ plane ] );
        }
      }

      for( size_t plane = 0; plane < planes; ++plane )
      {
        out( i, j, plane ) =
          saturate_pixel_even< T >( total[ plane ] / weight_total );
      }
    }
  }

  return out;
}


namespace detail {

/// The five-tap binomial kernel both pyramid operations use, unnormalised.
constexpr long pyramid_taps[ 5 ] = { 1, 4, 6, 4, 1 };

} // namespace detail

// ----------------------------------------------------------------------------
/// Halve an image with the binomial filter, which is `cv::pyrDown`.
///
/// The output is `(width + 1) / 2` by `(height + 1) / 2`, the filter is the
/// separable `1 4 6 4 1` at a total gain of 256, the border reflects without
/// repeating the edge, and the rounding adds 128 before the shift. Exact
/// against cv2 on every size measured, odd ones included.
template < typename T >
viame::image_of< T >
pyramid_down( viame::image_of< T > const& image )
{
  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );
  auto const planes = image.depth();

  auto const out_width = ( image.width() + 1 ) / 2;
  auto const out_height = ( image.height() + 1 ) / 2;

  if( out_width == 0 || out_height == 0 )
  {
    throw std::invalid_argument( "pyramid_down: the source has no area" );
  }

  // The horizontal pass, kept for every source row and every output column.
  std::vector< long > across( image.height() * out_width * planes, 0 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < out_width; ++i )
    {
      for( int tap = 0; tap < 5; ++tap )
      {
        auto const at = detail::border_index(
          static_cast< long >( i ) * 2 + tap - 2, width,
          border_mode::REFLECT_101 );

        for( size_t plane = 0; plane < planes; ++plane )
        {
          across[ ( j * out_width + i ) * planes + plane ] +=
            detail::pyramid_taps[ tap ] *
            static_cast< long >( image( static_cast< size_t >( at ), j,
                                        plane ) );
        }
      }
    }
  }

  viame::image_of< T > out( out_width, out_height, planes );

  for( size_t j = 0; j < out_height; ++j )
  {
    for( size_t i = 0; i < out_width; ++i )
    {
      for( size_t plane = 0; plane < planes; ++plane )
      {
        long total = 0;

        for( int tap = 0; tap < 5; ++tap )
        {
          auto const at = detail::border_index(
            static_cast< long >( j ) * 2 + tap - 2, height,
            border_mode::REFLECT_101 );

          total += detail::pyramid_taps[ tap ] *
                   across[ ( static_cast< size_t >( at ) * out_width + i ) *
                           planes + plane ];
        }

        out( i, j, plane ) = saturate_pixel< T >(
          static_cast< double >( ( total + 128 ) >> 8 ) );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Double an image with the binomial filter, which is `cv::pyrUp`.
///
/// The requested size may be `2n` or `2n - 1` on either axis, which is what
/// OpenCV allows. **The filter runs on the full `2n` grid and the result is
/// then cropped**, rather than on a grid of the requested size: reflecting
/// inside an odd grid instead disagrees with cv2 on about four percent of the
/// pixels by up to nine counts, and cropping a `2n` grid agrees on every one
/// of the seven shapes measured.
template < typename T >
viame::image_of< T >
pyramid_up( viame::image_of< T > const& image, size_t width, size_t height )
{
  auto const planes = image.depth();
  auto const grid_width = image.width() * 2;
  auto const grid_height = image.height() * 2;

  if( width == 0 ) { width = grid_width; }
  if( height == 0 ) { height = grid_height; }

  if( width > grid_width || height > grid_height ||
      width + 1 < grid_width || height + 1 < grid_height )
  {
    throw std::invalid_argument(
      "pyramid_up: the target must be twice the source, or one less" );
  }

  // The zero-filled grid, only the even positions carrying a sample.
  auto const sample = [ & ]( long i, long j, size_t plane ) -> long
  {
    if( ( i & 1 ) || ( j & 1 ) ) { return 0; }

    return static_cast< long >(
      image( static_cast< size_t >( i / 2 ), static_cast< size_t >( j / 2 ),
             plane ) );
  };

  std::vector< long > across( grid_height * width * planes, 0 );

  for( size_t j = 0; j < grid_height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( int tap = 0; tap < 5; ++tap )
      {
        auto const at = detail::border_index(
          static_cast< long >( i ) + tap - 2,
          static_cast< long >( grid_width ), border_mode::REFLECT_101 );

        for( size_t plane = 0; plane < planes; ++plane )
        {
          across[ ( j * width + i ) * planes + plane ] +=
            detail::pyramid_taps[ tap ] *
            sample( at, static_cast< long >( j ), plane );
        }
      }
    }
  }

  viame::image_of< T > out( width, height, planes );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      for( size_t plane = 0; plane < planes; ++plane )
      {
        long total = 0;

        for( int tap = 0; tap < 5; ++tap )
        {
          auto const at = detail::border_index(
            static_cast< long >( j ) + tap - 2,
            static_cast< long >( grid_height ), border_mode::REFLECT_101 );

          total += detail::pyramid_taps[ tap ] *
                   across[ ( static_cast< size_t >( at ) * width + i ) *
                           planes + plane ];
        }

        // Four times the gain of the halving pass, because three quarters of
        // the grid is zero.
        out( i, j, plane ) = saturate_pixel< T >(
          static_cast< double >( ( total * 4 + 128 ) >> 8 ) );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Mean-shift filtering in space and colour, one level.
///
/// Each pixel walks to the mean of the neighbours within \p space_radius of
/// its position **and** within \p colour_radius of its colour, and repeats
/// until it stops moving or the iteration budget runs out; the colour it ends
/// on is the output. Flat regions collapse to one colour and an edge survives,
/// which is what makes it a segmentation pre-filter rather than a blur.
///
/// This is `cv::pyrMeanShiftFiltering` **at `maxLevel = 0`**, exact: the
/// integer window, the squared-distance test against `round(sr*sr)`, the
/// rounded centroid, OpenCV's two-part stopping rule and its iteration
/// clamping. OpenCV's default builds a pyramid and combines the levels
/// through a mask, and that combination is **not** reproduced here -- see
/// `design/lite-findings.md` 2.75 for what was established and what was not.
/// `pyramid_down` and `pyramid_up` above are exact, so the missing piece is
/// the combination rule and not the arithmetic.
template < typename T >
viame::image_of< T >
mean_shift_blur( viame::image_of< T > const& image, double space_radius,
                 double colour_radius, int max_iterations = 5,
                 double epsilon = 1.0 )
{
  if( image.depth() != 3 )
  {
    throw std::invalid_argument( "mean_shift_blur takes a three plane image" );
  }

  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );

  auto const radius = std::max(
    static_cast< long >( std::nearbyint( space_radius ) ), 1L );
  auto const colour_limit = static_cast< long long >(
    std::nearbyint( colour_radius * colour_radius ) );

  max_iterations = std::min( std::max( max_iterations, 1 ), 100 );
  epsilon = std::max( epsilon, 0.0 );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  for( long j = 0; j < height; ++j )
  {
    for( long i = 0; i < width; ++i )
    {
      auto x = i;
      auto y = j;
      long colour[ 3 ] = {
        static_cast< long >( image( static_cast< size_t >( i ),
                                    static_cast< size_t >( j ), 0 ) ),
        static_cast< long >( image( static_cast< size_t >( i ),
                                    static_cast< size_t >( j ), 1 ) ),
        static_cast< long >( image( static_cast< size_t >( i ),
                                    static_cast< size_t >( j ), 2 ) ) };

      for( int iteration = 0; iteration < max_iterations; ++iteration )
      {
        auto const low_x = std::max( x - radius, 0L );
        auto const low_y = std::max( y - radius, 0L );
        auto const high_x = std::min( x + radius, width - 1 );
        auto const high_y = std::min( y + radius, height - 1 );

        long long sum[ 3 ] = { 0, 0, 0 };
        long long sum_x = 0;
        long long sum_y = 0;
        long long count = 0;

        for( auto at_y = low_y; at_y <= high_y; ++at_y )
        {
          long long row_count = 0;

          for( auto at_x = low_x; at_x <= high_x; ++at_x )
          {
            long long distance = 0;
            long value[ 3 ];

            for( size_t plane = 0; plane < 3; ++plane )
            {
              value[ plane ] = static_cast< long >(
                image( static_cast< size_t >( at_x ),
                       static_cast< size_t >( at_y ), plane ) );
              auto const difference = value[ plane ] - colour[ plane ];
              distance += static_cast< long long >( difference ) * difference;
            }

            if( distance > colour_limit ) { continue; }

            for( size_t plane = 0; plane < 3; ++plane )
            {
              sum[ plane ] += value[ plane ];
            }

            sum_x += at_x;
            ++row_count;
          }

          count += row_count;
          sum_y += at_y * row_count;
        }

        if( count == 0 ) { break; }

        auto const inverse = 1.0 / static_cast< double >( count );

        auto const next_x = static_cast< long >( std::nearbyint(
          static_cast< double >( sum_x ) * inverse ) );
        auto const next_y = static_cast< long >( std::nearbyint(
          static_cast< double >( sum_y ) * inverse ) );

        long next[ 3 ];

        for( size_t plane = 0; plane < 3; ++plane )
        {
          next[ plane ] = static_cast< long >( std::nearbyint(
            static_cast< double >( sum[ plane ] ) * inverse ) );
        }

        // OpenCV's rule, both halves: either nothing moved, or the total of
        // the spatial step and the squared colour step is within epsilon.
        long long moved = std::abs( next_x - x ) + std::abs( next_y - y );

        for( size_t plane = 0; plane < 3; ++plane )
        {
          auto const difference = next[ plane ] - colour[ plane ];
          moved += static_cast< long long >( difference ) * difference;
        }

        auto const settled = ( x == next_x && y == next_y ) ||
                             static_cast< double >( moved ) <= epsilon;

        x = next_x;
        y = next_y;
        colour[ 0 ] = next[ 0 ];
        colour[ 1 ] = next[ 1 ];
        colour[ 2 ] = next[ 2 ];

        if( settled ) { break; }
      }

      for( size_t plane = 0; plane < 3; ++plane )
      {
        out( static_cast< size_t >( i ), static_cast< size_t >( j ), plane ) =
          saturate_pixel< T >( static_cast< double >( colour[ plane ] ) );
      }
    }
  }

  return out;
}


// ----------------------------------------------------------------------------
/// The discrete Laplacian, which is `cv::Laplacian`.
///
/// \p aperture picks the kernel, and the three are **not** three precisions
/// of one operator -- 1 is the five-point stencil, and 3 and 5 come from the
/// separable Sobel pair, which puts the
/// weight on the **corners** at 3 rather than on the edges. Measured off a
/// delta image rather than derived, and exact for all three.
///
/// The result is signed, so a `float` image is the useful one; an integer
/// image saturates the way `cv::Laplacian` does when it is asked for the same
/// depth it was given.
template < typename T >
viame::image_of< T >
laplacian( viame::image_of< T > const& image, size_t aperture = 1,
           double scale = 1.0, double delta = 0.0,
           border_mode mode = border_mode::REFLECT_101 )
{
  static constexpr double five_point[ 9 ] = {
    0.0, 1.0, 0.0,
    1.0, -4.0, 1.0,
    0.0, 1.0, 0.0 };

  static constexpr double corners[ 9 ] = {
    2.0, 0.0, 2.0,
    0.0, -8.0, 0.0,
    2.0, 0.0, 2.0 };

  static constexpr double wide[ 25 ] = {
    2.0, 4.0, 4.0, 4.0, 2.0,
    4.0, 0.0, -8.0, 0.0, 4.0,
    4.0, -8.0, -24.0, -8.0, 4.0,
    4.0, 0.0, -8.0, 0.0, 4.0,
    2.0, 4.0, 4.0, 4.0, 2.0 };

  double const* taps = nullptr;
  long radius = 1;

  switch( aperture )
  {
    case 1: taps = five_point; break;
    case 3: taps = corners; break;
    case 5: taps = wide; radius = 2; break;
    default:
      throw std::invalid_argument(
        "laplacian: the aperture is 1, 3 or 5, got " +
        std::to_string( aperture ) );
  }

  auto const side = 2 * radius + 1;

  viame::image_of< T > out( image.width(), image.height(), image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        double total = 0.0;

        for( long dj = -radius; dj <= radius; ++dj )
        {
          for( long di = -radius; di <= radius; ++di )
          {
            auto const weight =
              taps[ ( dj + radius ) * side + ( di + radius ) ];

            if( weight == 0.0 ) { continue; }

            total += weight * sample_with_border(
              image, static_cast< long >( i ) + di,
              static_cast< long >( j ) + dj, plane, mode, 0.0 );
          }
        }

        out( i, j, plane ) = saturate_pixel_even< T >( total * scale + delta );
      }
    }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif
