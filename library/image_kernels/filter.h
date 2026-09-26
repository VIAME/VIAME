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

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace viame {
namespace image_kernels {

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
/// A Gaussian of \p size samples with standard deviation \p sigma.
///
/// When \p sigma is not positive, `cv::getGaussianKernel` does **not** simply
/// derive one from the size for a small kernel: for an odd size of seven or
/// less it returns a fixed table instead, and only falls back to
/// `0.3 * ((size - 1) * 0.5 - 1) + 0.8` above that. The table's rows are not
/// samples of any Gaussian -- (0.25, 0.5, 0.25) is a binomial -- so deriving
/// the sigma and evaluating gives a visibly different blur at size three,
/// which is the size the pipelines use most.
///
/// The table is reproduced here for that reason. An explicit \p sigma takes
/// the evaluated path whatever the size, which is also what OpenCV does.
///
/// The result sums to one either way.
inline std::vector< double >
gaussian_kernel_1d( size_t size, double sigma = 0.0 )
{
  if( size == 0 || size % 2 == 0 )
  {
    throw std::invalid_argument(
      "gaussian_kernel_1d: the size has to be odd and positive" );
  }

  if( sigma <= 0.0 && size <= 7 )
  {
    // cv::getGaussianKernel's small_gaussian_tab, indexed by (size - 1) / 2
    static std::vector< double > const table[] = {
      { 1.0 },
      { 0.25, 0.5, 0.25 },
      { 0.0625, 0.25, 0.375, 0.25, 0.0625 },
      { 0.03125, 0.109375, 0.21875, 0.28125, 0.21875, 0.109375, 0.03125 },
    };

    return table[ ( size - 1 ) / 2 ];
  }

  if( sigma <= 0.0 )
  {
    sigma = 0.3 * ( ( static_cast< double >( size ) - 1.0 ) * 0.5 - 1.0 ) + 0.8;
  }

  auto const centre = static_cast< double >( size / 2 );
  std::vector< double > out( size );
  double total = 0.0;

  for( size_t i = 0; i < size; ++i )
  {
    auto const offset = static_cast< double >( i ) - centre;
    out[ i ] = std::exp( -( offset * offset ) / ( 2.0 * sigma * sigma ) );
    total += out[ i ];
  }

  for( auto& weight : out )
  {
    weight /= total;
  }

  return out;
}

// ----------------------------------------------------------------------------
/// A Gaussian blur, which is `cv::GaussianBlur`.
///
/// @param image the image
/// @param size the kernel width and height, odd
/// @param sigma the standard deviation, derived from the size when not given
template < typename T >
viame::image_of< T >
gaussian_blur( viame::image_of< T > const& image, size_t size,
               double sigma = 0.0,
               border_mode mode = border_mode::REFLECT_101 )
{
  auto const line = gaussian_kernel_1d( size, sigma );
  return separable_filter< T, T >( image, line, line, mode );
}

// ----------------------------------------------------------------------------
/// A box blur, which is `cv::blur`: the mean of a \p size by \p size window.
template < typename T >
viame::image_of< T >
box_blur( viame::image_of< T > const& image, size_t size,
          border_mode mode = border_mode::REFLECT_101 )
{
  if( size == 0 )
  {
    throw std::invalid_argument( "box_blur: the size has to be positive" );
  }

  std::vector< double > const line(
    size, 1.0 / static_cast< double >( size ) );

  return separable_filter< T, T >( image, line, line, mode );
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
        out( i, j, plane ) = saturate_pixel< T >(
          alpha * static_cast< double >( first( i, j, plane ) ) +
          beta * static_cast< double >( second( i, j, plane ) ) + gamma );
      }
    }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif
