/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Dense optical flow by Farneback's polynomial expansion
///
/// What `cv::calcOpticalFlowFarneback` did, with `flags` at zero.
///
/// The idea is Farneback's: fit a quadratic to the neighbourhood of every
/// pixel in both frames,
///
///     f(x) ~ x^T A x + b^T x + c
///
/// and then, if the second frame is the first one displaced by d, the two
/// fits are related by `b2 = b1 - 2 A d`, which solves for d directly. Real
/// images are not globally quadratic, so the displacement is re-estimated
/// over a window, coarse to fine, a few times per level.
///
/// The details are OpenCV's, because `ocv_optical_flow` is a shipped filter
/// whose byte output is recorded:
///
/// * the quadratic is fitted by weighted least squares against the basis
///   `1, x, y, x^2, y^2, xy` with a separable Gaussian weight, and because
///   that weight is separable the normal equations are the same six by six
///   matrix at every pixel -- inverted once, up front;
/// * the fit is accumulated **vertically in float and horizontally in
///   double**, and both borders replicate;
/// * the matrix and right hand side are averaged over a plain box window,
///   not a Gaussian one, since `OPTFLOW_FARNEBACK_GAUSSIAN` is off;
/// * a five pixel rim is faded by a fixed weight table before that
///   averaging, so the edge of the frame does not pull the answer;
/// * the pyramid stops at 32 pixels, and each level is blurred by a Gaussian
///   whose width follows the scale and then resampled by **OpenCV's**
///   bilinear convention, which is not `resize_bilinear`'s;
/// * the two by two solve is regularised by 1e-3 on the determinant.
///
/// Measured against `cv2.calcOpticalFlowFarneback` over four frame sizes and
/// one, two and four pyramid levels: never more than 7e-6 of a pixel, which
/// is float32 rounding and nothing else.
///
/// The working buffers here are plain vectors with the five or two values of
/// a pixel adjacent, rather than `image_of`, because every loop below is
/// swept in order and an `image_of` subscript costs three multiplications to
/// reach a neighbour that is already under the pointer.

#ifndef VIAME_IMAGE_KERNELS_OPTICAL_FLOW_H
#define VIAME_IMAGE_KERNELS_OPTICAL_FLOW_H

#include <image_kernels/filter.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace viame {
namespace image_kernels {

/// How `cv::calcOpticalFlowFarneback` is configured.
struct farneback_params
{
  /// The ratio between one pyramid level and the next, under one.
  double pyr_scale = 0.5;

  /// How many levels to add above the frame itself.
  int levels = 3;

  /// The side of the box the displacement is averaged over.
  int winsize = 15;

  /// Re-estimations per level.
  int iterations = 3;

  /// The side of the neighbourhood the quadratic is fitted to.
  int poly_n = 5;

  /// The width of the Gaussian weighting that fit.
  double poly_sigma = 1.2;
};

namespace detail {

/// The smallest a pyramid level is allowed to get, as OpenCV's is.
constexpr int farneback_min_size = 32;

/// Where asking for "one thread per core" stops, as `windowed_trainer`'s does.
constexpr unsigned max_auto_threads = 32;

/// `cvRound`: to nearest, and a half to the even neighbour.
///
/// Not `lround`, which sends a half away from zero and so halves an odd
/// frame width to a different pixel count than OpenCV's pyramid does.
inline long
round_half_even( double value )
{
  return static_cast< long >( std::nearbyint( value ) );
}

/// Clamp an index to an extent, which is what both borders here do.
inline size_t
clamp_index( long at, size_t extent )
{
  if( at < 0 ) { return 0; }
  if( static_cast< size_t >( at ) >= extent ) { return extent - 1; }
  return static_cast< size_t >( at );
}

/// Resample \p planes interleaved planes by **OpenCV's** bilinear convention.
///
/// Not `resize_bilinear`, which reproduces VXL: that one spans the source
/// corner to corner, and this one puts the destination pixel's centre at
/// `(i + 0.5) * scale - 0.5` in the source and replicates past the edge.
/// The two differ by half a pixel of the scale factor, which is a whole
/// pixel of flow by the time the pyramid has climbed back down, so the
/// convention has to be OpenCV's here whatever it is elsewhere.
inline std::vector< float >
resize_linear_cv( std::vector< float > const& source,
                  size_t source_width, size_t source_height, size_t planes,
                  size_t width, size_t height )
{
  std::vector< float > out( width * height * planes );

  if( width == 0 || height == 0 || source_width == 0 || source_height == 0 )
  {
    return out;
  }

  auto const axis =
    []( size_t count, size_t extent )
    {
      std::vector< size_t > low( count ), high( count );
      std::vector< float > weight( count );

      auto const scale = static_cast< double >( extent ) /
                         static_cast< double >( count );

      for( size_t i = 0; i < count; ++i )
      {
        auto position = static_cast< float >(
          ( static_cast< double >( i ) + 0.5 ) * scale - 0.5 );

        auto whole = static_cast< long >( std::floor( position ) );
        position -= static_cast< float >( whole );

        if( whole < 0 ) { whole = 0; position = 0.0f; }

        if( whole >= static_cast< long >( extent ) - 1 )
        {
          whole = static_cast< long >( extent ) - 1;
          position = 0.0f;
        }

        low[ i ] = static_cast< size_t >( whole );
        high[ i ] = clamp_index( whole + 1, extent );
        weight[ i ] = position;
      }

      return std::make_tuple( low, high, weight );
    };

  auto const across = axis( width, source_width );
  auto const down = axis( height, source_height );

  for( size_t j = 0; j < height; ++j )
  {
    auto const* top = source.data() +
      std::get< 0 >( down )[ j ] * source_width * planes;
    auto const* bottom = source.data() +
      std::get< 1 >( down )[ j ] * source_width * planes;
    auto const fy = std::get< 2 >( down )[ j ];

    auto* destination = out.data() + j * width * planes;

    for( size_t i = 0; i < width; ++i )
    {
      auto const x0 = std::get< 0 >( across )[ i ] * planes;
      auto const x1 = std::get< 1 >( across )[ i ] * planes;
      auto const fx = std::get< 2 >( across )[ i ];

      for( size_t plane = 0; plane < planes; ++plane )
      {
        auto const a = top[ x0 + plane ] * ( 1.0f - fx ) +
                       top[ x1 + plane ] * fx;
        auto const b = bottom[ x0 + plane ] * ( 1.0f - fx ) +
                       bottom[ x1 + plane ] * fx;

        destination[ i * planes + plane ] = a * ( 1.0f - fy ) + b * fy;
      }
    }
  }

  return out;
}

/// The four entries of the inverted normal matrix the polynomial fit needs.
///
/// The basis is `1, x, y, x^2, y^2, xy` and the weight is `g(x) g(y)`, which
/// makes most of the six by six matrix vanish by symmetry: every odd moment
/// is zero, `x` and `y` and `xy` each stand alone, and `1`, `x^2` and `y^2`
/// are left coupled in a three by three block. Only four entries of the
/// inverse are ever read, and these are they.
struct poly_normal
{
  double ig11;  ///< the linear terms, `1 / sum w x^2`
  double ig03;  ///< the constant to quadratic coupling
  double ig33;  ///< the quadratic terms
  double ig55;  ///< the cross term, `1 / sum w x^2 y^2`
};

inline poly_normal
poly_normal_matrix( int n, std::vector< double > const& g )
{
  double m0 = 0.0, m2 = 0.0, m4 = 0.0, m22 = 0.0;

  for( int y = -n; y <= n; ++y )
  {
    for( int x = -n; x <= n; ++x )
    {
      auto const w = g[ static_cast< size_t >( y + n ) ] *
                     g[ static_cast< size_t >( x + n ) ];
      auto const xx = static_cast< double >( x ) * x;
      auto const yy = static_cast< double >( y ) * y;

      m0 += w;
      m2 += w * xx;
      m4 += w * xx * xx;
      m22 += w * xx * yy;
    }
  }

  // The coupled block: [ 1, x^2, y^2 ] against itself
  double const b[ 3 ][ 3 ] = { { m0, m2, m2 },
                               { m2, m4, m22 },
                               { m2, m22, m4 } };

  auto const cofactor =
    [ & ]( int r, int c )
    {
      int const r0 = ( r + 1 ) % 3, r1 = ( r + 2 ) % 3;
      int const c0 = ( c + 1 ) % 3, c1 = ( c + 2 ) % 3;
      return b[ r0 ][ c0 ] * b[ r1 ][ c1 ] - b[ r0 ][ c1 ] * b[ r1 ][ c0 ];
    };

  auto const determinant = b[ 0 ][ 0 ] * cofactor( 0, 0 ) +
                           b[ 0 ][ 1 ] * cofactor( 0, 1 ) +
                           b[ 0 ][ 2 ] * cofactor( 0, 2 );

  if( determinant == 0.0 || m2 == 0.0 || m22 == 0.0 )
  {
    throw std::invalid_argument(
      "farneback: the polynomial weighting is degenerate" );
  }

  poly_normal out;
  out.ig11 = 1.0 / m2;
  out.ig55 = 1.0 / m22;
  // The inverse of a symmetric matrix is symmetric, so the cofactor matrix
  // is its own adjugate here
  out.ig03 = cofactor( 1, 0 ) / determinant;
  out.ig33 = cofactor( 1, 1 ) / determinant;
  return out;
}

/// The quadratic fitted at every pixel, five values per pixel.
///
/// The five are the coefficients of `y`, `x`, `y^2`, `x^2` and `xy`, in that
/// order -- OpenCV's order, where the vertical term comes first.
inline std::vector< float >
poly_expansion( std::vector< float > const& image,
                size_t width, size_t height, int n, double sigma )
{
  if( n < 1 )
  {
    throw std::invalid_argument( "farneback: poly_n has to be positive" );
  }

  if( sigma < 1e-7 )
  {
    sigma = n * 0.3;
  }

  auto const size = static_cast< size_t >( 2 * n + 1 );
  std::vector< double > g( size ), xg( size ), xxg( size );

  double total = 0.0;

  for( int x = -n; x <= n; ++x )
  {
    auto const value = static_cast< float >(
      std::exp( -static_cast< double >( x ) * x / ( 2 * sigma * sigma ) ) );
    g[ static_cast< size_t >( x + n ) ] = value;
    total += value;
  }

  for( int x = -n; x <= n; ++x )
  {
    auto const at = static_cast< size_t >( x + n );
    g[ at ] = static_cast< float >( g[ at ] / total );
    xg[ at ] = static_cast< double >( x ) * g[ at ];
    xxg[ at ] = static_cast< double >( x ) * x * g[ at ];
  }

  auto const normal = poly_normal_matrix( n, g );

  // The vertical half, accumulated in float as OpenCV's is: the plain
  // weighting, the first derivative across the rows, and the second.
  std::vector< float > t0( width * height ), t1( width * height ),
                       t2( width * height );

  auto const g0 = static_cast< float >( g[ static_cast< size_t >( n ) ] );

  for( size_t at = 0; at < width * height; ++at )
  {
    t0[ at ] = image[ at ] * g0;
  }

  for( int k = 1; k <= n; ++k )
  {
    auto const gk = static_cast< float >( g[ static_cast< size_t >( n + k ) ] );
    auto const xgk = static_cast< float >( xg[ static_cast< size_t >( n + k ) ] );
    auto const xxgk = static_cast< float >( xxg[ static_cast< size_t >( n + k ) ] );

    for( size_t y = 0; y < height; ++y )
    {
      auto const* above = image.data() +
        clamp_index( static_cast< long >( y ) - k, height ) * width;
      auto const* below = image.data() +
        clamp_index( static_cast< long >( y ) + k, height ) * width;

      auto* a0 = t0.data() + y * width;
      auto* a1 = t1.data() + y * width;
      auto* a2 = t2.data() + y * width;

      for( size_t x = 0; x < width; ++x )
      {
        auto const sum = above[ x ] + below[ x ];

        a0[ x ] += gk * sum;
        a1[ x ] += xgk * ( below[ x ] - above[ x ] );
        a2[ x ] += xxgk * sum;
      }
    }
  }

  std::vector< float > out( width * height * 5 );

  auto const centre_g = g[ static_cast< size_t >( n ) ];

  // The horizontal half, accumulated in double as OpenCV's is
  for( size_t y = 0; y < height; ++y )
  {
    auto const* a0 = t0.data() + y * width;
    auto const* a1 = t1.data() + y * width;
    auto const* a2 = t2.data() + y * width;

    auto* destination = out.data() + y * width * 5;

    for( size_t x = 0; x < width; ++x )
    {
      // `g` is even in k and `xg` is odd, so the tap at -k and the tap at +k
      // share every weight: the pair sums for the even terms and the pair
      // difference for the odd ones. That is half the multiplications, and
      // it is the difference between this being the cost of the whole filter
      // and not.
      double b1 = static_cast< double >( a0[ x ] ) * centre_g;
      double b3 = static_cast< double >( a1[ x ] ) * centre_g;
      double b5 = static_cast< double >( a2[ x ] ) * centre_g;
      double b2 = 0.0, b4 = 0.0, b6 = 0.0;

      for( int k = 1; k <= n; ++k )
      {
        auto const low = clamp_index( static_cast< long >( x ) - k, width );
        auto const high = clamp_index( static_cast< long >( x ) + k, width );
        auto const index = static_cast< size_t >( n + k );

        auto const sum0 = static_cast< double >( a0[ low ] ) +
                          static_cast< double >( a0[ high ] );
        auto const sum1 = static_cast< double >( a1[ low ] ) +
                          static_cast< double >( a1[ high ] );
        auto const sum2 = static_cast< double >( a2[ low ] ) +
                          static_cast< double >( a2[ high ] );
        auto const difference0 = static_cast< double >( a0[ high ] ) -
                                 static_cast< double >( a0[ low ] );
        auto const difference1 = static_cast< double >( a1[ high ] ) -
                                 static_cast< double >( a1[ low ] );

        b1 += sum0 * g[ index ];
        b2 += difference0 * xg[ index ];
        b3 += sum1 * g[ index ];
        b4 += sum0 * xxg[ index ];
        b5 += sum2 * g[ index ];
        b6 += difference1 * xg[ index ];
      }

      destination[ x * 5 ] = static_cast< float >( b3 * normal.ig11 );
      destination[ x * 5 + 1 ] = static_cast< float >( b2 * normal.ig11 );
      destination[ x * 5 + 2 ] =
        static_cast< float >( b1 * normal.ig03 + b5 * normal.ig33 );
      destination[ x * 5 + 3 ] =
        static_cast< float >( b1 * normal.ig03 + b4 * normal.ig33 );
      destination[ x * 5 + 4 ] = static_cast< float >( b6 * normal.ig55 );
    }
  }

  return out;
}

/// How far into the frame the rim is faded, and by how much.
constexpr int farneback_border = 5;
constexpr float farneback_border_weight[ farneback_border ] =
  { 0.14f, 0.14f, 0.4472f, 0.4472f, 0.4472f };

/// The per-pixel two by two system, from the two fits and the flow so far.
///
/// Five values per pixel: the three distinct entries of the matrix and the
/// two of the right hand side.
inline std::vector< float >
update_matrices( std::vector< float > const& r0,
                 std::vector< float > const& r1,
                 std::vector< float > const& flow,
                 size_t width, size_t height )
{
  std::vector< float > out( width * height * 5 );

  auto const last_x = static_cast< long >( width ) - 1;
  auto const last_y = static_cast< long >( height ) - 1;
  auto const row = width * 5;

  auto const fade =
    []( size_t at, size_t extent )
    {
      auto weight = 1.0f;

      if( at < static_cast< size_t >( farneback_border ) )
      {
        weight *= farneback_border_weight[ at ];
      }

      if( at + farneback_border >= extent )
      {
        weight *= farneback_border_weight[ extent - at - 1 ];
      }

      return weight;
    };

  for( size_t y = 0; y < height; ++y )
  {
    auto const* here = r0.data() + y * row;
    auto const* motion = flow.data() + y * width * 2;
    auto* destination = out.data() + y * row;

    auto const vertical_fade = fade( y, height );
    auto const near_edge = y < static_cast< size_t >( farneback_border ) ||
                           y + farneback_border >= height;

    for( size_t x = 0; x < width; ++x )
    {
      auto const dx = motion[ x * 2 ];
      auto const dy = motion[ x * 2 + 1 ];

      auto fx = static_cast< float >( x ) + dx;
      auto fy = static_cast< float >( y ) + dy;

      auto const x1 = static_cast< long >( std::floor( fx ) );
      auto const y1 = static_cast< long >( std::floor( fy ) );

      fx -= static_cast< float >( x1 );
      fy -= static_cast< float >( y1 );

      auto const* mine = here + x * 5;

      float r2, r3, r4, r5, r6;

      if( x1 >= 0 && x1 < last_x && y1 >= 0 && y1 < last_y )
      {
        auto const a00 = ( 1.0f - fx ) * ( 1.0f - fy );
        auto const a01 = fx * ( 1.0f - fy );
        auto const a10 = ( 1.0f - fx ) * fy;
        auto const a11 = fx * fy;

        auto const* corner = r1.data() +
          static_cast< size_t >( y1 ) * row + static_cast< size_t >( x1 ) * 5;

        auto const at =
          [ & ]( size_t plane )
          {
            return a00 * corner[ plane ] + a01 * corner[ plane + 5 ] +
                   a10 * corner[ row + plane ] + a11 * corner[ row + plane + 5 ];
          };

        r2 = at( 0 );
        r3 = at( 1 );
        r4 = ( mine[ 2 ] + at( 2 ) ) * 0.5f;
        r5 = ( mine[ 3 ] + at( 3 ) ) * 0.5f;
        r6 = ( mine[ 4 ] + at( 4 ) ) * 0.25f;
      }
      else
      {
        r2 = 0.0f;
        r3 = 0.0f;
        r4 = mine[ 2 ];
        r5 = mine[ 3 ];
        r6 = mine[ 4 ] * 0.5f;
      }

      r2 = ( mine[ 0 ] - r2 ) * 0.5f;
      r3 = ( mine[ 1 ] - r3 ) * 0.5f;

      r2 += r4 * dy + r6 * dx;
      r3 += r6 * dy + r5 * dx;

      if( near_edge || x < static_cast< size_t >( farneback_border ) ||
          x + farneback_border >= width )
      {
        auto const scale = vertical_fade * fade( x, width );

        r2 *= scale; r3 *= scale;
        r4 *= scale; r5 *= scale; r6 *= scale;
      }

      destination[ x * 5 ] = r4 * r4 + r6 * r6;
      destination[ x * 5 + 1 ] = ( r4 + r5 ) * r6;
      destination[ x * 5 + 2 ] = r5 * r5 + r6 * r6;
      destination[ x * 5 + 3 ] = r4 * r2 + r6 * r3;
      destination[ x * 5 + 4 ] = r6 * r2 + r5 * r3;
    }
  }

  return out;
}

/// Average the systems over a box and solve each one.
///
/// The box is swept as a running sum in each axis, which is both what makes
/// it affordable and what makes it OpenCV's: the sums are kept in double and
/// the window slides rather than being re-added, so the rounding follows the
/// same path. The vertical sum is one row wide and is consumed as it is
/// produced -- keeping the whole column sum would be five doubles a pixel,
/// which on a 1080p frame is eighty megabytes written and read three times
/// per level for no reason.
inline std::vector< float >
update_flow( std::vector< float > const& matrices,
             size_t width, size_t height, int block_size )
{
  auto const m = block_size / 2;
  auto const scale = 1.0 / ( static_cast< double >( block_size ) * block_size );
  auto const row = width * 5;

  std::vector< float > flow( width * height * 2 );
  std::vector< double > running( row, 0.0 );

  for( int k = -m; k <= m; ++k )
  {
    auto const* source = matrices.data() + clamp_index( k, height ) * row;

    for( size_t i = 0; i < row; ++i )
    {
      running[ i ] += source[ i ];
    }
  }

  for( size_t y = 0; y < height; ++y )
  {
    auto const* source = running.data();
    auto* destination = flow.data() + y * width * 2;

    double sum[ 5 ] = { 0.0, 0.0, 0.0, 0.0, 0.0 };

    for( int k = -m; k <= m; ++k )
    {
      auto const at = clamp_index( k, width ) * 5;

      for( size_t plane = 0; plane < 5; ++plane )
      {
        sum[ plane ] += source[ at + plane ];
      }
    }

    for( size_t x = 0; x < width; ++x )
    {
      auto const g11 = sum[ 0 ] * scale;
      auto const g12 = sum[ 1 ] * scale;
      auto const g22 = sum[ 2 ] * scale;
      auto const h1 = sum[ 3 ] * scale;
      auto const h2 = sum[ 4 ] * scale;

      // The 1e-3 is OpenCV's, and is what keeps a flat neighbourhood -- one
      // with no gradient to measure a displacement against -- from dividing
      // by nothing at all.
      auto const inverse = 1.0 / ( g11 * g22 - g12 * g12 + 1e-3 );

      destination[ x * 2 ] =
        static_cast< float >( ( g11 * h2 - g12 * h1 ) * inverse );
      destination[ x * 2 + 1 ] =
        static_cast< float >( ( g22 * h1 - g12 * h2 ) * inverse );

      if( x + 1 < width )
      {
        auto const added = clamp_index(
          static_cast< long >( x ) + 1 + m, width ) * 5;
        auto const dropped = clamp_index(
          static_cast< long >( x ) - m, width ) * 5;

        for( size_t plane = 0; plane < 5; ++plane )
        {
          sum[ plane ] += source[ added + plane ] - source[ dropped + plane ];
        }
      }
    }

    if( y + 1 < height )
    {
      auto const* added = matrices.data() +
        clamp_index( static_cast< long >( y ) + 1 + m, height ) * row;
      auto const* dropped = matrices.data() +
        clamp_index( static_cast< long >( y ) - m, height ) * row;

      for( size_t i = 0; i < row; ++i )
      {
        running[ i ] += added[ i ] - dropped[ i ];
      }
    }
  }

  return flow;
}

/// One plane blurred by the same Gaussian `gaussian_blur` builds.
///
/// `gaussian_blur` is separable itself now -- this file's need for a
/// seventeen tap blur of a full resolution frame is what sent it that way --
/// so the two are the same arithmetic and the same cost. What this keeps is
/// the **buffer**: everything else here works on flat vectors, and calling
/// `gaussian_blur` would copy a plane into an `image_of` and back out of one
/// for every level of every frame, which at 1080p is a hundred and twenty
/// megabytes of copying to save twenty lines.
inline std::vector< float >
blur_plane( std::vector< float > const& source, size_t width, size_t height,
            size_t size, double sigma )
{
  auto const line = gaussian_kernel_1d( size, sigma );
  auto const radius = static_cast< long >( line.size() / 2 );

  std::vector< double > across( width * height );

  for( size_t y = 0; y < height; ++y )
  {
    auto const* row = source.data() + y * width;
    auto* destination = across.data() + y * width;

    for( size_t x = 0; x < width; ++x )
    {
      double total = 0.0;

      for( size_t k = 0; k < line.size(); ++k )
      {
        auto const at = border_index(
          static_cast< long >( x ) + static_cast< long >( k ) - radius,
          static_cast< long >( width ), border_mode::REFLECT_101 );

        total += line[ k ] * row[ static_cast< size_t >( at ) ];
      }

      destination[ x ] = total;
    }
  }

  std::vector< float > out( width * height );
  std::vector< double > accumulator( width );

  for( size_t y = 0; y < height; ++y )
  {
    std::fill( accumulator.begin(), accumulator.end(), 0.0 );

    // The tap loop is outside the pixel loop so each pass walks one row of
    // `across` in order; the other way round asks for `line.size()` rows at
    // once for every pixel, which on a wide frame is a cache miss each.
    for( size_t k = 0; k < line.size(); ++k )
    {
      auto const at = border_index(
        static_cast< long >( y ) + static_cast< long >( k ) - radius,
        static_cast< long >( height ), border_mode::REFLECT_101 );

      auto const* row = across.data() + static_cast< size_t >( at ) * width;
      auto const weight = line[ k ];

      for( size_t x = 0; x < width; ++x )
      {
        accumulator[ x ] += weight * row[ x ];
      }
    }

    auto* destination = out.data() + y * width;

    for( size_t x = 0; x < width; ++x )
    {
      destination[ x ] = static_cast< float >( accumulator[ x ] );
    }
  }

  return out;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// The dense flow from \p prev to \p next, as two planes of `float`.
///
/// Plane 0 is the horizontal displacement and plane 1 the vertical, both in
/// pixels of the input, which is how `cv::calcOpticalFlowFarneback` returns
/// them.
template < typename T >
viame::image_of< float >
farneback_optical_flow( viame::image_of< T > const& prev,
                        viame::image_of< T > const& next,
                        farneback_params const& params = farneback_params() )
{
  if( prev.depth() != 1 || next.depth() != 1 )
  {
    throw std::invalid_argument( "farneback takes single plane images" );
  }

  if( prev.width() != next.width() || prev.height() != next.height() )
  {
    throw std::invalid_argument( "farneback takes two frames of a size" );
  }

  if( params.pyr_scale <= 0.0 || params.pyr_scale >= 1.0 )
  {
    throw std::invalid_argument( "farneback: pyr_scale is between 0 and 1" );
  }

  if( params.levels < 0 || params.winsize < 1 || params.iterations < 0 )
  {
    throw std::invalid_argument(
      "farneback: levels, winsize or iterations out of range" );
  }

  auto const width = prev.width();
  auto const height = prev.height();

  std::vector< float > frame[ 2 ];
  frame[ 0 ].resize( width * height );
  frame[ 1 ].resize( width * height );

  for( size_t y = 0; y < height; ++y )
  {
    for( size_t x = 0; x < width; ++x )
    {
      frame[ 0 ][ y * width + x ] = static_cast< float >( prev( x, y, 0 ) );
      frame[ 1 ][ y * width + x ] = static_cast< float >( next( x, y, 0 ) );
    }
  }

  // How many levels the frame is actually big enough for
  int levels = 0;
  double scale = 1.0;

  while( levels < params.levels )
  {
    scale *= params.pyr_scale;

    if( width * scale < detail::farneback_min_size ||
        height * scale < detail::farneback_min_size )
    {
      break;
    }

    ++levels;
  }

  std::vector< float > flow;
  size_t flow_width = 0;
  size_t flow_height = 0;

  for( int level = levels; level >= 0; --level )
  {
    auto const level_scale = std::pow( params.pyr_scale, level );
    auto const sigma = ( 1.0 / level_scale - 1.0 ) * 0.5;

    auto smooth = static_cast< int >(
      detail::round_half_even( sigma * 5.0 ) ) | 1;
    smooth = std::max( smooth, 3 );

    auto const level_width = static_cast< size_t >(
      detail::round_half_even( width * level_scale ) );
    auto const level_height = static_cast< size_t >(
      detail::round_half_even( height * level_scale ) );

    if( flow.empty() )
    {
      flow.assign( level_width * level_height * 2, 0.0f );
    }
    else
    {
      flow = detail::resize_linear_cv( flow, flow_width, flow_height, 2,
                                       level_width, level_height );

      auto const gain = static_cast< float >( 1.0 / params.pyr_scale );

      for( auto& value : flow ) { value *= gain; }
    }

    flow_width = level_width;
    flow_height = level_height;

    std::vector< float > fits[ 2 ];

    for( int which = 0; which < 2; ++which )
    {
      auto plane = detail::blur_plane( frame[ which ], width, height,
                                       static_cast< size_t >( smooth ), sigma );

      if( level_width != width || level_height != height )
      {
        plane = detail::resize_linear_cv( plane, width, height, 1,
                                          level_width, level_height );
      }

      fits[ which ] = detail::poly_expansion( plane, level_width, level_height,
                                              params.poly_n,
                                              params.poly_sigma );
    }

    auto matrices = detail::update_matrices( fits[ 0 ], fits[ 1 ], flow,
                                             level_width, level_height );

    for( int pass = 0; pass < params.iterations; ++pass )
    {
      flow = detail::update_flow( matrices, level_width, level_height,
                                  params.winsize );

      if( pass < params.iterations - 1 )
      {
        matrices = detail::update_matrices( fits[ 0 ], fits[ 1 ], flow,
                                            level_width, level_height );
      }
    }
  }

  viame::image_of< float > out( flow_width, flow_height, 2 );

  for( size_t y = 0; y < flow_height; ++y )
  {
    for( size_t x = 0; x < flow_width; ++x )
    {
      out( x, y, 0 ) = flow[ ( y * flow_width + x ) * 2 ];
      out( x, y, 1 ) = flow[ ( y * flow_width + x ) * 2 + 1 ];
    }
  }

  return out;
}


// ----------------------------------------------------------------------------
/// How `cv::calcOpticalFlowPyrLK` is configured.
struct lucas_kanade_params
{
  /// The side of the window each point is matched over.
  int win_width = 21;
  int win_height = 21;

  /// How many levels to add above the frame itself.
  int levels = 3;

  /// The most iterations one point takes at one level.
  int iterations = 30;

  /// The step below which a point has settled, in pixels.
  double epsilon = 0.01;

  /// The corner strength below which a point is not trackable at all.
  double min_eigen = 1e-4;

  /// How many threads to follow the points on; 0 asks for one per core.
  ///
  /// Following a point reads the pyramid and writes its own two answers, so
  /// the work is independent and the result does not depend on how it was
  /// divided. `cv::calcOpticalFlowPyrLK` parallelises the same loop the same
  /// way -- measured on sixteen cores it is three times faster than itself on
  /// one -- so a single threaded port is not being compared with like.
  int threads = 0;
};

namespace detail {

/// Where each of \p count positions reads from, once, under reflect_101.
///
/// `border_index` is a switch on the border rule, and every user of it below
/// wants the same rule at the same offsets for every pixel of a row. Asking
/// it once per position and keeping the answer turns a dispatched call per
/// tap into a load: on a 1080p pair the pyramid alone was making some fifty
/// million of those calls, which was three quarters of the whole tracker.
inline std::vector< size_t >
reflect_map( long from, long count, long extent )
{
  std::vector< size_t > out( static_cast< size_t >( count ) );

  for( long i = 0; i < count; ++i )
  {
    out[ static_cast< size_t >( i ) ] = static_cast< size_t >(
      border_index( from + i, extent, border_mode::REFLECT_101 ) );
  }

  return out;
}

/// `cv::pyrDown` for bytes: the 5 tap binomial, halved, reflect_101.
inline viame::image_of< uint8_t >
pyr_down( viame::image_of< uint8_t > const& image )
{
  auto const width = image.width();
  auto const height = image.height();
  auto const out_width = ( width + 1 ) / 2;
  auto const out_height = ( height + 1 ) / 2;

  static int const tap[ 5 ] = { 1, 4, 6, 4, 1 };

  // The five source columns of each destination column, and likewise rows
  std::vector< size_t > columns( out_width * 5 ), rows( out_height * 5 );

  for( size_t x = 0; x < out_width; ++x )
  {
    for( int k = 0; k < 5; ++k )
    {
      columns[ x * 5 + static_cast< size_t >( k ) ] = static_cast< size_t >(
        border_index( static_cast< long >( x ) * 2 + k - 2,
                      static_cast< long >( width ),
                      border_mode::REFLECT_101 ) );
    }
  }

  for( size_t y = 0; y < out_height; ++y )
  {
    for( int k = 0; k < 5; ++k )
    {
      rows[ y * 5 + static_cast< size_t >( k ) ] = static_cast< size_t >(
        border_index( static_cast< long >( y ) * 2 + k - 2,
                      static_cast< long >( height ),
                      border_mode::REFLECT_101 ) );
    }
  }

  std::vector< int > across( out_width * height );

  auto const* const base = image.first_pixel();
  auto const across_step = image.w_step();
  auto const down_step = image.h_step();

  // The interior columns read five pixels at a fixed stride of two, which is
  // a shape the compiler can widen. Reaching them through
  // `image_of::operator()` -- three multiplications, and an index it cannot
  // prove monotonic -- is what stopped it: this is integer work with no
  // associativity question, so once it is pointers it vectorises and there is
  // nothing left here for hand written intrinsics to win.
  auto const first_interior = ( 2 + 1 ) / 2;
  auto const last_interior = width >= 3 ? ( width - 3 ) / 2 : size_t{ 0 };

  for( size_t y = 0; y < height; ++y )
  {
    auto const* source = base + down_step * static_cast< ptrdiff_t >( y );
    auto* destination = across.data() + y * out_width;

    if( across_step == 1 && last_interior >= first_interior )
    {
      for( size_t x = 0; x < first_interior && x < out_width; ++x )
      {
        auto const* at = columns.data() + x * 5;
        int total = 0;
        for( int k = 0; k < 5; ++k ) { total += tap[ k ] * source[ at[ k ] ]; }
        destination[ x ] = total;
      }

      auto const stop = std::min( last_interior + 1, out_width );

      for( size_t x = first_interior; x < stop; ++x )
      {
        auto const* window = source + x * 2 - 2;
        destination[ x ] = window[ 0 ] + 4 * window[ 1 ] + 6 * window[ 2 ] +
                           4 * window[ 3 ] + window[ 4 ];
      }

      for( size_t x = stop; x < out_width; ++x )
      {
        auto const* at = columns.data() + x * 5;
        int total = 0;
        for( int k = 0; k < 5; ++k ) { total += tap[ k ] * source[ at[ k ] ]; }
        destination[ x ] = total;
      }
    }
    else
    {
      for( size_t x = 0; x < out_width; ++x )
      {
        auto const* at = columns.data() + x * 5;
        int total = 0;

        for( int k = 0; k < 5; ++k )
        {
          total += tap[ k ] *
            source[ across_step * static_cast< ptrdiff_t >( at[ k ] ) ];
        }

        destination[ x ] = total;
      }
    }
  }

  viame::image_of< uint8_t > out( out_width, out_height, 1 );

  auto* const out_base = out.first_pixel();
  auto const out_across = out.w_step();
  auto const out_down = out.h_step();

  for( size_t y = 0; y < out_height; ++y )
  {
    int const* source[ 5 ];

    for( int k = 0; k < 5; ++k )
    {
      source[ k ] = across.data() +
        rows[ y * 5 + static_cast< size_t >( k ) ] * out_width;
    }

    auto* destination = out_base + out_down * static_cast< ptrdiff_t >( y );

    if( out_across == 1 )
    {
      for( size_t x = 0; x < out_width; ++x )
      {
        auto const total = source[ 0 ][ x ] + 4 * source[ 1 ][ x ] +
                           6 * source[ 2 ][ x ] + 4 * source[ 3 ][ x ] +
                           source[ 4 ][ x ];
        destination[ x ] = static_cast< uint8_t >( ( total + 128 ) >> 8 );
      }
    }
    else
    {
      for( size_t x = 0; x < out_width; ++x )
      {
        int total = 0;
        for( int k = 0; k < 5; ++k ) { total += tap[ k ] * source[ k ][ x ]; }
        destination[ out_across * static_cast< ptrdiff_t >( x ) ] =
          static_cast< uint8_t >( ( total + 128 ) >> 8 );
      }
    }
  }

  return out;
}

/// `calcSharrDeriv`: both Scharr derivatives, interleaved, as `int16`.
///
/// Scharr rather than Sobel because that is what `calcOpticalFlowPyrLK`
/// uses and the weights differ: 3, 10, 3 across the derivative rather than
/// 1, 2, 1. Both borders reflect without repeating the edge.
///
/// Written straight into \p out at \p pad_x, \p pad_y of a \p stride wide
/// buffer, because the only caller wants it inside a padded one and a plane
/// of its own would be eight megabytes written and copied for nothing.
inline void
scharr_deriv( viame::image_of< uint8_t > const& image,
              std::vector< int16_t >& out, size_t stride, size_t pad_x,
              size_t pad_y )
{
  auto const width = image.width();
  auto const height = image.height();

  std::vector< int > smoothed( width ), differenced( width );

  auto const left_of = reflect_map( -1, static_cast< long >( width ),
                                    static_cast< long >( width ) );
  auto const right_of = reflect_map( 1, static_cast< long >( width ),
                                     static_cast< long >( width ) );
  auto const above_of = reflect_map( -1, static_cast< long >( height ),
                                     static_cast< long >( height ) );
  auto const below_of = reflect_map( 1, static_cast< long >( height ),
                                     static_cast< long >( height ) );

  for( size_t y = 0; y < height; ++y )
  {
    auto const above = above_of[ y ];
    auto const below = below_of[ y ];

    auto const* const row_above =
      image.first_pixel() + image.h_step() * static_cast< ptrdiff_t >( above );
    auto const* const row_here =
      image.first_pixel() + image.h_step() * static_cast< ptrdiff_t >( y );
    auto const* const row_below =
      image.first_pixel() + image.h_step() * static_cast< ptrdiff_t >( below );
    auto const step = image.w_step();

    for( size_t x = 0; x < width; ++x )
    {
      auto const at = step * static_cast< ptrdiff_t >( x );
      auto const up = static_cast< int >( row_above[ at ] );
      auto const down = static_cast< int >( row_below[ at ] );

      smoothed[ x ] =
        ( up + down ) * 3 + static_cast< int >( row_here[ at ] ) * 10;
      differenced[ x ] = down - up;
    }

    auto* destination = out.data() +
      ( ( y + pad_y ) * stride + pad_x ) * 2;

    for( size_t x = 0; x < width; ++x )
    {
      auto const left = left_of[ x ];
      auto const right = right_of[ x ];

      destination[ x * 2 ] =
        static_cast< int16_t >( smoothed[ right ] - smoothed[ left ] );
      destination[ x * 2 + 1 ] = static_cast< int16_t >(
        ( differenced[ right ] + differenced[ left ] ) * 3 +
        differenced[ x ] * 10 );
    }
  }
}

/// The fixed point the window interpolation works in.
constexpr int lk_weight_bits = 14;
constexpr float lk_float_scale = 1.0f / ( 1 << 20 );

/// `CV_DESCALE`: shift right with a round rather than a truncate.
///
/// Templated so the window loops can stay in 32 bit, where they fit: the
/// interpolation weights sum to 2^14 and the samples are bytes or Scharr
/// gradients, so the widest product is about 2.6e8.
template < typename Int >
inline Int
descale( Int value, int bits )
{
  return ( value + ( static_cast< Int >( 1 ) << ( bits - 1 ) ) ) >> bits;
}

/// One pyramid level, with the borders the tracker reads past the edge into.
struct lk_level
{
  size_t width = 0;
  size_t height = 0;
  size_t pad_x = 0;
  size_t pad_y = 0;
  std::vector< uint8_t > first;    ///< reflect_101 past the edge
  std::vector< uint8_t > second;   ///< the same
  std::vector< int16_t > gradient; ///< zero past the edge, interleaved

  size_t stride() const { return width + 2 * pad_x; }

  /// The value at an image coordinate, which may be outside the image.
  uint8_t at_first( long x, long y ) const
  {
    return first[ static_cast< size_t >( y + static_cast< long >( pad_y ) ) *
                    stride() +
                  static_cast< size_t >( x + static_cast< long >( pad_x ) ) ];
  }

  uint8_t at_second( long x, long y ) const
  {
    return second[ static_cast< size_t >( y + static_cast< long >( pad_y ) ) *
                     stride() +
                   static_cast< size_t >( x + static_cast< long >( pad_x ) ) ];
  }

  int16_t const* at_gradient( long x, long y ) const
  {
    return gradient.data() +
           ( static_cast< size_t >( y + static_cast< long >( pad_y ) ) *
               stride() +
             static_cast< size_t >( x + static_cast< long >( pad_x ) ) ) * 2;
  }
};

inline std::vector< uint8_t >
pad_reflect( viame::image_of< uint8_t > const& image, size_t pad_x,
             size_t pad_y )
{
  auto const width = image.width();
  auto const height = image.height();
  auto const stride = width + 2 * pad_x;

  std::vector< uint8_t > out( stride * ( height + 2 * pad_y ) );

  auto const columns = reflect_map( -static_cast< long >( pad_x ),
                                    static_cast< long >( stride ),
                                    static_cast< long >( width ) );
  auto const rows = reflect_map( -static_cast< long >( pad_y ),
                                 static_cast< long >( height + 2 * pad_y ),
                                 static_cast< long >( height ) );

  // The already-padded rows are copies of each other, so a row whose source
  // row has been built before is copied rather than gathered again
  std::vector< long > built( height, -1 );

  auto const* const base = image.first_pixel();
  auto const across_step = image.w_step();
  auto const down_step = image.h_step();
  auto const packed = across_step == 1;

  for( size_t j = 0; j < rows.size(); ++j )
  {
    auto* destination = out.data() + j * stride;
    auto const y = rows[ j ];

    if( built[ y ] >= 0 )
    {
      auto const* from =
        out.data() + static_cast< size_t >( built[ y ] ) * stride;
      std::copy( from, from + stride, destination );
      continue;
    }

    auto const* source = base + down_step * static_cast< ptrdiff_t >( y );

    // The middle of the row is the row: only the two pads are a gather, and
    // they are `pad_x` wide against a frame that is two thousand across
    if( packed )
    {
      std::copy( source, source + width, destination + pad_x );
    }
    else
    {
      for( size_t i = 0; i < width; ++i )
      {
        destination[ pad_x + i ] =
          source[ across_step * static_cast< ptrdiff_t >( i ) ];
      }
    }

    for( size_t i = 0; i < pad_x; ++i )
    {
      destination[ i ] =
        source[ across_step * static_cast< ptrdiff_t >( columns[ i ] ) ];
    }

    for( size_t i = pad_x + width; i < stride; ++i )
    {
      destination[ i ] =
        source[ across_step * static_cast< ptrdiff_t >( columns[ i ] ) ];
    }

    built[ y ] = static_cast< long >( j );
  }

  return out;
}

} // namespace detail

// ----------------------------------------------------------------------------
/// Follow \p points from \p prev into \p next, as `cv::calcOpticalFlowPyrLK`.
///
/// Lucas and Kanade's method on a pyramid: over a window around each point,
/// the displacement that best explains the difference between the two frames
/// given the first frame's gradient, solved as a two by two system and
/// iterated, starting from the coarsest level so that a large motion is
/// found small.
///
/// The details are OpenCV's, and two of them decide the answer rather than
/// the third decimal place:
///
/// * the window is sampled in **fixed point** at fourteen bits, with the
///   first frame's patch held at five extra bits of headroom, so the
///   difference image is integer and the accumulation is exact;
/// * the termination epsilon is **squared once, up front**, and compared
///   against the squared step -- so the shipped 0.01 is a hundredth of a
///   pixel squared, not a hundredth of a pixel, and a port that misses it
///   stops several iterations early and lands a tenth of a pixel out.
///
/// \p status is resized to one entry per point: 1 where the point was
/// followed and 0 where it left the frame or had no corner to follow.
///
/// @param prev the frame the points are in
/// @param next the frame to find them in
/// @param points the positions to follow, as x, y
/// @param status filled with which of them were followed
/// @param params the window, the pyramid and the stopping rule
inline std::vector< std::pair< float, float > >
lucas_kanade_flow( viame::image_of< uint8_t > const& prev,
                   viame::image_of< uint8_t > const& next,
                   std::vector< std::pair< float, float > > const& points,
                   std::vector< uint8_t >& status,
                   lucas_kanade_params const& params =
                     lucas_kanade_params() )
{
  if( prev.depth() != 1 || next.depth() != 1 )
  {
    throw std::invalid_argument( "lucas_kanade_flow takes single planes" );
  }

  if( prev.width() != next.width() || prev.height() != next.height() )
  {
    throw std::invalid_argument(
      "lucas_kanade_flow takes two frames of a size" );
  }

  if( params.win_width < 1 || params.win_height < 1 || params.levels < 0 ||
      params.iterations < 1 )
  {
    throw std::invalid_argument(
      "lucas_kanade_flow: window, levels or iterations out of range" );
  }

  auto const win_w = static_cast< size_t >( params.win_width );
  auto const win_h = static_cast< size_t >( params.win_height );

  // OpenCV clamps the epsilon and squares it once, here rather than in the
  // loop, and then compares it against the squared step
  auto const epsilon = std::min( std::max( params.epsilon, 0.0 ), 10.0 );
  auto const squared_epsilon = static_cast< float >( epsilon * epsilon );

  std::vector< detail::lk_level > pyramid;

  {
    auto first = prev;
    auto second = next;

    for( int level = 0; level <= params.levels; ++level )
    {
      detail::lk_level held;
      held.width = first.width();
      held.height = first.height();
      held.pad_x = win_w;
      held.pad_y = win_h;
      held.first = detail::pad_reflect( first, win_w, win_h );
      held.second = detail::pad_reflect( second, win_w, win_h );

      // The gradient is padded with zeros rather than reflected, which is
      // `BORDER_CONSTANT` and is what OpenCV pads it with. Only the rim is
      // zeroed: the inside is about to be written over anyway, and at 1080p
      // the inside is eight of the nine megabytes.
      auto const stride = held.stride();
      held.gradient.resize( stride * ( held.height + 2 * win_h ) * 2 );

      std::fill( held.gradient.begin(),
                 held.gradient.begin() +
                   static_cast< ptrdiff_t >( win_h * stride * 2 ), 0 );
      std::fill( held.gradient.end() -
                   static_cast< ptrdiff_t >( win_h * stride * 2 ),
                 held.gradient.end(), 0 );

      for( size_t y = 0; y < held.height; ++y )
      {
        auto* row = held.gradient.data() + ( y + win_h ) * stride * 2;
        std::fill( row, row + win_w * 2, 0 );
        std::fill( row + ( win_w + held.width ) * 2, row + stride * 2, 0 );
      }

      detail::scharr_deriv( first, held.gradient, stride, win_w, win_h );

      pyramid.push_back( std::move( held ) );

      if( level >= params.levels ) { break; }

      // OpenCV stops the pyramid as soon as the **next** level would be no
      // bigger than the window in either axis, and the `<=` is its own. A
      // level the window does not fit inside is not a level a point can be
      // matched on, and building one anyway costs more than an iteration: on
      // a periodic scene it locks the coarse estimate onto the wrong repeat
      // and the point never comes back.
      auto const next_width = ( first.width() + 1 ) / 2;
      auto const next_height = ( first.height() + 1 ) / 2;

      if( next_width <= win_w || next_height <= win_h ) { break; }

      first = detail::pyr_down( first );
      second = detail::pyr_down( second );
    }
  }

  auto const count = points.size();

  std::vector< std::pair< float, float > > out( count, { 0.0f, 0.0f } );
  status.assign( count, 1 );

  auto const half_x = static_cast< float >( ( params.win_width - 1 ) * 0.5 );
  auto const half_y = static_cast< float >( ( params.win_height - 1 ) * 0.5 );

  // One point, coarsest level to finest. The levels are inside the point
  // rather than the other way about so that a pool is made once instead of
  // once per level, and because a point only ever reads its **own** answer
  // from the level above there is no barrier between them.
  auto const follow =
    [ & ]( size_t i, std::vector< int >& patch, std::vector< int >& slope_x,
           std::vector< int >& slope_y )
  {
    for( int level = static_cast< int >( pyramid.size() ) - 1; level >= 0;
         --level )
    {
      auto const& held = pyramid[ static_cast< size_t >( level ) ];
      auto const columns = static_cast< long >( held.width );
      auto const rows = static_cast< long >( held.height );

      auto const shrink = static_cast< float >( 1.0 / ( 1 << level ) );

      std::pair< float, float > here{ points[ i ].first * shrink,
                                      points[ i ].second * shrink };

      auto there = ( level == static_cast< int >( pyramid.size() ) - 1 )
        ? here
        : std::make_pair( out[ i ].first * 2.0f, out[ i ].second * 2.0f );

      out[ i ] = there;

      auto const px = here.first - half_x;
      auto const py = here.second - half_y;

      auto const ix = static_cast< long >( std::floor( px ) );
      auto const iy = static_cast< long >( std::floor( py ) );

      if( ix < -static_cast< long >( win_w ) || ix >= columns ||
          iy < -static_cast< long >( win_h ) || iy >= rows )
      {
        if( level == 0 ) { status[ i ] = 0; }
        continue;
      }

      auto const weights =
        []( float a, float b, int out[ 4 ] )
        {
          out[ 0 ] = static_cast< int >( std::nearbyint(
            ( 1.0f - a ) * ( 1.0f - b ) * ( 1 << detail::lk_weight_bits ) ) );
          out[ 1 ] = static_cast< int >( std::nearbyint(
            a * ( 1.0f - b ) * ( 1 << detail::lk_weight_bits ) ) );
          out[ 2 ] = static_cast< int >( std::nearbyint(
            ( 1.0f - a ) * b * ( 1 << detail::lk_weight_bits ) ) );
          out[ 3 ] = ( 1 << detail::lk_weight_bits ) - out[ 0 ] - out[ 1 ] -
                     out[ 2 ];
        };

      int w[ 4 ];
      weights( px - static_cast< float >( ix ), py - static_cast< float >( iy ),
               w );

      int64_t a11 = 0, a12 = 0, a22 = 0;

      auto const stride = held.stride();

      for( size_t y = 0; y < win_h; ++y )
      {
        // One index computed per window row rather than four per window
        // pixel: the four samples are the two adjacent entries of two
        // adjacent rows, and the pointers already know where they are
        auto const* top = &held.first[ 0 ] +
          static_cast< size_t >( iy + static_cast< long >( y ) +
                                 static_cast< long >( held.pad_y ) ) * stride +
          static_cast< size_t >( ix + static_cast< long >( held.pad_x ) );
        auto const* bottom = top + stride;

        auto const* g_top = held.gradient.data() +
          ( static_cast< size_t >( iy + static_cast< long >( y ) +
                                   static_cast< long >( held.pad_y ) ) * stride +
            static_cast< size_t >( ix + static_cast< long >( held.pad_x ) ) ) * 2;
        auto const* g_bottom = g_top + stride * 2;

        auto* to_patch = patch.data() + y * win_w;
        auto* to_x = slope_x.data() + y * win_w;
        auto* to_y = slope_y.data() + y * win_w;

        for( size_t x = 0; x < win_w; ++x )
        {
          auto const value = static_cast< int32_t >( detail::descale(
            static_cast< int32_t >( top[ x ] ) * w[ 0 ] +
            static_cast< int32_t >( top[ x + 1 ] ) * w[ 1 ] +
            static_cast< int32_t >( bottom[ x ] ) * w[ 2 ] +
            static_cast< int32_t >( bottom[ x + 1 ] ) * w[ 3 ],
            detail::lk_weight_bits - 5 ) );

          auto const gx = static_cast< int32_t >( detail::descale(
            static_cast< int32_t >( g_top[ x * 2 ] ) * w[ 0 ] +
            static_cast< int32_t >( g_top[ x * 2 + 2 ] ) * w[ 1 ] +
            static_cast< int32_t >( g_bottom[ x * 2 ] ) * w[ 2 ] +
            static_cast< int32_t >( g_bottom[ x * 2 + 2 ] ) * w[ 3 ],
            detail::lk_weight_bits ) );

          auto const gy = static_cast< int32_t >( detail::descale(
            static_cast< int32_t >( g_top[ x * 2 + 1 ] ) * w[ 0 ] +
            static_cast< int32_t >( g_top[ x * 2 + 3 ] ) * w[ 1 ] +
            static_cast< int32_t >( g_bottom[ x * 2 + 1 ] ) * w[ 2 ] +
            static_cast< int32_t >( g_bottom[ x * 2 + 3 ] ) * w[ 3 ],
            detail::lk_weight_bits ) );

          to_patch[ x ] = value;
          to_x[ x ] = gx;
          to_y[ x ] = gy;

          // The accumulators stay 64 bit. A saturated Scharr over a 21 by 21
          // window reaches 2.9e10, which is past what `int` holds, and
          // OpenCV's own accumulator is an `int` -- so this is the one place
          // the port declines to reproduce what OpenCV does, because what it
          // does there is wrap.
          a11 += static_cast< int64_t >( gx ) * gx;
          a12 += static_cast< int64_t >( gx ) * gy;
          a22 += static_cast< int64_t >( gy ) * gy;
        }
      }

      auto const A11 = static_cast< float >( a11 ) * detail::lk_float_scale;
      auto const A12 = static_cast< float >( a12 ) * detail::lk_float_scale;
      auto const A22 = static_cast< float >( a22 ) * detail::lk_float_scale;

      auto const determinant = A11 * A22 - A12 * A12;
      auto const smallest =
        ( A22 + A11 - std::sqrt( ( A11 - A22 ) * ( A11 - A22 ) +
                                 4.0f * A12 * A12 ) ) /
        static_cast< float >( 2 * win_w * win_h );

      if( smallest < static_cast< float >( params.min_eigen ) ||
          determinant < std::numeric_limits< float >::epsilon() )
      {
        if( level == 0 ) { status[ i ] = 0; }
        continue;
      }

      auto const inverse = 1.0f / determinant;

      auto nx = there.first - half_x;
      auto ny = there.second - half_y;

      bool have_previous = false;
      float previous_x = 0.0f, previous_y = 0.0f;

      for( int pass = 0; pass < params.iterations; ++pass )
      {
        auto const jx = static_cast< long >( std::floor( nx ) );
        auto const jy = static_cast< long >( std::floor( ny ) );

        if( jx < -static_cast< long >( win_w ) || jx >= columns ||
            jy < -static_cast< long >( win_h ) || jy >= rows )
        {
          if( level == 0 ) { status[ i ] = 0; }
          break;
        }

        int v[ 4 ];
        weights( nx - static_cast< float >( jx ),
                 ny - static_cast< float >( jy ), v );

        int64_t b1 = 0, b2 = 0;

        for( size_t y = 0; y < win_h; ++y )
        {
          auto const* top = &held.second[ 0 ] +
            static_cast< size_t >( jy + static_cast< long >( y ) +
                                   static_cast< long >( held.pad_y ) ) * stride +
            static_cast< size_t >( jx + static_cast< long >( held.pad_x ) );
          auto const* bottom = top + stride;

          auto const* from_patch = patch.data() + y * win_w;
          auto const* from_x = slope_x.data() + y * win_w;
          auto const* from_y = slope_y.data() + y * win_w;

          for( size_t x = 0; x < win_w; ++x )
          {
            auto const value = static_cast< int32_t >( detail::descale(
              static_cast< int32_t >( top[ x ] ) * v[ 0 ] +
              static_cast< int32_t >( top[ x + 1 ] ) * v[ 1 ] +
              static_cast< int32_t >( bottom[ x ] ) * v[ 2 ] +
              static_cast< int32_t >( bottom[ x + 1 ] ) * v[ 3 ],
              detail::lk_weight_bits - 5 ) );

            auto const difference = value - from_patch[ x ];

            b1 += static_cast< int64_t >( difference ) * from_x[ x ];
            b2 += static_cast< int64_t >( difference ) * from_y[ x ];
          }
        }

        auto const B1 = static_cast< float >( b1 ) * detail::lk_float_scale;
        auto const B2 = static_cast< float >( b2 ) * detail::lk_float_scale;

        auto const step_x = ( A12 * B2 - A22 * B1 ) * inverse;
        auto const step_y = ( A12 * B1 - A11 * B2 ) * inverse;

        nx += step_x;
        ny += step_y;
        out[ i ] = { nx + half_x, ny + half_y };

        if( step_x * step_x + step_y * step_y <= squared_epsilon )
        {
          break;
        }

        // Two steps that cancel are a point oscillating between two answers;
        // OpenCV splits the difference and stops rather than spending the
        // rest of the iterations on it.
        if( have_previous &&
            std::abs( step_x + previous_x ) < 0.01f &&
            std::abs( step_y + previous_y ) < 0.01f )
        {
          out[ i ] = { out[ i ].first - step_x * 0.5f,
                       out[ i ].second - step_y * 0.5f };
          break;
        }

        have_previous = true;
        previous_x = step_x;
        previous_y = step_y;
      }
    }
  };

  // The same shape `windowed_trainer` uses for its chipping: a positive count
  // is taken as given, zero asks for one thread per core up to a cap, and one
  // thread runs inline rather than paying for a pool.
  auto wanted = params.threads > 0
    ? static_cast< unsigned >( params.threads )
    : std::min( std::thread::hardware_concurrency(),
                detail::max_auto_threads );

  if( wanted == 0 ) { wanted = 1; }

  if( wanted > count ) { wanted = count > 0 ? static_cast< unsigned >( count ) : 1u; }

  if( wanted <= 1 )
  {
    std::vector< int > patch( win_w * win_h );
    std::vector< int > slope_x( win_w * win_h ), slope_y( win_w * win_h );

    for( size_t i = 0; i < count; ++i )
    {
      follow( i, patch, slope_x, slope_y );
    }
  }
  else
  {
    auto const worker =
      [ & ]( unsigned which )
      {
        // The window scratch is per thread; everything else a point touches
        // is either read only or indexed by the point itself
        std::vector< int > patch( win_w * win_h );
        std::vector< int > slope_x( win_w * win_h ), slope_y( win_w * win_h );

        for( size_t i = which; i < count; i += wanted )
        {
          follow( i, patch, slope_x, slope_y );
        }
      };

    std::vector< std::thread > pool;
    pool.reserve( wanted );

    for( unsigned which = 0; which < wanted; ++which )
    {
      pool.emplace_back( worker, which );
    }

    for( auto& one : pool ) { one.join(); }
  }

  return out;
}

} // namespace image_kernels
} // namespace viame

#endif // VIAME_IMAGE_KERNELS_OPTICAL_FLOW_H
