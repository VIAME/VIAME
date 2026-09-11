/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Geometric transforms: resize, affine, perspective and remap
///
/// What `cv::resize`, `cv::warpAffine`, `cv::warpPerspective` and
/// `cv::remap` did. All four are the same operation underneath -- for each
/// output pixel, work out where it comes from and sample there -- so what
/// differs between them is only how the source position is computed.
///
/// `resample.h` already has the bilinear resize VXL's callers were tuned
/// against, and it stays: its sample grid carries a deliberate shortfall
/// that phase 3's recordings depend on. What is here is OpenCV's geometry,
/// for the callers that were written against that instead.
///
/// The matrices are `core_types/math`'s, not Eigen's, which is what phase 6
/// left behind.

#ifndef VIAME_IMAGE_OPS_WARP_H
#define VIAME_IMAGE_OPS_WARP_H

#include <image_ops/filter.h>
#include <image_ops/pixel.h>

#include <viame/core_types/image.h>
#include <viame/core_types/matrix.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// How a sample between pixels is found.
///
/// `NEAREST`, `BILINEAR` and `AREA` are OpenCV's `INTER_NEAREST`,
/// `INTER_LINEAR` and `INTER_AREA`. Area is a mean over the source rectangle
/// an output pixel covers, which is the only one of the three that does not
/// alias when shrinking -- and the reason OpenCV's own documentation tells
/// you to use it for that.
enum class interpolation
{
  NEAREST,
  BILINEAR,
  AREA,
};

// ----------------------------------------------------------------------------
/// Bilinear sample of one plane at a real position, with a border rule.
///
/// The counterpart of `resample.h`'s `bilinear_sample`, which returns zero
/// outside the image because that is what VXL did. This one asks the border
/// rule, because a warp routinely lands outside and `cv::warpAffine` takes a
/// `borderMode` for exactly that reason.
template < typename T >
double
sample_bilinear( kwiver::vital::image_of< T > const& image, double x,
                 double y, size_t plane, border_mode mode,
                 double constant = 0.0 )
{
  auto const left = static_cast< long >( std::floor( x ) );
  auto const top = static_cast< long >( std::floor( y ) );

  auto const fx = x - static_cast< double >( left );
  auto const fy = y - static_cast< double >( top );

  auto const a =
    sample_with_border( image, left, top, plane, mode, constant );
  auto const b =
    sample_with_border( image, left + 1, top, plane, mode, constant );
  auto const c =
    sample_with_border( image, left, top + 1, plane, mode, constant );
  auto const d =
    sample_with_border( image, left + 1, top + 1, plane, mode, constant );

  return a * ( 1.0 - fx ) * ( 1.0 - fy ) + b * fx * ( 1.0 - fy ) +
         c * ( 1.0 - fx ) * fy + d * fx * fy;
}

// ----------------------------------------------------------------------------
/// Nearest sample of one plane, with a border rule.
///
/// **Rounds** the position: a sample at 1.9 comes from pixel 2. That is what
/// `cv::warpPerspective`, `cv::warpAffine` and `cv::remap` do with
/// `INTER_NEAREST`.
///
/// `cv::resize` with the same flag does not -- it truncates instead, so its
/// nearest grid is `out * scale` floored. Two conventions for one word in
/// one library; `resize` below spells its own out rather than calling this,
/// so that neither can quietly become the other.
template < typename T >
double
sample_nearest( kwiver::vital::image_of< T > const& image, double x, double y,
                size_t plane, border_mode mode, double constant = 0.0 )
{
  return sample_with_border( image, std::lround( x ), std::lround( y ), plane,
                             mode, constant );
}

// ----------------------------------------------------------------------------
/// Resize to \p width by \p height, OpenCV's way.
///
/// The sample grid is OpenCV's, and OpenCV uses **two different grids**:
///
/// * `BILINEAR` maps an output pixel's centre to `(out + 0.5) * scale - 0.5`,
///   so the two images cover the same area rather than sharing their corner
///   pixels;
/// * `NEAREST` maps it to `out * scale` instead -- no centring at all. That
///   is not an oversight to be tidied up: using the centred grid for nearest
///   shifts the result by half a source pixel and disagrees with OpenCV on a
///   third of the pixels of an enlargement.
///
/// `resample.h`'s `resize_bilinear` uses VXL's grid, which spans corner to
/// corner and carries a deliberate shortfall; which one a caller wants
/// depends on which implementation it replaces.
///
/// `AREA` is the mean over the source rectangle the output pixel covers,
/// enlarging as well as shrinking. OpenCV's documentation says it is
/// "similar to INTER_NEAREST" when enlarging; it is not, and treating it as
/// nearest disagrees by up to eighty counts.
namespace detail {

// ----------------------------------------------------------------------------
/// `cv::resize`'s fixed-point coefficients for one axis.
///
/// OpenCV does bilinear resizing of an 8-bit image in fixed point, not in
/// floating point, and the two do not agree: on natural imagery about an
/// eighth of the pixels come out one count apart. For most callers that is
/// noise. For one handing the result to a neural network it is not -- the
/// same frame resized the two ways gives `darknet` 37 detections one way and
/// 35 the other -- so this reproduces the fixed point exactly.
///
/// The coefficients are eleven-bit: `alpha0 + alpha1` is 2048.
///
/// \p clamp_ends is the one place OpenCV's two axes differ, and it is not
/// cosmetic. The horizontal setup pins a sample that would fall outside the
/// image to the edge **and drops its fraction**; the vertical setup leaves
/// both alone and clamps only the row index when it reads. Since the
/// vertical accumulation shifts twice rather than once, feeding it
/// `(2048, 0)` where OpenCV feeds it `(114, 1934)` changes the answer, and
/// clamping both axes the same way is wrong by a count along the first and
/// last row.
struct linear_axis
{
  std::vector< long > offset;
  std::vector< int > alpha0;
  std::vector< int > alpha1;
};

// ----------------------------------------------------------------------------
/// `cvRound`, which is round-half-to-even rather than round-half-away.
inline int
round_to_even( float value )
{
  return static_cast< int >( std::lrint( value ) );
}

// ----------------------------------------------------------------------------
inline linear_axis
byte_linear_axis( size_t dst_size, size_t src_size, double scale,
                  bool clamp_ends )
{
  // INTER_RESIZE_COEF_BITS is 11
  constexpr float coefficient_scale = 2048.0f;

  linear_axis axis;
  axis.offset.resize( dst_size );
  axis.alpha0.resize( dst_size );
  axis.alpha1.resize( dst_size );

  auto const last = static_cast< long >( src_size ) - 1;

  for( size_t d = 0; d < dst_size; ++d )
  {
    // Single precision, because that is what OpenCV computes it in and the
    // difference reaches the eleventh bit
    auto fraction = static_cast< float >(
      ( static_cast< double >( d ) + 0.5 ) * scale - 0.5 );

    auto source = static_cast< long >( std::floor( fraction ) );
    fraction -= static_cast< float >( source );

    if( clamp_ends )
    {
      if( source < 0 )
      {
        fraction = 0.0f;
        source = 0;
      }

      if( source >= last )
      {
        fraction = 0.0f;
        source = last;
      }
    }

    axis.offset[d] = source;
    axis.alpha0[d] = round_to_even( ( 1.0f - fraction ) * coefficient_scale );
    axis.alpha1[d] = round_to_even( fraction * coefficient_scale );
  }

  return axis;
}

// ----------------------------------------------------------------------------
/// `cv::resize` with `INTER_LINEAR` on an 8-bit image, to the count.
///
/// Horizontally into an integer buffer in units of 1/2048, then vertically
/// through the shifts `VResizeLinear`'s 8-bit specialisation uses -- which
/// are **not** the generic fixed-point cast, and that is where the count
/// comes from.
inline kwiver::vital::image_of< uint8_t >
resize_byte_linear( kwiver::vital::image_of< uint8_t > const& image,
                    size_t width, size_t height )
{
  auto const src_width = image.width();
  auto const src_height = image.height();
  auto const depth = image.depth();

  auto const scale_x =
    static_cast< double >( src_width ) / static_cast< double >( width );
  auto const scale_y =
    static_cast< double >( src_height ) / static_cast< double >( height );

  auto const horizontal =
    byte_linear_axis( width, src_width, scale_x, true );
  auto const vertical =
    byte_linear_axis( height, src_height, scale_y, false );

  // The horizontal pass, every source row, in units of 1/2048
  std::vector< int > buffer( src_height * width * depth );

  auto const clamp_column =
    [ src_width ]( long column ) -> size_t
    {
      if( column < 0 ) { return 0; }
      if( column >= static_cast< long >( src_width ) )
      {
        return src_width - 1;
      }
      return static_cast< size_t >( column );
    };

  for( size_t plane = 0; plane < depth; ++plane )
  {
    for( size_t row = 0; row < src_height; ++row )
    {
      int* out = buffer.data() + ( plane * src_height + row ) * width;

      for( size_t column = 0; column < width; ++column )
      {
        auto const first = clamp_column( horizontal.offset[ column ] );
        auto const second = clamp_column( horizontal.offset[ column ] + 1 );

        out[ column ] =
          static_cast< int >( image( first, row, plane ) ) *
            horizontal.alpha0[ column ] +
          static_cast< int >( image( second, row, plane ) ) *
            horizontal.alpha1[ column ];
      }
    }
  }

  kwiver::vital::image_of< uint8_t > out( width, height, depth );

  auto const clamp_row =
    [ src_height ]( long row ) -> size_t
    {
      if( row < 0 ) { return 0; }
      if( row >= static_cast< long >( src_height ) )
      {
        return src_height - 1;
      }
      return static_cast< size_t >( row );
    };

  for( size_t plane = 0; plane < depth; ++plane )
  {
    int const* rows = buffer.data() + plane * src_height * width;

    for( size_t j = 0; j < height; ++j )
    {
      int const* first = rows + clamp_row( vertical.offset[j] ) * width;
      int const* second = rows + clamp_row( vertical.offset[j] + 1 ) * width;

      auto const beta0 = vertical.alpha0[j];
      auto const beta1 = vertical.alpha1[j];

      for( size_t i = 0; i < width; ++i )
      {
        auto const value =
          ( ( ( beta0 * ( first[i] >> 4 ) ) >> 16 ) +
            ( ( beta1 * ( second[i] >> 4 ) ) >> 16 ) + 2 ) >> 2;

        out( i, j, plane ) =
          static_cast< uint8_t >( std::min( 255, std::max( 0, value ) ) );
      }
    }
  }

  return out;
}

} // namespace detail

template < typename T >
kwiver::vital::image_of< T >
resize( kwiver::vital::image_of< T > const& image, size_t width,
        size_t height, interpolation how = interpolation::BILINEAR,
        border_mode mode = border_mode::REPLICATE )
{
  // 8-bit bilinear is OpenCV's fixed point, which the floating point below
  // disagrees with by a count on about an eighth of the pixels. That is
  // noise to most callers and not to one feeding a network, so the exact
  // path is the one taken -- it is also what every caller replacing a
  // `cv::resize` on bytes was written against.
  if constexpr( std::is_same_v< T, uint8_t > )
  {
    if( how == interpolation::BILINEAR &&
        mode == border_mode::REPLICATE &&
        width != 0 && height != 0 &&
        image.width() != 0 && image.height() != 0 )
    {
      return detail::resize_byte_linear( image, width, height );
    }
  }

  if( width == 0 || height == 0 )
  {
    throw std::invalid_argument( "resize: the target has no area" );
  }

  if( image.width() == 0 || image.height() == 0 )
  {
    throw std::invalid_argument( "resize: the source has no area" );
  }

  auto const scale_x =
    static_cast< double >( image.width() ) / static_cast< double >( width );
  auto const scale_y =
    static_cast< double >( image.height() ) / static_cast< double >( height );

  kwiver::vital::image_of< T > out( width, height, image.depth() );

  for( size_t plane = 0; plane < image.depth(); ++plane )
  {
    for( size_t j = 0; j < height; ++j )
    {
      for( size_t i = 0; i < width; ++i )
      {
        double value = 0.0;

        if( how == interpolation::AREA )
        {
          // The source rectangle this output pixel covers, clamped to the
          // image: OpenCV averages whole and partial source pixels alike,
          // and on an integer ratio -- which is the common case -- that is
          // exactly a block mean.
          auto const x0 = static_cast< double >( i ) * scale_x;
          auto const x1 = x0 + scale_x;
          auto const y0 = static_cast< double >( j ) * scale_y;
          auto const y1 = y0 + scale_y;

          auto const first_x = static_cast< long >( std::floor( x0 ) );
          auto const last_x = static_cast< long >( std::ceil( x1 ) );
          auto const first_y = static_cast< long >( std::floor( y0 ) );
          auto const last_y = static_cast< long >( std::ceil( y1 ) );

          double total = 0.0;
          double weight = 0.0;

          for( long y = first_y; y < last_y; ++y )
          {
            auto const covered_y =
              std::min( y1, static_cast< double >( y ) + 1.0 ) -
              std::max( y0, static_cast< double >( y ) );

            if( covered_y <= 0.0 ) { continue; }

            for( long x = first_x; x < last_x; ++x )
            {
              auto const covered_x =
                std::min( x1, static_cast< double >( x ) + 1.0 ) -
                std::max( x0, static_cast< double >( x ) );

              if( covered_x <= 0.0 ) { continue; }

              auto const area = covered_x * covered_y;
              total += area *
                sample_with_border( image, x, y, plane, mode );
              weight += area;
            }
          }

          value = ( weight > 0.0 ) ? total / weight : 0.0;
        }
        else if( how == interpolation::NEAREST )
        {
          // Floored, not rounded, and off the uncentred grid: see the note
          // on `sample_nearest`, which rounds because the warps do
          value = sample_with_border(
            image,
            static_cast< long >(
              std::floor( static_cast< double >( i ) * scale_x ) ),
            static_cast< long >(
              std::floor( static_cast< double >( j ) * scale_y ) ),
            plane, mode );
        }
        else
        {
          value = sample_bilinear(
            image, ( static_cast< double >( i ) + 0.5 ) * scale_x - 0.5,
            ( static_cast< double >( j ) + 0.5 ) * scale_y - 0.5, plane,
            mode );
        }

        out( i, j, plane ) = saturate_pixel< T >( value );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Warp by a three by three homography, which is `cv::warpPerspective`.
///
/// \p transform maps **source to destination**, and the warp inverts it to
/// find where each output pixel comes from. That is OpenCV's convention with
/// its default flags; `cv::WARP_INVERSE_MAP` is the other one, and a caller
/// holding an already-inverted matrix wants `warp_perspective_inverse`.
///
/// @param image the source
/// @param transform source to destination, three by three
/// @param width the output width, the source's when zero
/// @param height the output height, the source's when zero
template < typename T >
kwiver::vital::image_of< T >
warp_perspective_inverse( kwiver::vital::image_of< T > const& image,
                          kwiver::vital::matrix_3x3d const& inverse,
                          size_t width, size_t height,
                          interpolation how = interpolation::BILINEAR,
                          border_mode mode = border_mode::CONSTANT,
                          double constant = 0.0 )
{
  if( width == 0 || height == 0 )
  {
    throw std::invalid_argument(
      "warp_perspective_inverse: the target has no area" );
  }

  kwiver::vital::image_of< T > out( width, height, image.depth() );

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      auto const dx = static_cast< double >( i );
      auto const dy = static_cast< double >( j );

      auto const w = inverse( 2, 0 ) * dx + inverse( 2, 1 ) * dy +
                     inverse( 2, 2 );

      // A point on the horizon has no source; the border rule answers for it
      if( w == 0.0 )
      {
        for( size_t plane = 0; plane < image.depth(); ++plane )
        {
          out( i, j, plane ) = saturate_pixel< T >( constant );
        }
        continue;
      }

      auto const sx =
        ( inverse( 0, 0 ) * dx + inverse( 0, 1 ) * dy + inverse( 0, 2 ) ) / w;
      auto const sy =
        ( inverse( 1, 0 ) * dx + inverse( 1, 1 ) * dy + inverse( 1, 2 ) ) / w;

      for( size_t plane = 0; plane < image.depth(); ++plane )
      {
        auto const value = ( how == interpolation::NEAREST )
          ? sample_nearest( image, sx, sy, plane, mode, constant )
          : sample_bilinear( image, sx, sy, plane, mode, constant );

        out( i, j, plane ) = saturate_pixel< T >( value );
      }
    }
  }

  return out;
}

/// `warp_perspective_inverse` with the forward matrix, inverted here.
template < typename T >
kwiver::vital::image_of< T >
warp_perspective( kwiver::vital::image_of< T > const& image,
                  kwiver::vital::matrix_3x3d const& transform,
                  size_t width = 0, size_t height = 0,
                  interpolation how = interpolation::BILINEAR,
                  border_mode mode = border_mode::CONSTANT,
                  double constant = 0.0 )
{
  if( std::abs( transform.determinant() ) <
      std::numeric_limits< double >::epsilon() )
  {
    throw std::invalid_argument(
      "warp_perspective: the transform is not invertible" );
  }

  return warp_perspective_inverse(
    image, transform.inverse(),
    width ? width : image.width(), height ? height : image.height(),
    how, mode, constant );
}

// ----------------------------------------------------------------------------
/// Warp by a two by three affine matrix, which is `cv::warpAffine`.
///
/// The same operation as the perspective warp with a bottom row of
/// (0, 0, 1), and written as one so that the two cannot drift apart.
template < typename T >
kwiver::vital::image_of< T >
warp_affine( kwiver::vital::image_of< T > const& image,
             kwiver::vital::matrix_< 2, 3, double > const& transform,
             size_t width = 0, size_t height = 0,
             interpolation how = interpolation::BILINEAR,
             border_mode mode = border_mode::CONSTANT,
             double constant = 0.0 )
{
  kwiver::vital::matrix_3x3d full;

  for( unsigned r = 0; r < 2; ++r )
  {
    for( unsigned c = 0; c < 3; ++c )
    {
      full( r, c ) = transform( r, c );
    }
  }

  full( 2, 0 ) = 0.0;
  full( 2, 1 ) = 0.0;
  full( 2, 2 ) = 1.0;

  return warp_perspective( image, full, width, height, how, mode, constant );
}

// ----------------------------------------------------------------------------
/// The affine matrix `cv::getRotationMatrix2D` builds.
///
/// A rotation of \p degrees counter-clockwise about (\p centre_x,
/// \p centre_y), scaled by \p scale, as a two by three that maps source to
/// destination.
inline kwiver::vital::matrix_< 2, 3, double >
rotation_matrix_2d( double centre_x, double centre_y, double degrees,
                    double scale = 1.0 )
{
  auto const radians = degrees * 3.14159265358979323846 / 180.0;
  auto const alpha = std::cos( radians ) * scale;
  auto const beta = std::sin( radians ) * scale;

  kwiver::vital::matrix_< 2, 3, double > out;

  out( 0, 0 ) = alpha;
  out( 0, 1 ) = beta;
  out( 0, 2 ) = ( 1.0 - alpha ) * centre_x - beta * centre_y;
  out( 1, 0 ) = -beta;
  out( 1, 1 ) = alpha;
  out( 1, 2 ) = beta * centre_x + ( 1.0 - alpha ) * centre_y;

  return out;
}

// ----------------------------------------------------------------------------
/// Sample \p image at the positions two maps give, which is `cv::remap`.
///
/// \p map_x and \p map_y are the source position for each output pixel, one
/// plane each, and the same size as each other; the output takes that size.
/// This is what a rectification uses: the calibration step produces the two
/// maps once and every frame is sampled through them.
template < typename T, typename M >
kwiver::vital::image_of< T >
remap( kwiver::vital::image_of< T > const& image,
       kwiver::vital::image_of< M > const& map_x,
       kwiver::vital::image_of< M > const& map_y,
       interpolation how = interpolation::BILINEAR,
       border_mode mode = border_mode::CONSTANT, double constant = 0.0 )
{
  if( map_x.width() != map_y.width() || map_x.height() != map_y.height() )
  {
    throw std::invalid_argument( "remap: the two maps differ in size" );
  }

  if( map_x.depth() != 1 || map_y.depth() != 1 )
  {
    throw std::invalid_argument( "remap: a map has one plane" );
  }

  kwiver::vital::image_of< T > out( map_x.width(), map_x.height(),
                                    image.depth() );

  for( size_t j = 0; j < map_x.height(); ++j )
  {
    for( size_t i = 0; i < map_x.width(); ++i )
    {
      auto const sx = static_cast< double >( map_x( i, j, 0 ) );
      auto const sy = static_cast< double >( map_y( i, j, 0 ) );

      for( size_t plane = 0; plane < image.depth(); ++plane )
      {
        auto const value = ( how == interpolation::NEAREST )
          ? sample_nearest( image, sx, sy, plane, mode, constant )
          : sample_bilinear( image, sx, sy, plane, mode, constant );

        out( i, j, plane ) = saturate_pixel< T >( value );
      }
    }
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
