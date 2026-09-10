/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Colour space conversion and Bayer demosaicing
///
/// What `cv::cvtColor` did, which is the single most-asked-for OpenCV call in
/// the tree -- twelve files reach for it. The arithmetic here is OpenCV's,
/// not a textbook's, because the outputs are held to a recording of what
/// OpenCV produced: the fixed-point coefficients, the rounding, the 0..255
/// hue scaling and the L*a*b* offsets are all its.
///
/// The `image_ops` convention applies: `kwiver::vital::image_of< T >` in and
/// out, planes rather than interleaved channels, no OpenCV type anywhere.
/// Channel order is RGB -- what `vital::image` carries and what a decoded
/// file gives -- so where OpenCV names a conversion BGR2X the same operation
/// is `rgb_to_x` here, and `swap_rb` is what turns one into the other.

#ifndef VIAME_IMAGE_OPS_COLOR_H
#define VIAME_IMAGE_OPS_COLOR_H

#include <image_ops/pixel.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace viame {
namespace image_ops {

// ----------------------------------------------------------------------------
/// Luminance coefficients, ITU-R BT.601, which is what `cv::cvtColor` uses.
///
/// `channels.h` has BT.709's, because that is what VXL used and the phase 3
/// recordings are held to it. The two differ by a few counts on a saturated
/// colour, so which one a caller wants depends on which implementation it is
/// replacing -- a distinction worth keeping visible rather than picking one.
constexpr double bt601_red = 0.299;
constexpr double bt601_green = 0.587;
constexpr double bt601_blue = 0.114;

namespace detail {

/// OpenCV's fixed-point luminance weights: BT.601 at 14 fractional bits.
constexpr int gray_shift = 14;
constexpr int gray_red = 4899;      // 0.299 * (1 << 14), rounded
constexpr int gray_green = 9617;    // 0.587 * (1 << 14), rounded
constexpr int gray_blue = 1868;     // 0.114 * (1 << 14), rounded

/// One integer luminance, rounded the way OpenCV rounds it.
inline int
gray_of( int red, int green, int blue )
{
  return ( red * gray_red + green * gray_green + blue * gray_blue +
           ( 1 << ( gray_shift - 1 ) ) ) >> gray_shift;
}

/// Whether \p image has at least \p wanted planes, and complain if not.
template < typename T >
void
require_planes( kwiver::vital::image_of< T > const& image, size_t wanted,
                char const* what )
{
  if( image.depth() < wanted )
  {
    throw std::invalid_argument(
      std::string( what ) + " needs at least " + std::to_string( wanted ) +
      " planes, got " + std::to_string( image.depth() ) );
  }
}

} // namespace detail

// ----------------------------------------------------------------------------
/// Three planes to one, by BT.601 luminance.
///
/// The integer path for 8 and 16 bit is OpenCV's fixed point, so an 8 bit
/// image converts to the same bytes it would have. Floating point goes
/// through doubles, as OpenCV's float path does.
///
/// @param image RGB, three planes or more; any extra are ignored
template < typename T >
kwiver::vital::image_of< T >
rgb_to_gray( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_gray" );

  kwiver::vital::image_of< T > out( image.width(), image.height(), 1 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      if constexpr( std::is_integral< T >::value )
      {
        out( i, j, 0 ) = static_cast< T >(
          detail::gray_of( static_cast< int >( image( i, j, 0 ) ),
                           static_cast< int >( image( i, j, 1 ) ),
                           static_cast< int >( image( i, j, 2 ) ) ) );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >(
          image( i, j, 0 ) * bt601_red +
          image( i, j, 1 ) * bt601_green +
          image( i, j, 2 ) * bt601_blue );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// One plane to three, by copying it into each.
template < typename T >
kwiver::vital::image_of< T >
gray_to_rgb( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 1, "gray_to_rgb" );

  kwiver::vital::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const value = image( i, j, 0 );
      out( i, j, 0 ) = value;
      out( i, j, 1 ) = value;
      out( i, j, 2 ) = value;
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// Planes 0 and 2 exchanged, which is RGB to BGR and back.
///
/// A fourth plane, where there is one, is left where it is: that is alpha,
/// and `cv::cvtColor`'s BGRA2RGBA does the same.
template < typename T >
kwiver::vital::image_of< T >
swap_rb( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 3, "swap_rb" );

  kwiver::vital::image_of< T > out( image.width(), image.height(),
                                    image.depth() );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      out( i, j, 0 ) = image( i, j, 2 );
      out( i, j, 1 ) = image( i, j, 1 );
      out( i, j, 2 ) = image( i, j, 0 );

      for( size_t plane = 3; plane < image.depth(); ++plane )
      {
        out( i, j, plane ) = image( i, j, plane );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// RGB to HSV, in OpenCV's 8 bit scaling.
///
/// Hue is 0..179 -- degrees halved, so that it fits a byte -- and saturation
/// and value are 0..255. That scaling is OpenCV's and is what
/// `random_hue_shift` and the enhancer's saturation path are written against;
/// a caller wanting real degrees multiplies by two.
///
/// For a floating point pixel type the ranges are the conventional ones:
/// hue 0..360, saturation and value 0..1, which is also what OpenCV does.
template < typename T >
kwiver::vital::image_of< T >
rgb_to_hsv( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_hsv" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  kwiver::vital::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const red = static_cast< double >( image( i, j, 0 ) );
      auto const green = static_cast< double >( image( i, j, 1 ) );
      auto const blue = static_cast< double >( image( i, j, 2 ) );

      auto const high = std::max( { red, green, blue } );
      auto const low = std::min( { red, green, blue } );
      auto const span = high - low;

      double hue = 0.0;

      if( span > 0.0 )
      {
        if( high == red )
        {
          hue = 60.0 * ( green - blue ) / span;
        }
        else if( high == green )
        {
          hue = 120.0 + 60.0 * ( blue - red ) / span;
        }
        else
        {
          hue = 240.0 + 60.0 * ( red - green ) / span;
        }

        if( hue < 0.0 )
        {
          hue += 360.0;
        }
      }

      auto const saturation = ( high > 0.0 ) ? span / high : 0.0;

      if constexpr( integral )
      {
        // Halved degrees so hue fits a byte, which is OpenCV's 8 bit
        // convention and what every caller here expects
        out( i, j, 0 ) = saturate_pixel< T >( hue * 0.5 );
        out( i, j, 1 ) = saturate_pixel< T >( saturation * top );
        out( i, j, 2 ) = saturate_pixel< T >( high );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( hue );
        out( i, j, 1 ) = static_cast< T >( saturation );
        out( i, j, 2 ) = static_cast< T >( high );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// HSV back to RGB, undoing `rgb_to_hsv` in the same scaling.
template < typename T >
kwiver::vital::image_of< T >
hsv_to_rgb( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 3, "hsv_to_rgb" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  kwiver::vital::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      double hue = static_cast< double >( image( i, j, 0 ) );
      double saturation = static_cast< double >( image( i, j, 1 ) );
      double value = static_cast< double >( image( i, j, 2 ) );

      if constexpr( integral )
      {
        hue *= 2.0;
        saturation /= top;
      }

      hue = std::fmod( hue, 360.0 );
      if( hue < 0.0 )
      {
        hue += 360.0;
      }

      auto const sector = static_cast< int >( hue / 60.0 ) % 6;
      auto const fraction = hue / 60.0 - std::floor( hue / 60.0 );

      auto const p = value * ( 1.0 - saturation );
      auto const q = value * ( 1.0 - saturation * fraction );
      auto const t = value * ( 1.0 - saturation * ( 1.0 - fraction ) );

      double red = value;
      double green = t;
      double blue = p;

      switch( sector )
      {
        case 0: red = value; green = t;     blue = p;     break;
        case 1: red = q;     green = value; blue = p;     break;
        case 2: red = p;     green = value; blue = t;     break;
        case 3: red = p;     green = q;     blue = value; break;
        case 4: red = t;     green = p;     blue = value; break;
        default: red = value; green = p;    blue = q;     break;
      }

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel< T >( red );
        out( i, j, 1 ) = saturate_pixel< T >( green );
        out( i, j, 2 ) = saturate_pixel< T >( blue );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( red );
        out( i, j, 1 ) = static_cast< T >( green );
        out( i, j, 2 ) = static_cast< T >( blue );
      }
    }
  }

  return out;
}

namespace detail {

/// sRGB's transfer function, inverted: an encoded value to linear light.
inline double
srgb_to_linear( double value )
{
  return ( value <= 0.04045 ) ? value / 12.92
                              : std::pow( ( value + 0.055 ) / 1.055, 2.4 );
}

inline double
linear_to_srgb( double value )
{
  return ( value <= 0.0031308 ) ? value * 12.92
                                : 1.055 * std::pow( value, 1.0 / 2.4 ) - 0.055;
}

/// CIE's f, and its inverse, for the L*a*b* transfer.
inline double
lab_f( double value )
{
  constexpr double epsilon = 216.0 / 24389.0;
  constexpr double kappa = 24389.0 / 27.0;

  return ( value > epsilon ) ? std::cbrt( value )
                             : ( kappa * value + 16.0 ) / 116.0;
}

inline double
lab_f_inverse( double value )
{
  constexpr double epsilon = 216.0 / 24389.0;
  constexpr double kappa = 24389.0 / 27.0;

  auto const cubed = value * value * value;

  return ( cubed > epsilon ) ? cubed : ( 116.0 * value - 16.0 ) / kappa;
}

/// D65, which is the white point OpenCV's sRGB conversions assume.
constexpr double white_x = 0.950456;
constexpr double white_y = 1.0;
constexpr double white_z = 1.088754;

} // namespace detail

// ----------------------------------------------------------------------------
/// RGB to CIE L*a*b*, in OpenCV's 8 bit scaling.
///
/// L is 0..255 rather than 0..100, and a and b are shifted by 128 so that
/// they fit an unsigned byte: `L * 255 / 100`, `a + 128`, `b + 128`. That is
/// what `cv::cvtColor` produces for CV_8U and what the enhancer's CLAHE path
/// -- which equalises L and leaves a and b alone -- is written against.
///
/// A floating point pixel type gets the real ranges: L 0..100, a and b
/// roughly -128..127, again as OpenCV does.
template < typename T >
kwiver::vital::image_of< T >
rgb_to_lab( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_lab" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  kwiver::vital::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const red = detail::srgb_to_linear(
        static_cast< double >( image( i, j, 0 ) ) / ( integral ? top : 1.0 ) );
      auto const green = detail::srgb_to_linear(
        static_cast< double >( image( i, j, 1 ) ) / ( integral ? top : 1.0 ) );
      auto const blue = detail::srgb_to_linear(
        static_cast< double >( image( i, j, 2 ) ) / ( integral ? top : 1.0 ) );

      // sRGB to XYZ, D65
      auto const x = 0.412453 * red + 0.357580 * green + 0.180423 * blue;
      auto const y = 0.212671 * red + 0.715160 * green + 0.072169 * blue;
      auto const z = 0.019334 * red + 0.119193 * green + 0.950227 * blue;

      auto const fx = detail::lab_f( x / detail::white_x );
      auto const fy = detail::lab_f( y / detail::white_y );
      auto const fz = detail::lab_f( z / detail::white_z );

      auto const lightness = 116.0 * fy - 16.0;
      auto const a = 500.0 * ( fx - fy );
      auto const b = 200.0 * ( fy - fz );

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel< T >( lightness * 255.0 / 100.0 );
        out( i, j, 1 ) = saturate_pixel< T >( a + 128.0 );
        out( i, j, 2 ) = saturate_pixel< T >( b + 128.0 );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( lightness );
        out( i, j, 1 ) = static_cast< T >( a );
        out( i, j, 2 ) = static_cast< T >( b );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// L*a*b* back to RGB, undoing `rgb_to_lab` in the same scaling.
template < typename T >
kwiver::vital::image_of< T >
lab_to_rgb( kwiver::vital::image_of< T > const& image )
{
  detail::require_planes( image, 3, "lab_to_rgb" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  kwiver::vital::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      double lightness = static_cast< double >( image( i, j, 0 ) );
      double a = static_cast< double >( image( i, j, 1 ) );
      double b = static_cast< double >( image( i, j, 2 ) );

      if constexpr( integral )
      {
        lightness = lightness * 100.0 / 255.0;
        a -= 128.0;
        b -= 128.0;
      }

      auto const fy = ( lightness + 16.0 ) / 116.0;
      auto const fx = fy + a / 500.0;
      auto const fz = fy - b / 200.0;

      auto const x = detail::lab_f_inverse( fx ) * detail::white_x;
      auto const y = detail::lab_f_inverse( fy ) * detail::white_y;
      auto const z = detail::lab_f_inverse( fz ) * detail::white_z;

      auto const red = detail::linear_to_srgb(
         3.240479 * x - 1.537150 * y - 0.498535 * z );
      auto const green = detail::linear_to_srgb(
        -0.969256 * x + 1.875992 * y + 0.041556 * z );
      auto const blue = detail::linear_to_srgb(
         0.055648 * x - 0.204043 * y + 1.057311 * z );

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel< T >( red * top );
        out( i, j, 1 ) = saturate_pixel< T >( green * top );
        out( i, j, 2 ) = saturate_pixel< T >( blue * top );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( red );
        out( i, j, 1 ) = static_cast< T >( green );
        out( i, j, 2 ) = static_cast< T >( blue );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// The four Bayer mosaics, named the way every camera and OpenCV name them.
///
/// The letters are the top-left two by two, read in rows: `BG` is
///
///     B G
///     G R
///
/// which is what every VIAME pipeline that debayers asks for, and what a
/// camera datasheet means by the word.
///
/// **OpenCV's constants read the other way round.** `cv::COLOR_BayerBG2RGB`
/// on a mosaic with blue at (0, 0) returns the channels reversed; the one
/// that decodes it correctly to RGB is `COLOR_BayerRG2RGB`, and
/// `COLOR_BayerBG2BGR` gives the same pixels in BGR. `debayer_filter` used
/// `COLOR_BayerBG2BGR` for `pattern: BG` and the OpenCV bridge swapped the
/// channels on the way into `vital::image`, so the two cancelled and the
/// config letter meant what it says. This enum keeps that meaning -- the
/// letter is the mosaic, not OpenCV's spelling of it.
enum class bayer_pattern
{
  BG,
  GB,
  RG,
  GR,
};

namespace detail {

/// Which colour a Bayer site holds: 0 red, 1 green, 2 blue.
///
/// Derived from the pattern's top-left two by two rather than tabulated for
/// every position, so the four patterns are one expression and a new one
/// would be a new row rather than a new branch.
inline int
bayer_colour( bayer_pattern pattern, size_t i, size_t j )
{
  // (i, j) parity within the two by two, and what each corner holds
  static constexpr int corners[ 4 ][ 4 ] = {
    // top-left, top-right, bottom-left, bottom-right
    { 2, 1, 1, 0 },   // BG
    { 1, 2, 0, 1 },   // GB
    { 0, 1, 1, 2 },   // RG
    { 1, 0, 2, 1 },   // GR
  };

  auto const corner = ( j % 2 ) * 2 + ( i % 2 );
  return corners[ static_cast< int >( pattern ) ][ corner ];
}

/// A pixel with the coordinates mirrored back inside the image.
///
/// Reflection rather than clamping, because a demosaic reads one and two
/// pixels out and clamping there duplicates a sample of the wrong colour,
/// which shows as a coloured fringe on the border. OpenCV reflects too.
template < typename T >
T
reflected( kwiver::vital::image_of< T > const& image, long i, long j,
           size_t plane = 0 )
{
  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );

  if( i < 0 ) { i = -i; }
  if( j < 0 ) { j = -j; }
  if( i >= width ) { i = 2 * width - 2 - i; }
  if( j >= height ) { j = 2 * height - 2 - j; }

  i = std::max( 0L, std::min( width - 1, i ) );
  j = std::max( 0L, std::min( height - 1, j ) );

  return image( static_cast< size_t >( i ), static_cast< size_t >( j ),
                plane );
}

} // namespace detail

// ----------------------------------------------------------------------------
/// Bilinear demosaic of a Bayer mosaic into RGB.
///
/// The classic four-neighbour interpolation: a missing green is the mean of
/// its four edge neighbours, a missing red or blue is the mean of the two or
/// four sites that hold it. It is what `cv::cvtColor`'s `COLOR_BayerBG2RGB`
/// does -- the plain one, not the VNG or edge-aware variants, which OpenCV
/// spells `_VNG` and `_EA` and which nothing here asks for.
///
/// @param image the mosaic, one plane
/// @param pattern which colour the top-left two by two holds
template < typename T >
kwiver::vital::image_of< T >
demosaic( kwiver::vital::image_of< T > const& image, bayer_pattern pattern )
{
  if( image.depth() != 1 )
  {
    throw std::invalid_argument( "demosaic needs a single plane mosaic" );
  }

  kwiver::vital::image_of< T > out( image.width(), image.height(), 3 );

  auto const mean_of = []( std::initializer_list< double > values )
  {
    double total = 0.0;
    for( auto const value : values ) { total += value; }
    return total / static_cast< double >( values.size() );
  };

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const here = detail::bayer_colour( pattern, i, j );
      auto const x = static_cast< long >( i );
      auto const y = static_cast< long >( j );

      auto const at = [ & ]( long di, long dj )
      {
        return static_cast< double >(
          detail::reflected( image, x + di, y + dj ) );
      };

      double red = 0.0;
      double green = 0.0;
      double blue = 0.0;

      if( here == 1 )
      {
        // A green site. One of its horizontal neighbours is red and the
        // other vertical pair is blue, or the other way round; which is
        // decided by what the pixel to the left holds.
        green = at( 0, 0 );

        auto const left_colour = detail::bayer_colour(
          pattern, i == 0 ? 1 : i - 1, j );

        auto const horizontal = mean_of( { at( -1, 0 ), at( 1, 0 ) } );
        auto const vertical = mean_of( { at( 0, -1 ), at( 0, 1 ) } );

        if( left_colour == 0 )
        {
          red = horizontal;
          blue = vertical;
        }
        else
        {
          blue = horizontal;
          red = vertical;
        }
      }
      else
      {
        // A red or a blue site. The greens are the four edge neighbours and
        // the other colour is the four corners.
        green = mean_of( { at( -1, 0 ), at( 1, 0 ), at( 0, -1 ), at( 0, 1 ) } );

        auto const diagonal =
          mean_of( { at( -1, -1 ), at( 1, -1 ), at( -1, 1 ), at( 1, 1 ) } );

        if( here == 0 )
        {
          red = at( 0, 0 );
          blue = diagonal;
        }
        else
        {
          blue = at( 0, 0 );
          red = diagonal;
        }
      }

      out( i, j, 0 ) = saturate_pixel< T >( red );
      out( i, j, 1 ) = saturate_pixel< T >( green );
      out( i, j, 2 ) = saturate_pixel< T >( blue );
    }
  }

  return out;
}

} // namespace image_ops
} // namespace viame

#endif
