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
/// The `image_kernels` convention applies: `viame::image_of< T >` in and
/// out, planes rather than interleaved channels, no OpenCV type anywhere.
/// Channel order is RGB -- what `viame::image` carries and what a decoded
/// file gives -- so where OpenCV names a conversion BGR2X the same operation
/// is `rgb_to_x` here, and `swap_rb` is what turns one into the other.

#ifndef VIAME_IMAGE_KERNELS_COLOR_H
#define VIAME_IMAGE_KERNELS_COLOR_H

#include <image_kernels/pixel.h>

#include <viame/core_types/image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace viame {
namespace image_kernels {

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

/// OpenCV's fixed-point luminance weights: BT.601 at 15 fractional bits.
///
/// Fifteen, not the fourteen OpenCV used up to version 4. The three weights
/// are not simply doubled by the extra bit -- green gains a count and blue
/// loses one -- so the two disagree by one grey level on about a quarter of
/// a percent of pixels, which is enough to move a corner or a descriptor
/// downstream. Found by fitting the shift and the three weights to what the
/// installed `cv2.cvtColor(..., COLOR_RGB2GRAY)` returns: these reproduce it
/// on every one of 22500 random colours, and the fourteen bit set does not.
constexpr int gray_shift = 15;
constexpr int gray_red = 9798;      // 0.299 * (1 << 15), rounded
constexpr int gray_green = 19235;   // 0.587 * (1 << 15), rounded
constexpr int gray_blue = 3735;     // 0.114 * (1 << 15), rounded

/// One integer luminance, rounded the way OpenCV rounds it.
inline int
gray_of( int red, int green, int blue )
{
  return ( red * gray_red + green * gray_green + blue * gray_blue +
           ( 1 << ( gray_shift - 1 ) ) ) >> gray_shift;
}

/// \p value, forced to a float and not carried on in wider precision.
///
/// The colour conversions that follow OpenCV's float paths depend on each
/// multiply and each subtract rounding to float where OpenCV's does. GCC
/// defaults to `-ffp-contract=fast`, which is free to fuse `1 - s * h` into a
/// single multiply-add and skip the rounding in the middle, and the answers
/// then differ by a count on a few pixels in ten thousand -- and differ with
/// the build flags, which is worse than differing. Passing the product
/// through a volatile blocks the fusion wherever it would happen.
inline float
exact( float value )
{
  volatile float held = value;

  return held;
}

/// Whether \p image has at least \p wanted planes, and complain if not.
template < typename T >
void
require_planes( viame::image_of< T > const& image, size_t wanted,
                char const* what )
{
  if( image.depth() < wanted )
  {
    throw std::invalid_argument(
      std::string( what ) + " needs at least " + std::to_string( wanted ) +
      " planes, got " + std::to_string( image.depth() ) );
  }
}

/// OpenCV's fixed-point tables for the 8 bit RGB to HSV conversion.
///
/// `cv::cvtColor` does not take 8 bit RGB to HSV through the real-valued
/// formula. It has a dedicated integer path -- two reciprocal tables and a
/// 12 bit shift -- and the difference is not a rounding one: the real-valued
/// hue wraps where the integer one does not, so on 398486 of the 16777216
/// triples they disagree by up to **179**, which is half the circle rather
/// than a grey level.
struct hsv_tables
{
  static constexpr int shift = 12;

  int range;
  std::array< int, 256 > saturation;
  std::array< int, 256 > hue;

  explicit hsv_tables( int hue_range )
    : range( hue_range )
  {
    saturation[ 0 ] = 0;
    hue[ 0 ] = 0;

    for( int i = 1; i < 256; ++i )
    {
      auto const at = static_cast< size_t >( i );

      saturation[ at ] = static_cast< int >( std::nearbyint(
        static_cast< double >( 255 << shift ) / i ) );
      hue[ at ] = static_cast< int >( std::nearbyint(
        static_cast< double >( range << shift ) / ( 6.0 * i ) ) );
    }
  }
};

/// The half-degree tables, or the full-byte ones OpenCV's `_FULL` codes use.
///
/// `cv::COLOR_RGB2HSV_FULL` is the same integer path with the circle spread
/// over the whole byte instead of over 180, which in OpenCV is one `hrange`
/// threaded through the table and the wrap. It is threaded through here the
/// same way rather than being a second copy of the loop.
inline hsv_tables const&
hsv_table( bool full = false )
{
  static hsv_tables const halved( 180 );
  static hsv_tables const whole( 256 );

  return full ? whole : halved;
}

// ----------------------------------------------------------------------------
/// A signed hue in degrees, as the halved degrees an 8 bit image holds.
///
/// Halve, round half up, and only then wrap a negative into range. That is
/// the order `cv::cvtColor`'s **HSV** conversion uses, and the order
/// matters: a hue of -0.98 degrees wrapped first becomes 180 after halving,
/// which is outside the range, and halved first becomes 0. Its **HLS**
/// conversion wraps first and does produce the 180, so `rgb_to_hls` does not
/// call this. The two disagreeing is OpenCV's, not a choice made here.
inline double
wrapped_half( double hue )
{
  auto const half = std::floor( hue * 0.5 + 0.5 );
  return half < 0.0 ? half + 180.0 : half;
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
viame::image_of< T >
rgb_to_gray( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_gray" );

  viame::image_of< T > out( image.width(), image.height(), 1 );

  auto const width = image.width();
  auto const step = image.w_step();
  for( size_t j = 0; j < image.height(); ++j )
  {
    auto const* red = image.first_pixel() + j * image.h_step();
    auto const* green = red + image.d_step();
    auto const* blue = green + image.d_step();
    auto* destination = out.first_pixel() + j * out.h_step();
    for( size_t i = 0; i < width; ++i )
    {
      if constexpr( std::is_integral< T >::value )
      {
        destination[i] = static_cast< T >(
          detail::gray_of( static_cast< int >( red[i * step] ),
                           static_cast< int >( green[i * step] ),
                           static_cast< int >( blue[i * step] ) ) );
      }
      else
      {
        destination[i] = static_cast< T >(
          red[i * step] * bt601_red + green[i * step] * bt601_green +
          blue[i * step] * bt601_blue );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// One plane to three, by copying it into each.
template < typename T >
viame::image_of< T >
gray_to_rgb( viame::image_of< T > const& image )
{
  detail::require_planes( image, 1, "gray_to_rgb" );

  viame::image_of< T > out( image.width(), image.height(), 3 );

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
viame::image_of< T >
swap_rb( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "swap_rb" );

  viame::image_of< T > out( image.width(), image.height(),
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
///
/// With `full` set, an 8 bit hue covers 0..255 instead of 0..179 --
/// `cv::COLOR_RGB2HSV_FULL`. A float image is unaffected, as OpenCV's `_FULL`
/// code is, since 0..360 is already the whole circle.
template < typename T >
viame::image_of< T >
rgb_to_hsv( viame::image_of< T > const& image, bool full = false )
{
  detail::require_planes( image, 3, "rgb_to_hsv" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    auto const& table = detail::hsv_table( full );
    constexpr int shift = detail::hsv_tables::shift;
    constexpr int half = 1 << ( shift - 1 );

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const red = static_cast< int >( image( i, j, 0 ) );
        auto const green = static_cast< int >( image( i, j, 1 ) );
        auto const blue = static_cast< int >( image( i, j, 2 ) );

        auto const high = std::max( { red, green, blue } );
        auto const low = std::min( { red, green, blue } );
        auto const span = high - low;

        // The two masks are 0 or -1, and the arithmetic is OpenCV's own: the
        // branch is done with `&` so that all three cases cost the same.
        auto const is_red = ( high == red ) ? -1 : 0;
        auto const is_green = ( high == green ) ? -1 : 0;

        auto hue =
          ( is_red & ( green - blue ) ) +
          ( ~is_red & ( ( is_green & ( blue - red + 2 * span ) ) +
                        ( ~is_green & ( red - green + 4 * span ) ) ) );

        hue = ( hue * table.hue[ static_cast< size_t >( span ) ] + half ) >>
              shift;

        if( hue < 0 )
        {
          hue += table.range;
        }

        auto const saturation =
          ( span * table.saturation[ static_cast< size_t >( high ) ] + half ) >>
          shift;

        out( i, j, 0 ) = saturate_pixel< T >( hue );
        out( i, j, 1 ) = saturate_pixel< T >( saturation );
        out( i, j, 2 ) = saturate_pixel< T >( high );
      }
    }

    return out;
  }

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

      }

      auto const saturation = ( high > 0.0 ) ? span / high : 0.0;

      if constexpr( integral )
      {
        // Halved degrees so hue fits a byte, which is OpenCV's 8 bit
        // convention and what every caller here expects.
        //
        // The wrap comes **after** the halving, which is OpenCV's order and
        // not the obvious one. A hue of -0.98 degrees wrapped first is
        // 359.02, and halved and rounded that is 180 -- outside the range
        // entirely. Halved first it is -0.49, which rounds to zero and needs
        // no wrap. One pixel in six thousand of a real image lands there.
        out( i, j, 0 ) = saturate_pixel< T >( detail::wrapped_half( hue ) );
        out( i, j, 1 ) = saturate_pixel< T >( saturation * top );
        out( i, j, 2 ) = saturate_pixel< T >( high );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( hue < 0.0 ? hue + 360.0 : hue );
        out( i, j, 1 ) = static_cast< T >( saturation );
        out( i, j, 2 ) = static_cast< T >( high );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// RGB to HLS: hue, lightness, saturation.
///
/// `cv::COLOR_RGB2HLS`. The hue is the same angle HSV uses and is halved for
/// an integral type the same way; the lightness is the midpoint of the
/// extremes rather than the maximum, and the saturation is measured against
/// how far that midpoint is from the middle of the range. The plane order is
/// hue, **lightness**, saturation -- not the HSV order with two of them
/// exchanged, which is the mistake this is easiest to make.
template < typename T >
viame::image_of< T >
rgb_to_hls( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_hls" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  // `cv::cvtColor` runs the 8 bit HLS conversion in **float**, not double, and
  // the difference is not small: in double this disagreed with cv2 on 2113124
  // of the 16777216 triples, all of them by one count of hue. In float, on
  // 1744. The scalings are OpenCV's too -- the byte is divided by 255 going in,
  // the hue is halved rather than scaled by 180/360 after being built in
  // degrees, and lightness and saturation are multiplied by 255 on the way out.
  if constexpr( std::is_same< T, uint8_t >::value )
  {
    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        // `src[i] * (1.f / 255.f)`, a multiply by the reciprocal rather than a
        // division. In float those are not the same operation, and taking the
        // division instead is worth 200000 triples.
        constexpr float inverse = 1.0f / 255.0f;

        auto const red = static_cast< float >( image( i, j, 0 ) ) * inverse;
        auto const green = static_cast< float >( image( i, j, 1 ) ) * inverse;
        auto const blue = static_cast< float >( image( i, j, 2 ) ) * inverse;

        auto const high = std::max( { red, green, blue } );
        auto const low = std::min( { red, green, blue } );
        auto const span = high - low;
        auto const lightness = ( high + low ) * 0.5f;

        auto hue = 0.0f;
        auto saturation = 0.0f;

        if( span > std::numeric_limits< float >::epsilon() )
        {
          saturation = ( lightness < 0.5f )
                       ? span / ( high + low )
                       : span / ( 2.0f - high - low );

          auto const rate = 60.0f / span;

          if( high == red )
          {
            hue = ( green - blue ) * rate;
          }
          else if( high == green )
          {
            hue = ( blue - red ) * rate + 120.0f;
          }
          else
          {
            hue = ( red - green ) * rate + 240.0f;
          }

          if( hue < 0.0f )
          {
            hue += 360.0f;
          }
        }

        out( i, j, 0 ) = saturate_pixel_even< T >( hue * 0.5f );
        out( i, j, 1 ) = saturate_pixel_even< T >( lightness * 255.0f );
        out( i, j, 2 ) = saturate_pixel_even< T >( saturation * 255.0f );
      }
    }

    return out;
  }

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const scale = integral ? top : 1.0;

      auto const red = static_cast< double >( image( i, j, 0 ) ) / scale;
      auto const green = static_cast< double >( image( i, j, 1 ) ) / scale;
      auto const blue = static_cast< double >( image( i, j, 2 ) ) / scale;

      auto const high = std::max( { red, green, blue } );
      auto const low = std::min( { red, green, blue } );
      auto const span = high - low;
      auto const lightness = ( high + low ) * 0.5;

      double hue = 0.0;
      double saturation = 0.0;

      if( span > 0.0 )
      {
        saturation = ( lightness < 0.5 )
                     ? span / ( high + low )
                     : span / ( 2.0 - high - low );

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

      }

      // **Wrapped before halving, unlike HSV above.** `cv::cvtColor`'s two
      // conversions disagree about the order, and on the one pixel of the
      // recorded fixture where the hue comes out slightly negative they give
      // 0 for HSV and 180 for HLS. Not a difference anyone would predict,
      // and only a recording finds it.
      if( hue < 0.0 )
      {
        hue += 360.0;
      }

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel< T >( hue * 0.5 );
        out( i, j, 1 ) = saturate_pixel< T >( lightness * top );
        out( i, j, 2 ) = saturate_pixel< T >( saturation * top );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( hue );
        out( i, j, 1 ) = static_cast< T >( lightness );
        out( i, j, 2 ) = static_cast< T >( saturation );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// HLS back to RGB, undoing `rgb_to_hls` in the same scaling.
template < typename T >
viame::image_of< T >
hls_to_rgb( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "hls_to_rgb" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto hue = static_cast< double >( image( i, j, 0 ) );
      auto lightness = static_cast< double >( image( i, j, 1 ) );
      auto saturation = static_cast< double >( image( i, j, 2 ) );

      if constexpr( integral )
      {
        hue *= 2.0;
        lightness /= top;
        saturation /= top;
      }

      auto const chroma = ( 1.0 - std::abs( 2.0 * lightness - 1.0 ) ) *
                          saturation;
      auto const sector = hue / 60.0;
      auto const second = chroma *
                          ( 1.0 - std::abs( std::fmod( sector, 2.0 ) - 1.0 ) );
      auto const lift = lightness - chroma * 0.5;

      double red = 0.0;
      double green = 0.0;
      double blue = 0.0;

      switch( static_cast< int >( std::floor( sector ) ) % 6 )
      {
        case 0: red = chroma; green = second; break;
        case 1: red = second; green = chroma; break;
        case 2: green = chroma; blue = second; break;
        case 3: green = second; blue = chroma; break;
        case 4: red = second; blue = chroma; break;
        default: red = chroma; blue = second; break;
      }

      auto const scale = integral ? top : 1.0;

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel< T >( ( red + lift ) * scale );
        out( i, j, 1 ) = saturate_pixel< T >( ( green + lift ) * scale );
        out( i, j, 2 ) = saturate_pixel< T >( ( blue + lift ) * scale );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( red + lift );
        out( i, j, 1 ) = static_cast< T >( green + lift );
        out( i, j, 2 ) = static_cast< T >( blue + lift );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// HSV back to RGB, undoing `rgb_to_hsv` in the same scaling.
///
/// `full` is the flag `rgb_to_hsv` takes: an 8 bit hue over 0..255 rather
/// than 0..179, which is `cv::COLOR_HSV2RGB_FULL`.
template < typename T >
viame::image_of< T >
hsv_to_rgb( viame::image_of< T > const& image, bool full = false )
{
  detail::require_planes( image, 3, "hsv_to_rgb" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    // OpenCV's `HSV2RGB_b`: hue scaled to sixths, the other two to 0..1, the
    // six sector formula in **float**, and back to bytes by
    // **truncation**. The truncation is the surprise and it is not a reading
    // of `saturate_cast`, which rounds: the vectorised body finishes with
    // `v_trunc`, and the scalar tail that rounds only ever sees the last few
    // pixels of a buffer. Truncating is therefore what a recording of
    // `cv2.cvtColor( ..., COLOR_HSV2RGB )` contains for all but a handful of
    // its pixels, and rounding instead is a count low on 74 percent of them.
    //
    // Reproduced, not corrected, for that reason. What is left over is 1758
    // of the 11796480 legal 8-bit triples, one count each, where OpenCV's
    // float chain lands a hair either side of an integer that this one hits
    // exactly; the arithmetic is written in OpenCV's own order and no
    // reassociation tried gets closer.
    static constexpr int sector_data[ 6 ][ 3 ] =
      { { 1, 3, 0 }, { 1, 0, 2 }, { 3, 0, 1 },
        { 0, 2, 1 }, { 0, 1, 3 }, { 2, 1, 0 } };

    // 255, not the 256 the forward table uses. OpenCV's two `hrange`
    // choices are not each other's inverse: `RGB2HSV_FULL` spreads the
    // circle over 256 and `HSV2RGB_FULL` reads it back over 255, so a
    // round trip through the pair is not the identity even in principle.
    // Both halves are reproduced as they are.
    float const hue_scale = full ? 6.0f / 255.0f : 6.0f / 180.0f;
    constexpr float inverse = 1.0f / 255.0f;

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        // Each of these three is a product that a subtract further down
        // reads, so each is a fusion candidate in its own right -- `1 - s`
        // where s is `byte * inverse` would become one fnmsub. OpenCV cannot
        // fuse there, because it computes the scalings once into a vector
        // register and reads them four times, so the rounding happens. The
        // guard makes that true here too, and the `std::fma` below asks for
        // the one fusion OpenCV does get.
        auto hue = detail::exact(
          static_cast< float >( image( i, j, 0 ) ) * hue_scale );
        auto const saturation = detail::exact(
          static_cast< float >( image( i, j, 1 ) ) * inverse );
        auto const value = detail::exact(
          static_cast< float >( image( i, j, 2 ) ) * inverse );

        auto sector = static_cast< int >( hue );
        hue = detail::exact( hue - static_cast< float >( sector ) );
        sector %= 6;
        if( sector < 0 ) { sector += 6; }

        // `1 - s * h` is one **fused** multiply-add, not a multiply and a
        // subtract. OpenCV writes it as two universal intrinsics, but GCC
        // contracts `_mm256_sub_ps( one, _mm256_mul_ps( s, h ) )` into a
        // single `fnmadd` on any host with FMA, and the rounding that is
        // skipped in the middle is visible: unfused leaves 1758 of the
        // 11796480 triples a count out, fused leaves **none**. `1 - s` is
        // left alone, since there is no product inside it to fuse with.
        // Fused on the half-degree scale and unfused on the full one, for
        // the same reason the store below truncates on one and rounds on the
        // other: the two are different roads through OpenCV. Fusing costs 89
        // of the 50331648 plane values on an exhaustive `_FULL` sweep and
        // not fusing costs 1758 of the 11796480 legal triples on the other,
        // so each road gets the form that is exact for it.
        float const paired = full
          ? detail::exact( 1.0f - detail::exact( saturation * hue ) )
          : std::fma( -saturation, hue, 1.0f );
        float const complement = full
          ? detail::exact(
              1.0f - detail::exact( saturation * ( 1.0f - hue ) ) )
          : std::fma( -saturation, 1.0f - hue, 1.0f );

        float const tab[ 4 ] = {
          value,
          value * ( 1.0f - saturation ),
          value * paired,
          value * complement };

        for( int k = 0; k < 3; ++k )
        {
          // sector_data gives blue, green, red in that order
          auto const scaled = tab[ sector_data[ sector ][ 2 - k ] ] * 255.0f;

          // Truncated on the half-degree scale and **rounded** on the full
          // one. Not a choice: `COLOR_HSV2RGB` on a wide row finishes in the
          // vectorised body, which truncates, and `COLOR_HSV2RGB_FULL` does
          // not take that body at all -- its hue does not fit the fixed
          // point the SIMD path is written for -- so it ends in the scalar
          // `saturate_cast< uchar >`, which rounds. Over a 256 by 256 block
          // of every legal triple, truncating is exact for the first and
          // wrong on 34 percent of the second, and rounding is the other way
          // round. This is the same split as finding 2.56, with the cause
          // named: `_FULL` is the scalar road every time.
          auto const quantised = full
            ? static_cast< int >( std::nearbyint( scaled ) )
            : static_cast< int >( scaled );

          out( i, j, static_cast< size_t >( k ) ) = static_cast< uint8_t >(
            std::min( std::max( quantised, 0 ), 255 ) );
        }
      }
    }

    return out;
  }

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

// ----------------------------------------------------------------------------
/// OpenCV's fixed-point tables for the 8 bit L*a*b* transfer.
///
/// `cv::cvtColor` does not take 8 bit RGB through the real-valued formula
/// above. It goes through two integer tables -- a gamma table on the input
/// byte and a cube-root table on the fixed-point XYZ value -- and the two
/// answers disagree by up to two counts. The goldens are recorded from cv2
/// with a tolerance of zero, so close is not a pass; these are its tables.
///
/// The subtlety is in the cube-root table. Its argument is computed in
/// float, which the index-to-value division has to match, but its cube root
/// is OpenCV's own, and at two entries that lands on the far side of a
/// rounding tie from a correctly rounded one. Those two are named below
/// rather than derived, because deriving them would mean reproducing that
/// cube root. One of them is reachable, and it alone moves 17645 of the
/// 16777216 possible triples -- a table entry only shows up where the value
/// it feeds is itself near a rounding boundary, which is about one time in
/// seventy.
struct lab_tables
{
  static constexpr int gamma_shift = 3;
  static constexpr int lab_shift = 12;
  static constexpr int lab_shift2 = lab_shift + gamma_shift;
  static constexpr int cbrt_size = 256 * 3 / 2 * ( 1 << gamma_shift );

  /// `(116 * 255 + 50) / 100` and its offset, which put L on 0..255.
  static constexpr int lightness_scale = ( 116 * 255 + 50 ) / 100;
  static constexpr int lightness_shift =
    -( ( 16 * 255 * ( 1 << lab_shift2 ) + 50 ) / 100 );

  std::array< int, 256 > gamma;
  std::array< int, cbrt_size > cbrt;
  std::array< int, 9 > coefficient;

  /// \p straight selects OpenCV's `linearGammaTab_b` over its sRGB one, which
  /// is what `COLOR_LBGR2Lab` uses -- the input is already linear light, so the
  /// transfer curve is the identity and the table is just the shift.
  explicit lab_tables( bool straight = false )
  {
    for( int i = 0; i < 256; ++i )
    {
      auto const value = straight
                         ? static_cast< double >( i ) / 255.0
                         : srgb_to_linear( static_cast< double >( i ) / 255.0 );

      gamma[ static_cast< size_t >( i ) ] = static_cast< int >(
        std::nearbyint( 255.0 * ( 1 << gamma_shift ) * value ) );
    }

    // CIE's own rationals rather than the truncated 0.008856 and 7.787 that
    // older references print; the difference lands on a rounding boundary.
    constexpr float knee = 216.0f / 24389.0f;
    constexpr float slope = 841.0f / 108.0f;
    constexpr float offset = 4.0f / 29.0f;

    for( int i = 0; i < cbrt_size; ++i )
    {
      auto const x = static_cast< float >( i ) /
                     static_cast< float >( 255 * ( 1 << gamma_shift ) );
      auto const value = ( x < knee )
                         ? static_cast< double >( x * slope + offset )
                         : std::cbrt( static_cast< double >( x ) );

      cbrt[ static_cast< size_t >( i ) ] = static_cast< int >(
        std::nearbyint( ( 1 << lab_shift2 ) * value ) );
    }

    // The two ties. Both products are a ten-thousandth above the halfway
    // point -- 9454.500194 and 37088.500396 -- so rounding sends them up,
    // and OpenCV's cube root, being a shade low, sends them down. Only the
    // first can be reached: no row of the matrix below sums to more than one
    // once divided by the white point, so no index exceeds 2040.
    cbrt[ 49 ] = 9454;
    cbrt[ 2958 ] = 37088;

    constexpr double rows[ 9 ] = { 0.412453, 0.357580, 0.180423,
                                   0.212671, 0.715160, 0.072169,
                                   0.019334, 0.119193, 0.950227 };
    constexpr double white[ 3 ] = { white_x, white_y, white_z };

    for( int i = 0; i < 9; ++i )
    {
      coefficient[ static_cast< size_t >( i ) ] = static_cast< int >(
        std::nearbyint( rows[ i ] / white[ i / 3 ] * ( 1 << lab_shift ) ) );
    }
  }
};

inline lab_tables const&
lab_table()
{
  static lab_tables const tables;
  return tables;
}

inline lab_tables const&
lab_linear_table()
{
  static lab_tables const tables( true );
  return tables;
}

// ----------------------------------------------------------------------------
/// OpenCV's fixed-point tables for the 8 bit L*a*b* to RGB transfer.
///
/// A different implementation from the forward one and from its own float
/// path: `cv::cvtColor`'s float answer rounded to a byte disagrees with its
/// 8 bit answer by a count on 2.8% of triples, so neither the real-valued
/// formula nor the float path reproduces this. Three tables and a 14 bit
/// fixed point do.
///
/// `lightness` carries both y and f(y) per L. `transfer` is f inverted, over
/// the whole range f(x) and f(z) can reach, biased by `ab_floor` so that a
/// negative one indexes it. `gamma` is the sRGB transfer on 12 bits, which is
/// also where the byte comes from: its entries are already 0..255.
struct lab_inverse_tables
{
  static constexpr int lab_shift = 12;
  static constexpr int base_shift = 14;
  static constexpr int gamma_shift = 12;
  static constexpr int base = 1 << base_shift;
  static constexpr int shift = lab_shift + ( base_shift - gamma_shift );
  static constexpr int gamma_size = 1 << gamma_shift;
  static constexpr int ab_floor = -8145;
  static constexpr int ab_size = base * 9 / 4;
  /// Where f's linear leg gives way to its cube, on this scaling.
  static constexpr int knee = 3390;

  std::array< int, 256 > lightness_y;
  std::array< int, 256 > lightness_f;
  std::array< int, ab_size > transfer;
  std::array< int, gamma_size > gamma;
  std::array< int, 9 > coefficient;
  bool m_straight = false;

  explicit lab_inverse_tables( bool straight = false )
  {
    m_straight = straight;

    // Built in float, which is what OpenCV builds them in: i * 100 * 16384
    // passes 2^24 before L reaches 3, so the rounding is part of the table.
    for( int i = 0; i < 256; ++i )
    {
      auto const at = static_cast< size_t >( i );

      if( i <= 20 )
      {
        lightness_y[ at ] = static_cast< int >( std::nearbyint(
          static_cast< float >( i * base * 20 * 9 ) /
          static_cast< float >( 17 * 29 * 29 * 29 ) ) );
        lightness_f[ at ] = static_cast< int >( std::nearbyint(
          static_cast< float >( base ) *
          ( 16.0f / 116.0f +
            static_cast< float >( i * 5 ) /
            static_cast< float >( 3 * 17 * 29 ) ) ) );
      }
      else
      {
        auto const f = static_cast< float >( i * 100 * base ) /
                       static_cast< float >( 255 * 116 ) +
                       static_cast< float >( 16 * base ) / 116.0f;

        lightness_f[ at ] = static_cast< int >( std::nearbyint( f ) );
        lightness_y[ at ] = static_cast< int >( std::nearbyint(
          f * f * f / static_cast< float >( base * base ) ) );
      }
    }

    // Integer division truncates toward zero here, and the index runs
    // negative, so this is not a floor. OpenCV's arithmetic, kept as it is.
    for( int i = ab_floor; i < ab_size + ab_floor; ++i )
    {
      auto const at = static_cast< size_t >( i - ab_floor );

      transfer[ at ] = ( i <= knee )
                       ? i * 108 / 841 - ( base * 16 / 116 * 108 / 841 )
                       : i * i / base * i / base;
    }

    for( int i = 0; i < gamma_size; ++i )
    {
      auto const value = static_cast< double >( i ) / gamma_size;

      // `linearInvGammaTab_b` **truncates** where the sRGB one rounds, which
      // is OpenCV's `cvTrunc` against its `cvRound`.
      gamma[ static_cast< size_t >( i ) ] = straight
        ? static_cast< int >( 255.0 * value )
        : static_cast< int >( std::nearbyint( 255.0 * linear_to_srgb( value ) ) );
    }

    constexpr double rows[ 9 ] = { 3.240479, -1.53715, -0.498535,
                                   -0.969256, 1.875991, 0.041556,
                                   0.055648, -0.204043, 1.057311 };
    constexpr double white[ 3 ] = { white_x, white_y, white_z };

    for( int i = 0; i < 9; ++i )
    {
      coefficient[ static_cast< size_t >( i ) ] = static_cast< int >(
        std::nearbyint( ( 1 << lab_shift ) * rows[ i ] * white[ i % 3 ] ) );
    }
  }
};

inline lab_inverse_tables const&
lab_inverse_table()
{
  static lab_inverse_tables const tables;
  return tables;
}

inline lab_inverse_tables const&
lab_inverse_linear_table()
{
  static lab_inverse_tables const tables( true );
  return tables;
}

/// OpenCV's rounding right shift, which floors on a negative value.
inline int
descale( int value, int bits )
{
  return ( value + ( 1 << ( bits - 1 ) ) ) >> bits;
}

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
viame::image_of< T >
rgb_to_lab( viame::image_of< T > const& image, bool linear = false )
{
  detail::require_planes( image, 3, "rgb_to_lab" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    auto const& table = linear ? detail::lab_linear_table()
                               : detail::lab_table();
    constexpr int shift = detail::lab_tables::lab_shift;
    constexpr int shift2 = detail::lab_tables::lab_shift2;
    constexpr int half = 128 * ( 1 << shift2 );
    auto const& c = table.coefficient;

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const r = table.gamma[ image( i, j, 0 ) ];
        auto const g = table.gamma[ image( i, j, 1 ) ];
        auto const b = table.gamma[ image( i, j, 2 ) ];

        int f[ 3 ];

        for( int k = 0; k < 3; ++k )
        {
          auto const raw = detail::descale(
            r * c[ 3 * k ] + g * c[ 3 * k + 1 ] + b * c[ 3 * k + 2 ], shift );

          // The table is sized for a value this cannot reach -- every row of
          // the matrix sums to one once divided by the white point, so the
          // largest is 2040 against 3072 -- but an index is an index.
          f[ k ] = table.cbrt[ static_cast< size_t >( std::min(
            std::max( raw, 0 ), detail::lab_tables::cbrt_size - 1 ) ) ];
        }

        out( i, j, 0 ) = saturate_pixel< T >( detail::descale(
          detail::lab_tables::lightness_scale * f[ 1 ] +
          detail::lab_tables::lightness_shift, shift2 ) );
        out( i, j, 1 ) = saturate_pixel< T >(
          detail::descale( 500 * ( f[ 0 ] - f[ 1 ] ) + half, shift2 ) );
        out( i, j, 2 ) = saturate_pixel< T >(
          detail::descale( 200 * ( f[ 1 ] - f[ 2 ] ) + half, shift2 ) );
      }
    }

    return out;
  }

  // A floating point channel is **clamped to [0, 1]** before anything else,
  // which is what OpenCV does and is not an approximation of it: a float
  // image holding 0..65535 -- which is what the enhancer produces when it
  // casts a 16 bit image up -- converts to a uniform L=100, a=b=0, so the
  // clamp is the whole of the answer there rather than a detail of it.
  auto const channel = [ & ]( size_t i, size_t j, size_t plane )
  {
    auto const value =
      static_cast< double >( image( i, j, plane ) ) / ( integral ? top : 1.0 );
    auto const bounded = integral ? value
                                  : std::min( std::max( value, 0.0 ), 1.0 );

    return linear ? bounded : detail::srgb_to_linear( bounded );
  };

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const red = channel( i, j, 0 );
      auto const green = channel( i, j, 1 );
      auto const blue = channel( i, j, 2 );

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
viame::image_of< T >
lab_to_rgb( viame::image_of< T > const& image, bool linear = false )
{
  detail::require_planes( image, 3, "lab_to_rgb" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    using tables = detail::lab_inverse_tables;
    auto const& table = linear ? detail::lab_inverse_linear_table()
                               : detail::lab_inverse_table();
    auto const& c = table.coefficient;

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        auto const l = static_cast< int >( image( i, j, 0 ) );
        auto const a = static_cast< int >( image( i, j, 1 ) );
        auto const b = static_cast< int >( image( i, j, 2 ) );

        auto const y = table.lightness_y[ static_cast< size_t >( l ) ];
        auto const f = table.lightness_f[ static_cast< size_t >( l ) ];

        // a and b are divided by 500 and 200 by reciprocal multiplication,
        // and the +1 on the second is OpenCV's, not a typo of ours.
        auto const across = ( ( 5 * a * 53687 + ( 1 << 7 ) ) >> 13 ) -
                            128 * tables::base / 500;
        auto const along = ( ( b * 41943 + ( 1 << 4 ) ) >> 9 ) -
                           128 * tables::base / 200 + 1;

        auto const at = [ & ]( int value )
        {
          return table.transfer[ static_cast< size_t >( std::min(
            std::max( value - tables::ab_floor, 0 ),
            tables::ab_size - 1 ) ) ];
        };

        int const xyz[ 3 ] = { at( f + across ), y, at( f - along ) };

        for( int k = 0; k < 3; ++k )
        {
          auto const raw = detail::descale( c[ 3 * k ] * xyz[ 0 ] +
                                            c[ 3 * k + 1 ] * xyz[ 1 ] +
                                            c[ 3 * k + 2 ] * xyz[ 2 ],
                                            tables::shift );

          out( i, j, static_cast< size_t >( k ) ) = saturate_pixel< T >(
            table.gamma[ static_cast< size_t >( std::min(
              std::max( raw, 0 ), tables::gamma_size - 1 ) ) ] );
        }
      }
    }

    return out;
  }

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

      // OpenCV clamps the linear triple before the transfer, not after it,
      // so an out of gamut L*a*b* triple comes back as a corner of the cube
      // rather than as a channel that overshoots and then saturates.
      auto const transfer = [ linear ]( double value )
      {
        auto const bounded = std::min( std::max( value, 0.0 ), 1.0 );

        return linear ? bounded : detail::linear_to_srgb( bounded );
      };

      auto const red = transfer(
         3.240479 * x - 1.537150 * y - 0.498535 * z );
      auto const green = transfer(
        -0.969256 * x + 1.875992 * y + 0.041556 * z );
      auto const blue = transfer(
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
/// `COLOR_BayerBG2BGR` gives the same pixels in BGR. Here the letter is the
/// **mosaic** -- `BG` means blue at (0, 0) -- not OpenCV's spelling of it,
/// so `bayer_pattern::BG` is `COLOR_BayerRG2RGB`.
///
/// `debayer_filter`'s config letter follows OpenCV's spelling rather than
/// this one, and so decodes the wrong pattern; it maps its letters across
/// on the way in, and says why. Do not assume the two agree.
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
reflected( viame::image_of< T > const& image, long i, long j,
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
viame::image_of< T >
demosaic( viame::image_of< T > const& image, bayer_pattern pattern )
{
  if( image.depth() != 1 )
  {
    throw std::invalid_argument( "demosaic needs a single plane mosaic" );
  }

  viame::image_of< T > out( image.width(), image.height(), 3 );

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

  // The outermost ring is replicated from the pixel beside it rather than
  // interpolated, which is what `cv::cvtColor` does: its Bayer conversions
  // compute the interior and then copy the first and last interior row and
  // column outwards. Rows before columns, so a corner ends up holding its
  // diagonal neighbour -- checked against the recording, which has
  // `out(0, 0) == out(1, 1)` exactly.
  //
  // Reflection is still what the interior reads through: a pixel at index
  // one reaches index minus one, and that read is part of the value the
  // recording agrees with.
  if( out.width() >= 3 && out.height() >= 3 )
  {
    for( size_t plane = 0; plane < 3; ++plane )
    {
      for( size_t i = 0; i < out.width(); ++i )
      {
        out( i, 0, plane ) = out( i, 1, plane );
        out( i, out.height() - 1, plane ) = out( i, out.height() - 2, plane );
      }

      for( size_t j = 0; j < out.height(); ++j )
      {
        out( 0, j, plane ) = out( 1, j, plane );
        out( out.width() - 1, j, plane ) = out( out.width() - 2, j, plane );
      }
    }
  }

  return out;
}


namespace detail {

// ----------------------------------------------------------------------------
/// The two shifts OpenCV's linear colour conversions use.
///
/// Both are `cvtColor`'s own: XYZ is a 3 by 3 matrix in 12 bit fixed point
/// and the luma-chroma pair is 14 bit. Rounding is OpenCV's `CV_DESCALE`,
/// which adds half before the shift, so a negative product rounds **towards
/// positive infinity** rather than away from zero -- an arithmetic right
/// shift, not a division. That matters for chroma, which is signed.
constexpr int xyz_shift = 12;
constexpr int yuv_shift = 14;

inline int
descale( long long value, int shift )
{
  return static_cast< int >( ( value + ( 1LL << ( shift - 1 ) ) ) >> shift );
}

/// sRGB to CIE XYZ, D65, and back. OpenCV's numbers to six decimals.
constexpr double rgb_to_xyz_matrix[ 9 ] = {
  0.412453, 0.357580, 0.180423,
  0.212671, 0.715160, 0.072169,
  0.019334, 0.119193, 0.950227 };

constexpr double xyz_to_rgb_matrix[ 9 ] = {
   3.240479, -1.537150, -0.498535,
  -0.969256,  1.875991,  0.041556,
   0.055648, -0.204043,  1.057311 };

/// The luma weights, and the four chroma scalings OpenCV rounds them to.
///
/// `COLOR_RGB2YCrCb` and `COLOR_RGB2YUV` are the same luma and two different
/// chroma pairs -- 0.713 and 0.564 against 0.492 and 0.877 -- which is why
/// they are one function here with a flag rather than two that would drift.
constexpr int luma_red = 4899;      // 0.299 << 14
constexpr int luma_green = 9617;    // 0.587 << 14
constexpr int luma_blue = 1868;     // 0.114 << 14

constexpr int ycrcb_cr = 11682;     // 0.713 << 14
constexpr int ycrcb_cb = 9241;      // 0.564 << 14
constexpr int yuv_u = 8061;         // 0.492 << 14
constexpr int yuv_v = 14369;        // 0.877 << 14

constexpr int ycrcb_to_red = 22987;
constexpr int ycrcb_to_green_cr = -11698;
constexpr int ycrcb_to_green_cb = -5636;
constexpr int ycrcb_to_blue = 29049;

constexpr int yuv_to_blue = 33292;
constexpr int yuv_to_green_u = -6472;
constexpr int yuv_to_green_v = -9519;
constexpr int yuv_to_red = 18678;

/// The same eight numbers **unrounded**, which is what OpenCV's float path
/// uses. Dividing the fixed-point forms back by 1 << 14 is not the same
/// thing: 11682 / 16384 is 0.7130127, and the difference from 0.713 shows up
/// at 2e-05 in a float conversion, which is two hundred times the ULP.
constexpr float ycrcb_cr_f = 0.713f;
constexpr float ycrcb_cb_f = 0.564f;
constexpr float yuv_u_f = 0.492f;
constexpr float yuv_v_f = 0.877f;

constexpr float ycrcb_to_red_f = 1.403f;
constexpr float ycrcb_to_green_cr_f = -0.714f;
constexpr float ycrcb_to_green_cb_f = -0.344f;
constexpr float ycrcb_to_blue_f = 1.773f;

constexpr float yuv_to_blue_f = 2.032f;
constexpr float yuv_to_green_u_f = -0.395f;
constexpr float yuv_to_green_v_f = -0.581f;
constexpr float yuv_to_red_f = 1.140f;

/// A fixed-point copy of one of the 3 by 3 matrices above.
inline std::array< int, 9 > const&
scaled_matrix( double const ( &source )[ 9 ], std::array< int, 9 >& store )
{
  for( size_t at = 0; at < 9; ++at )
  {
    store[ at ] = static_cast< int >(
      std::nearbyint( source[ at ] * ( 1 << xyz_shift ) ) );
  }

  return store;
}

inline std::array< int, 9 > const&
rgb_to_xyz_fixed()
{
  static std::array< int, 9 > store;
  static auto const& built = scaled_matrix( rgb_to_xyz_matrix, store );

  return built;
}

inline std::array< int, 9 > const&
xyz_to_rgb_fixed()
{
  static std::array< int, 9 > store;
  static auto const& built = scaled_matrix( xyz_to_rgb_matrix, store );

  return built;
}

/// The u' and v' of the white point, which `L*u*v*` measures chroma from.
constexpr double luv_white_u = 0.19793943;
constexpr double luv_white_v = 0.46831096;

} // namespace detail

// ----------------------------------------------------------------------------
/// RGB to CIE XYZ, which is `cv::COLOR_RGB2XYZ`.
///
/// Note what this is **not**: the XYZ a colour scientist means. OpenCV
/// applies the matrix to the encoded sRGB value without linearising it
/// first, so the result is a linear combination of gamma-encoded numbers.
/// Reproduced as OpenCV has it, because a caller converting back expects to
/// get its image again. `rgb_to_lab` and `rgb_to_luv` do linearise, which is
/// the inconsistency OpenCV carries and not one introduced here.
///
/// 8 bit goes through OpenCV's 12 bit fixed point and is exact over a random
/// 256 by 256 block; float32 is the matrix in float and is within one ULP.
template < typename T >
viame::image_of< T >
rgb_to_xyz( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_xyz" );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    auto const& matrix = detail::rgb_to_xyz_fixed();

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        long long const red = image( i, j, 0 );
        long long const green = image( i, j, 1 );
        long long const blue = image( i, j, 2 );

        for( size_t plane = 0; plane < 3; ++plane )
        {
          auto const value = detail::descale(
            red * matrix[ plane * 3 ] + green * matrix[ plane * 3 + 1 ] +
            blue * matrix[ plane * 3 + 2 ], detail::xyz_shift );

          out( i, j, plane ) = saturate_pixel< T >( value );
        }
      }
    }

    return out;
  }

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const red = static_cast< float >( image( i, j, 0 ) );
      auto const green = static_cast< float >( image( i, j, 1 ) );
      auto const blue = static_cast< float >( image( i, j, 2 ) );

      for( size_t plane = 0; plane < 3; ++plane )
      {
        auto const value =
          red * static_cast< float >( detail::rgb_to_xyz_matrix[ plane * 3 ] ) +
          green *
            static_cast< float >( detail::rgb_to_xyz_matrix[ plane * 3 + 1 ] ) +
          blue *
            static_cast< float >( detail::rgb_to_xyz_matrix[ plane * 3 + 2 ] );

        out( i, j, plane ) = static_cast< T >( value );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// CIE XYZ back to RGB, `cv::COLOR_XYZ2RGB`.
template < typename T >
viame::image_of< T >
xyz_to_rgb( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "xyz_to_rgb" );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    auto const& matrix = detail::xyz_to_rgb_fixed();

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        long long const x = image( i, j, 0 );
        long long const y = image( i, j, 1 );
        long long const z = image( i, j, 2 );

        for( size_t plane = 0; plane < 3; ++plane )
        {
          auto const value = detail::descale(
            x * matrix[ plane * 3 ] + y * matrix[ plane * 3 + 1 ] +
            z * matrix[ plane * 3 + 2 ], detail::xyz_shift );

          out( i, j, plane ) = saturate_pixel< T >( value );
        }
      }
    }

    return out;
  }

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const x = static_cast< float >( image( i, j, 0 ) );
      auto const y = static_cast< float >( image( i, j, 1 ) );
      auto const z = static_cast< float >( image( i, j, 2 ) );

      for( size_t plane = 0; plane < 3; ++plane )
      {
        auto const value =
          x * static_cast< float >( detail::xyz_to_rgb_matrix[ plane * 3 ] ) +
          y *
            static_cast< float >( detail::xyz_to_rgb_matrix[ plane * 3 + 1 ] ) +
          z *
            static_cast< float >( detail::xyz_to_rgb_matrix[ plane * 3 + 2 ] );

        out( i, j, plane ) = static_cast< T >( value );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// RGB to a luma-chroma space: `COLOR_RGB2YCrCb`, or `COLOR_RGB2YUV`.
///
/// One function for both because that is what they are -- the same luma and
/// a different pair of chroma scalings. The plane order differs too, and it
/// is OpenCV's: Y, Cr, Cb for the first and Y, U, V for the second, where
/// Cr and V both carry red and Cb and U both carry blue.
///
/// 8 bit is exact over a random 256 by 256 block, in both directions and
/// both spaces.
template < typename T >
viame::image_of< T >
rgb_to_luma_chroma( viame::image_of< T > const& image, bool yuv )
{
  detail::require_planes( image, 3, "rgb_to_luma_chroma" );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  auto const first = yuv ? detail::yuv_u : detail::ycrcb_cr;
  auto const second = yuv ? detail::yuv_v : detail::ycrcb_cb;

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    constexpr int delta = 128;

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        long long const red = image( i, j, 0 );
        long long const green = image( i, j, 1 );
        long long const blue = image( i, j, 2 );

        auto const luma = detail::descale(
          red * detail::luma_red + green * detail::luma_green +
          blue * detail::luma_blue, detail::yuv_shift );

        // YCrCb carries red first and YUV carries blue first, and the
        // scalings go with the channel rather than with the position.
        auto const red_chroma = detail::descale(
          ( red - luma ) * ( yuv ? detail::yuv_v : detail::ycrcb_cr ),
          detail::yuv_shift ) + delta;
        auto const blue_chroma = detail::descale(
          ( blue - luma ) * ( yuv ? detail::yuv_u : detail::ycrcb_cb ),
          detail::yuv_shift ) + delta;

        out( i, j, 0 ) = saturate_pixel< T >( luma );
        out( i, j, 1 ) = saturate_pixel< T >( yuv ? blue_chroma : red_chroma );
        out( i, j, 2 ) = saturate_pixel< T >( yuv ? red_chroma : blue_chroma );
      }
    }

    return out;
  }

  auto const red_scale = yuv ? detail::yuv_v_f : detail::ycrcb_cr_f;
  auto const blue_scale = yuv ? detail::yuv_u_f : detail::ycrcb_cb_f;
  ( void ) first;
  ( void ) second;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const red = static_cast< float >( image( i, j, 0 ) );
      auto const green = static_cast< float >( image( i, j, 1 ) );
      auto const blue = static_cast< float >( image( i, j, 2 ) );

      auto const luma = red * 0.299f + green * 0.587f + blue * 0.114f;
      auto const red_chroma = ( red - luma ) * red_scale + 0.5f;
      auto const blue_chroma = ( blue - luma ) * blue_scale + 0.5f;

      out( i, j, 0 ) = static_cast< T >( luma );
      out( i, j, 1 ) = static_cast< T >( yuv ? blue_chroma : red_chroma );
      out( i, j, 2 ) = static_cast< T >( yuv ? red_chroma : blue_chroma );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// A luma-chroma space back to RGB: `COLOR_YCrCb2RGB`, or `COLOR_YUV2RGB`.
template < typename T >
viame::image_of< T >
luma_chroma_to_rgb( viame::image_of< T > const& image, bool yuv )
{
  detail::require_planes( image, 3, "luma_chroma_to_rgb" );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  if constexpr( std::is_same< T, uint8_t >::value )
  {
    constexpr int delta = 128;

    for( size_t j = 0; j < image.height(); ++j )
    {
      for( size_t i = 0; i < image.width(); ++i )
      {
        long long const luma = image( i, j, 0 );
        long long const first = static_cast< long long >(
          image( i, j, 1 ) ) - delta;
        long long const second = static_cast< long long >(
          image( i, j, 2 ) ) - delta;

        long long const red_chroma = yuv ? second : first;
        long long const blue_chroma = yuv ? first : second;

        int red, green, blue;

        if( yuv )
        {
          red = static_cast< int >( luma ) + detail::descale(
            red_chroma * detail::yuv_to_red, detail::yuv_shift );
          green = static_cast< int >( luma ) + detail::descale(
            blue_chroma * detail::yuv_to_green_u +
            red_chroma * detail::yuv_to_green_v, detail::yuv_shift );
          blue = static_cast< int >( luma ) + detail::descale(
            blue_chroma * detail::yuv_to_blue, detail::yuv_shift );
        }
        else
        {
          red = static_cast< int >( luma ) + detail::descale(
            red_chroma * detail::ycrcb_to_red, detail::yuv_shift );
          green = static_cast< int >( luma ) + detail::descale(
            blue_chroma * detail::ycrcb_to_green_cb +
            red_chroma * detail::ycrcb_to_green_cr, detail::yuv_shift );
          blue = static_cast< int >( luma ) + detail::descale(
            blue_chroma * detail::ycrcb_to_blue, detail::yuv_shift );
        }

        out( i, j, 0 ) = saturate_pixel< T >( red );
        out( i, j, 1 ) = saturate_pixel< T >( green );
        out( i, j, 2 ) = saturate_pixel< T >( blue );
      }
    }

    return out;
  }

  auto const to_red = yuv ? detail::yuv_to_red_f : detail::ycrcb_to_red_f;
  auto const to_green_red =
    yuv ? detail::yuv_to_green_v_f : detail::ycrcb_to_green_cr_f;
  auto const to_green_blue =
    yuv ? detail::yuv_to_green_u_f : detail::ycrcb_to_green_cb_f;
  auto const to_blue = yuv ? detail::yuv_to_blue_f : detail::ycrcb_to_blue_f;

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      auto const luma = static_cast< float >( image( i, j, 0 ) );
      auto const first = static_cast< float >( image( i, j, 1 ) ) - 0.5f;
      auto const second = static_cast< float >( image( i, j, 2 ) ) - 0.5f;

      auto const red_chroma = yuv ? second : first;
      auto const blue_chroma = yuv ? first : second;

      out( i, j, 0 ) = static_cast< T >( luma + red_chroma * to_red );
      out( i, j, 1 ) = static_cast< T >(
        luma + blue_chroma * to_green_blue + red_chroma * to_green_red );
      out( i, j, 2 ) = static_cast< T >( luma + blue_chroma * to_blue );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// RGB to CIE L*u*v*, which is `cv::COLOR_RGB2Luv`.
///
/// Unlike `rgb_to_xyz` above, this **does** linearise first, as OpenCV's
/// does: L*u*v* is a perceptual space and applying it to gamma-encoded
/// numbers would mean nothing. 8 bit carries OpenCV's scaling -- L over
/// 0..255, u offset by 134 and scaled by 255/354, v offset by 140 and scaled
/// by 255/262 -- and float32 the real ranges, L 0..100 and the chroma pair
/// about -134..220 and -140..122.
///
/// **Not bit exact, and the reason is the one finding 2.60 records for
/// L*a*b*.** OpenCV evaluates the sRGB transfer off an interpolated spline
/// rather than calling `pow`, so a float conversion here is within 0.13 of a
/// u unit and an 8-bit one is a count out on about eighteen percent of
/// pixels, never more than one. Reproducing the spline is what `lab_tables`
/// does for L*a*b*, where the goldens demanded it; nothing records L*u*v*.
template < typename T >
viame::image_of< T >
rgb_to_luv( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "rgb_to_luv" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

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

      auto const x = detail::rgb_to_xyz_matrix[ 0 ] * red +
                     detail::rgb_to_xyz_matrix[ 1 ] * green +
                     detail::rgb_to_xyz_matrix[ 2 ] * blue;
      auto const y = detail::rgb_to_xyz_matrix[ 3 ] * red +
                     detail::rgb_to_xyz_matrix[ 4 ] * green +
                     detail::rgb_to_xyz_matrix[ 5 ] * blue;
      auto const z = detail::rgb_to_xyz_matrix[ 6 ] * red +
                     detail::rgb_to_xyz_matrix[ 7 ] * green +
                     detail::rgb_to_xyz_matrix[ 8 ] * blue;

      // OpenCV's own constants, not the CIE ones: 903.3 and 0.008856 are
      // rounded forms of 24389/27 and 216/24389, and using the exact pair
      // moves the answer where the two disagree.
      auto const lightness = ( y > 0.008856 ) ? 116.0 * std::cbrt( y ) - 16.0
                                              : 903.3 * y;

      auto const denominator = x + 15.0 * y + 3.0 * z;
      auto const u_prime = ( denominator > 0.0 ) ? 4.0 * x / denominator : 0.0;
      auto const v_prime = ( denominator > 0.0 ) ? 9.0 * y / denominator : 0.0;

      auto const u = 13.0 * lightness * ( u_prime - detail::luv_white_u );
      auto const v = 13.0 * lightness * ( v_prime - detail::luv_white_v );

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel_even< T >( lightness * 255.0 / 100.0 );
        out( i, j, 1 ) =
          saturate_pixel_even< T >( ( u + 134.0 ) * 255.0 / 354.0 );
        out( i, j, 2 ) =
          saturate_pixel_even< T >( ( v + 140.0 ) * 255.0 / 262.0 );
      }
      else
      {
        out( i, j, 0 ) = static_cast< T >( lightness );
        out( i, j, 1 ) = static_cast< T >( u );
        out( i, j, 2 ) = static_cast< T >( v );
      }
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// CIE L*u*v* back to RGB, `cv::COLOR_Luv2RGB`, on the same scalings.
template < typename T >
viame::image_of< T >
luv_to_rgb( viame::image_of< T > const& image )
{
  detail::require_planes( image, 3, "luv_to_rgb" );

  constexpr bool integral = std::is_integral< T >::value;
  auto const top = static_cast< double >( pixel_max< T >() );

  viame::image_of< T > out( image.width(), image.height(), 3 );

  for( size_t j = 0; j < image.height(); ++j )
  {
    for( size_t i = 0; i < image.width(); ++i )
    {
      double lightness, u, v;

      if constexpr( integral )
      {
        lightness = static_cast< double >( image( i, j, 0 ) ) * 100.0 / 255.0;
        u = static_cast< double >( image( i, j, 1 ) ) * 354.0 / 255.0 - 134.0;
        v = static_cast< double >( image( i, j, 2 ) ) * 262.0 / 255.0 - 140.0;
      }
      else
      {
        lightness = static_cast< double >( image( i, j, 0 ) );
        u = static_cast< double >( image( i, j, 1 ) );
        v = static_cast< double >( image( i, j, 2 ) );
      }

      auto const y = ( lightness > 8.0 )
        ? std::pow( ( lightness + 16.0 ) / 116.0, 3.0 )
        : lightness / 903.3;

      double x = 0.0, z = 0.0;

      if( lightness > 0.0 )
      {
        auto const u_prime = u / ( 13.0 * lightness ) + detail::luv_white_u;
        auto const v_prime = v / ( 13.0 * lightness ) + detail::luv_white_v;

        if( v_prime != 0.0 )
        {
          x = 2.25 * y * u_prime / v_prime;
          z = y * ( 3.0 - 0.75 * u_prime - 5.0 * v_prime ) / v_prime;
        }
      }

      auto const red = detail::linear_to_srgb(
        detail::xyz_to_rgb_matrix[ 0 ] * x + detail::xyz_to_rgb_matrix[ 1 ] * y +
        detail::xyz_to_rgb_matrix[ 2 ] * z );
      auto const green = detail::linear_to_srgb(
        detail::xyz_to_rgb_matrix[ 3 ] * x + detail::xyz_to_rgb_matrix[ 4 ] * y +
        detail::xyz_to_rgb_matrix[ 5 ] * z );
      auto const blue = detail::linear_to_srgb(
        detail::xyz_to_rgb_matrix[ 6 ] * x + detail::xyz_to_rgb_matrix[ 7 ] * y +
        detail::xyz_to_rgb_matrix[ 8 ] * z );

      if constexpr( integral )
      {
        out( i, j, 0 ) = saturate_pixel_even< T >( red * top );
        out( i, j, 1 ) = saturate_pixel_even< T >( green * top );
        out( i, j, 2 ) = saturate_pixel_even< T >( blue * top );
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

} // namespace image_kernels
} // namespace viame

#endif
