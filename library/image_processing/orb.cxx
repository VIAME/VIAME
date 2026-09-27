/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief ORB, ported from OpenCV's `features/src/orb.cpp`
///
/// The algorithm, its constants and its sampling pattern are OpenCV's.
/// Their copyright and licence for that file:
///
///   Copyright (C) 2009, Willow Garage Inc., all rights reserved.
///   Copyright (C) 2013, OpenCV Foundation, all rights reserved.
///
///   Redistribution and use in source and binary forms, with or without
///   modification, are permitted provided that the following conditions are
///   met: redistributions of source code must retain the above copyright
///   notice, this list of conditions and the following disclaimer;
///   redistributions in binary form must reproduce the above copyright
///   notice, this list of conditions and the following disclaimer in the
///   documentation and/or other materials provided with the distribution;
///   neither the names of the copyright holders nor the names of the
///   contributors may be used to endorse or promote products derived from
///   this software without specific prior written permission.
///
///   This software is provided by the copyright holders and contributors
///   "as is" and any express or implied warranties, including, but not
///   limited to, the implied warranties of merchantability and fitness for a
///   particular purpose are disclaimed. In no event shall the Intel
///   Corporation or contributors be liable for any direct, indirect,
///   incidental, special, exemplary, or consequential damages however caused
///   and on any theory of liability, whether in contract, strict liability,
///   or tort arising in any way out of the use of this software, even if
///   advised of the possibility of such damage.
///
/// What changed in the port, and nothing else did:
///
/// - The pyramid is a vector of separately allocated bordered levels rather
///   than one packed buffer with a rectangle per level. OpenCV packs them so
///   that the blur can read across the border without a second copy; since
///   every level here carries its own reflected border the reads land on the
///   same pixels, and no level was ever near enough another to reach it --
///   the packing aligns its width to sixteen, which leaves no room for a
///   second level on a row.
/// - `cv::parallel_for_` became plain loops.
/// - The culls are **`std::nth_element` and `std::partition` in OpenCV**,
///   and neither specifies its output order. What they select is well
///   defined -- every keypoint whose response reaches the n-th largest,
///   ties included, which is exactly what the `partition` is there to keep
///   -- and that is reproduced. The order is not, so this returns detection
///   order; orb.h says why nothing downstream can tell.
/// - `wta_k` of 3 or 4 and a `patch_size` other than 31 are refused rather
///   than approximated. Both draw their sampling pattern from `cv::RNG`'s
///   multiply-with-carry generator, seeded with a literal, and a descriptor
///   built from a different pattern is not comparable with cv2's at all --
///   there is no "close". Nothing on this branch selects either, and the
///   two-bit descriptors would need a Hamming-2 matcher that `matching.py`
///   does not offer.
/// - OpenCL, CUDA and the mask argument are dropped, as in the other two.

#include "orb.h"

#include <image_kernels/corners.h>
#include <image_kernels/filter.h>
#include <image_kernels/warp.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <functional>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace viame {

namespace orb {

namespace {

using viame::image_kernels::border_mode;
using viame::image_kernels::interpolation;

constexpr float harris_k = 0.04f;
constexpr int harris_block_size = 9;
constexpr int harris_response_block = 7;

// ----------------------------------------------------------------------------
/// `cvRound`, which is round-half-to-even and not round-half-away.
inline int
round_to_even( float value )
{
  return static_cast< int >( std::lrint( value ) );
}

inline int
round_to_even( double value )
{
  return static_cast< int >( std::lrint( value ) );
}

// ----------------------------------------------------------------------------
/// `bit_pattern_31_`: the 256 learned point pairs, as 1024 signed offsets.
///
/// Four numbers per descriptor bit -- the x and y of the first sample and
/// then of the second -- inside a 31 pixel patch, so every one is in
/// [-13, 12]. ORB learned these once from a training set and shipped the
/// table; for the default patch size OpenCV copies it verbatim and the
/// generator it carries for other sizes is never reached. That is the whole
/// reason this port can be exact without reproducing `cv::RNG`.
int const bit_pattern_31[ 1024 ] = {
     8,   -3,    9,    5,    4,    2,    7,  -12,
   -11,    9,   -8,    2,    7,  -12,   12,  -13,
     2,  -13,    2,   12,    1,   -7,    1,    6,
    -2,  -10,   -2,   -4,  -13,  -13,  -11,   -8,
   -13,   -3,  -12,   -9,   10,    4,   11,    9,
   -13,   -8,   -8,   -9,  -11,    7,   -9,   12,
     7,    7,   12,    6,   -4,   -5,   -3,    0,
   -13,    2,  -12,   -3,   -9,    0,   -7,    5,
    12,   -6,   12,   -1,   -3,    6,   -2,   12,
    -6,  -13,   -4,   -8,   11,  -13,   12,   -8,
     4,    7,    5,    1,    5,   -3,   10,   -3,
     3,   -7,    6,   12,   -8,   -7,   -6,   -2,
    -2,   11,   -1,  -10,  -13,   12,   -8,   10,
    -7,    3,   -5,   -3,   -4,    2,   -3,    7,
   -10,  -12,   -6,   11,    5,  -12,    6,   -7,
     5,   -6,    7,   -1,    1,    0,    4,   -5,
     9,   11,   11,  -13,    4,    7,    4,   12,
     2,   -1,    4,    4,   -4,  -12,   -2,    7,
    -8,   -5,   -7,  -10,    4,   11,    9,   12,
     0,   -8,    1,  -13,  -13,   -2,   -8,    2,
    -3,   -2,   -2,    3,   -6,    9,   -4,   -9,
     8,   12,   10,    7,    0,    9,    1,    3,
     7,   -5,   11,  -10,  -13,   -6,  -11,    0,
    10,    7,   12,    1,   -6,   -3,   -6,   12,
    10,   -9,   12,   -4,  -13,    8,   -8,  -12,
   -13,    0,   -8,   -4,    3,    3,    7,    8,
     5,    7,   10,   -7,   -1,    7,    1,  -12,
     3,  -10,    5,    6,    2,   -4,    3,  -10,
   -13,    0,  -13,    5,  -13,   -7,  -12,   12,
   -13,    3,  -11,    8,   -7,   12,   -4,    7,
     6,  -10,   12,    8,   -9,   -1,   -7,   -6,
    -2,   -5,    0,   12,  -12,    5,   -7,    5,
     3,  -10,    8,  -13,   -7,   -7,   -4,    5,
    -3,   -2,   -1,   -7,    2,    9,    5,  -11,
   -11,  -13,   -5,  -13,   -1,    6,    0,   -1,
     5,   -3,    5,    2,   -4,  -13,   -4,   12,
    -9,   -6,   -9,    6,  -12,  -10,   -8,   -4,
    10,    2,   12,   -3,    7,   12,   12,   12,
    -7,  -13,   -6,    5,   -4,    9,   -3,    4,
     7,   -1,   12,    2,   -7,    6,   -5,    1,
   -13,   11,  -12,    5,   -3,    7,   -2,   -6,
     7,   -8,   12,   -7,  -13,   -7,  -11,  -12,
     1,   -3,   12,   12,    2,   -6,    3,    0,
    -4,    3,   -2,  -13,   -1,  -13,    1,    9,
     7,    1,    8,   -6,    1,   -1,    3,   12,
     9,    1,   12,    6,   -1,   -9,   -1,    3,
   -13,  -13,  -10,    5,    7,    7,   10,   12,
    12,   -5,   12,    9,    6,    3,    7,   11,
     5,  -13,    6,   10,    2,  -12,    2,    3,
     3,    8,    4,   -6,    2,    6,   12,  -13,
     9,  -12,   10,    3,   -8,    4,   -7,    9,
   -11,   12,   -4,   -6,    1,   12,    2,   -8,
     6,   -9,    7,   -4,    2,    3,    3,   -2,
     6,    3,   11,    0,    3,   -3,    8,   -8,
     7,    8,    9,    3,  -11,   -5,   -6,   -4,
   -10,   11,   -5,   10,   -5,   -8,   -3,   12,
   -10,    5,   -9,    0,    8,   -1,   12,   -6,
     4,   -6,    6,  -11,  -10,   12,   -8,    7,
     4,   -2,    6,    7,   -2,    0,   -2,   12,
    -5,   -8,   -5,    2,    7,   -6,   10,   12,
    -9,  -13,   -8,   -8,   -5,  -13,   -5,   -2,
     8,   -8,    9,  -13,   -9,  -11,   -9,    0,
     1,   -8,    1,   -2,    7,   -4,    9,    1,
    -2,    1,   -1,   -4,   11,   -6,   12,  -11,
   -12,   -9,   -6,    4,    3,    7,    7,   12,
     5,    5,   10,    8,    0,   -4,    2,    8,
    -9,   12,   -5,  -13,    0,    7,    2,   12,
    -1,    2,    1,    7,    5,   11,    7,   -9,
     3,    5,    6,   -8,  -13,   -4,   -8,    9,
    -5,    9,   -3,   -3,   -4,   -7,   -3,  -12,
     6,    5,    8,    0,   -7,    6,   -6,   12,
   -13,    6,   -5,   -2,    1,  -10,    3,   10,
     4,    1,    8,   -4,   -2,   -2,    2,  -13,
     2,  -12,   12,   12,   -2,  -13,    0,   -6,
     4,    1,    9,    3,   -6,  -10,   -3,   -5,
    -3,  -13,   -1,    1,    7,    5,   12,  -11,
     4,   -2,    5,   -7,  -13,    9,   -9,   -5,
     7,    1,    8,    6,    7,   -8,    7,    6,
    -7,   -4,   -7,    1,   -8,   11,   -7,   -8,
   -13,    6,  -12,   -8,    2,    4,    3,    9,
    10,   -5,   12,    3,   -6,   -5,   -6,    7,
     8,   -3,    9,   -8,    2,  -12,    2,    8,
   -11,   -2,  -10,    3,  -12,  -13,   -7,   -9,
   -11,    0,  -10,   -5,    5,   -3,   11,    8,
    -2,  -13,   -1,   12,   -1,   -8,    0,    9,
   -13,  -11,  -12,   -5,  -10,   -2,  -10,   11,
    -3,    9,   -2,  -13,    2,   -3,    3,    2,
    -9,  -13,   -4,    0,   -4,    6,   -3,  -10,
    -4,   12,   -2,   -7,   -6,  -11,   -4,    9,
     6,   -3,    6,   11,  -13,   11,   -5,    5,
    11,   11,   12,    6,    7,   -5,   12,   -2,
    -1,   12,    0,    7,   -4,   -8,   -3,   -2,
    -7,    1,   -6,    7,  -13,  -12,   -8,  -13,
    -7,   -2,   -6,   -8,   -8,    5,   -6,   -9,
    -5,   -1,   -4,    5,  -13,    7,   -8,   10,
     1,    5,    5,  -13,    1,    0,   10,  -13,
     9,   12,   10,   -1,    5,   -8,   10,   -9,
    -1,   11,    1,  -13,   -9,   -3,   -6,    2,
    -1,  -10,    1,   12,  -13,    1,   -8,  -10,
     8,  -11,   10,   -6,    2,  -13,    3,   -6,
     7,  -13,   12,   -9,  -10,  -10,   -5,   -7,
   -10,   -8,   -8,  -13,    4,   -6,    8,    5,
     3,   12,    8,  -13,   -4,    2,   -3,   -3,
     5,  -13,   10,  -12,    4,  -13,    5,   -1,
    -9,    9,   -4,    3,    0,    3,    3,   -9,
   -12,    1,   -6,    1,    3,    2,    4,   -8,
   -10,  -10,  -10,    9,    8,  -13,   12,   12,
    -8,  -12,   -6,   -5,    2,    2,    3,    7,
    10,    6,   11,   -8,    6,    8,    8,  -12,
    -7,   10,   -6,    5,   -3,   -9,   -3,    9,
    -1,  -13,   -1,    5,   -3,   -7,   -3,    4,
    -8,   -2,   -8,    3,    4,    2,   12,   12,
     2,   -5,    3,   11,    6,   -9,   11,  -13,
     3,   -1,    7,   12,   11,   -1,   12,    4,
    -3,    0,   -3,    6,    4,  -11,    4,   12,
     2,   -4,    2,    1,  -10,   -6,   -8,    1,
   -13,    7,  -11,    1,  -13,   12,  -11,  -13,
     6,    0,   11,  -13,    0,   -1,    1,    4,
   -13,    3,   -9,   -2,   -9,    8,   -6,   -3,
   -13,   -6,   -8,   -2,    5,   -9,    8,   10,
     2,    7,    3,   -9,   -1,   -6,   -1,   -1,
     9,    5,   11,   -2,   11,   -3,   12,   -8,
     3,    0,    3,    5,   -1,    4,    0,   10,
     3,   -6,    4,    5,  -13,    0,  -10,    5,
     5,    8,   12,   11,    8,    9,    9,   -6,
     7,   -4,    8,  -12,  -10,    4,  -10,    9,
     7,    3,   12,    4,    9,   -7,   10,   -2,
     7,    0,   12,   -2,   -1,   -6,    0,  -11,
};


// ----------------------------------------------------------------------------
/// The bordered pyramid level and where its interior starts.
struct level_image
{
  viame::image_of< uint8_t > padded;
  size_t width = 0;
  size_t height = 0;
  int border = 0;

  /// The level itself, sharing the padded image's memory so that a blur
  /// written here is seen by a descriptor read from there.
  viame::image_of< uint8_t >
  interior() const
  {
    auto const* first =
      padded.first_pixel() + border * padded.h_step() + border * padded.w_step();
    return viame::image_of< uint8_t >(
      padded.memory(), first, width, height, 1,
      padded.w_step(), padded.h_step(), padded.d_step() );
  }
};

// ----------------------------------------------------------------------------
/// `copyMakeBorder` with `BORDER_REFLECT_101` around a whole image.
level_image
with_border( viame::image_of< uint8_t > const& image, int border )
{
  level_image out;
  out.width = image.width();
  out.height = image.height();
  out.border = border;
  out.padded = viame::image_of< uint8_t >(
    image.width() + 2 * border, image.height() + 2 * border, 1 );

  auto const width = static_cast< long >( image.width() );
  auto const height = static_cast< long >( image.height() );

  for( size_t y = 0; y < out.padded.height(); ++y )
  {
    auto const from_y = viame::image_kernels::detail::border_index(
      static_cast< long >( y ) - border, height, border_mode::REFLECT_101 );
    for( size_t x = 0; x < out.padded.width(); ++x )
    {
      auto const from_x = viame::image_kernels::detail::border_index(
        static_cast< long >( x ) - border, width, border_mode::REFLECT_101 );
      out.padded( x, y ) = image( static_cast< size_t >( from_x ),
                                  static_cast< size_t >( from_y ) );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// `getScale`: the ratio raised to the level, in double and returned as a
/// float, which is how OpenCV computes and stores it.
inline float
level_scale( int level, int first_level, float scale_factor )
{
  return static_cast< float >(
    std::pow( static_cast< double >( scale_factor ),
              static_cast< double >( level - first_level ) ) );
}

// ----------------------------------------------------------------------------
/// The Harris cornerness of a 7x7 patch, scaled as `HarrisResponses` scales
/// it: by `1 / (4 * blockSize * 255)` to the fourth, which puts the answer in
/// units of a normalised gradient rather than of the pixel.
float
harris_response( level_image const& level, int x0, int y0 )
{
  constexpr int block = harris_response_block;
  constexpr int radius = block / 2;

  auto const scale = 1.0f / ( ( 1 << 2 ) * block * 255.0f );
  auto const scale_sq_sq = scale * scale * scale * scale;

  auto const step = level.padded.h_step();
  auto const column = level.padded.w_step();
  auto const* origin = level.padded.first_pixel() +
    ( y0 - radius + level.border ) * step +
    ( x0 - radius + level.border ) * column;

  int a = 0, b = 0, c = 0;
  for( int i = 0; i < block; ++i )
  {
    for( int j = 0; j < block; ++j )
    {
      auto const* p = origin + i * step + j * column;
      int const ix = ( p[ column ] - p[ -column ] ) * 2 +
        ( p[ -step + column ] - p[ -step - column ] ) +
        ( p[ step + column ] - p[ step - column ] );
      int const iy = ( p[ step ] - p[ -step ] ) * 2 +
        ( p[ step - column ] - p[ -step - column ] ) +
        ( p[ step + column ] - p[ -step + column ] );
      a += ix * ix;
      b += iy * iy;
      c += ix * iy;
    }
  }

  return ( static_cast< float >( a ) * b - static_cast< float >( c ) * c -
           harris_k * ( static_cast< float >( a ) + b ) *
           ( static_cast< float >( a ) + b ) ) * scale_sq_sq;
}

// ----------------------------------------------------------------------------
/// `cv::fastAtan2`: degrees in [0, 360), to about a third of a degree.
///
/// Reproduced rather than replaced with `std::atan2`, as in `surf.cxx`: the
/// descriptor is sampled along the angle this returns, so a more accurate
/// answer here is a different descriptor.
float
fast_atan2( float y, float x )
{
  static float const degrees = static_cast< float >( 180.0 / M_PI );
  static float const p1 = 0.9997878412794807f * degrees;
  static float const p3 = -0.3258083974640975f * degrees;
  static float const p5 = 0.1555786518463281f * degrees;
  static float const p7 = -0.04432655554792128f * degrees;

  float const ax = std::abs( x );
  float const ay = std::abs( y );
  float a, c, c2;

  if( ax >= ay )
  {
    c = ay / ( ax + static_cast< float >( DBL_EPSILON ) );
    c2 = c * c;
    a = ( ( ( p7 * c2 + p5 ) * c2 + p3 ) * c2 + p1 ) * c;
  }
  else
  {
    c = ax / ( ay + static_cast< float >( DBL_EPSILON ) );
    c2 = c * c;
    a = 90.0f - ( ( ( p7 * c2 + p5 ) * c2 + p3 ) * c2 + p1 ) * c;
  }

  if( x < 0 ) { a = 180.0f - a; }
  if( y < 0 ) { a = 360.0f - a; }

  return a;
}

// ----------------------------------------------------------------------------
/// The intensity centroid angle, `ICAngles`.
///
/// The patch is circular rather than square, and \p u_max is how far the
/// circle reaches on row v. The first moments of the intensity about the
/// centre point towards the bright side, and the keypoint is described
/// rotated to that direction -- which is the whole of ORB's rotation
/// invariance.
float
centroid_angle( level_image const& level, int x0, int y0,
                std::vector< int > const& u_max, int half_patch )
{
  auto const step = level.padded.h_step();
  auto const column = level.padded.w_step();
  auto const* centre = level.padded.first_pixel() +
    ( y0 + level.border ) * step + ( x0 + level.border ) * column;

  int m_01 = 0, m_10 = 0;

  for( int u = -half_patch; u <= half_patch; ++u )
  { m_10 += u * centre[ u * column ]; }

  for( int v = 1; v <= half_patch; ++v )
  {
    int v_sum = 0;
    int const d = u_max[ v ];
    for( int u = -d; u <= d; ++u )
    {
      int const above = centre[ u * column + v * step ];
      int const below = centre[ u * column - v * step ];
      v_sum += above - below;
      m_10 += u * ( above + below );
    }
    m_01 += v * v_sum;
  }

  return fast_atan2( static_cast< float >( m_01 ),
                     static_cast< float >( m_10 ) );
}

// ----------------------------------------------------------------------------
/// `computeOrbDescriptors` for `wta_k == 2`: one bit per learned pair, set
/// when the first sample is darker than the second.
///
/// The pattern is rotated by the keypoint's angle and **rounded to the
/// pixel** -- `GET_VALUE`'s bilinear alternative is compiled out in OpenCV
/// and has been for as long as the file has existed.
void
describe( level_image const& level, int x0, int y0, float angle,
          uint8_t* out )
{
  // `angle *= (float)(CV_PI/180.f)` then `(float)cos(angle)`. The cast in
  // OpenCV is off a **double** cosine: unqualified `cos` inside `namespace
  // cv` with no `using namespace std` finds the C one. In double the
  // difference from the float overload is a last-bit one, and a last bit of
  // a cosine moves a rotated pattern point across a rounding boundary often
  // enough to matter at this patch size.
  auto const radians = angle * static_cast< float >( M_PI / 180.0 );
  auto const a =
    static_cast< float >( std::cos( static_cast< double >( radians ) ) );
  auto const b =
    static_cast< float >( std::sin( static_cast< double >( radians ) ) );

  auto const step = level.padded.h_step();
  auto const column = level.padded.w_step();
  auto const* centre = level.padded.first_pixel() +
    ( y0 + level.border ) * step + ( x0 + level.border ) * column;

  auto const* pattern = bit_pattern_31;

  auto const sample = [&]( int index ) -> int
  {
    auto const px = static_cast< float >( pattern[ 2 * index ] );
    auto const py = static_cast< float >( pattern[ 2 * index + 1 ] );
    auto const ix = round_to_even( px * a - py * b );
    auto const iy = round_to_even( px * b + py * a );
    return centre[ iy * step + ix * column ];
  };

  for( int i = 0; i < 32; ++i )
  {
    int value = 0;
    for( int bit = 0; bit < 8; ++bit )
    {
      auto const first = sample( i * 16 + 2 * bit );
      auto const second = sample( i * 16 + 2 * bit + 1 );
      value |= ( first < second ) << bit;
    }
    out[ i ] = static_cast< uint8_t >( value );
  }
}

// ----------------------------------------------------------------------------
/// `KeyPointsFilter::retainBest`, for the set rather than the order.
///
/// OpenCV partitions with `std::nth_element` and then keeps everything that
/// ties with the n-th, which is what the `std::partition` behind it is for.
/// So the survivors are exactly the keypoints whose response reaches the
/// n-th largest -- a well defined set, even though the order the two
/// algorithms leave behind is not. This keeps that set in its original
/// order.
void
retain_best( std::vector< keypoint >& keypoints, int wanted )
{
  if( wanted < 0 || keypoints.size() <= static_cast< size_t >( wanted ) )
  { return; }

  if( wanted == 0 ) { keypoints.clear(); return; }

  std::vector< float > responses;
  responses.reserve( keypoints.size() );
  for( auto const& one : keypoints ) { responses.push_back( one.response ); }

  std::nth_element( responses.begin(), responses.begin() + wanted - 1,
                    responses.end(), std::greater< float >() );
  auto const cutoff = responses[ wanted - 1 ];

  keypoints.erase(
    std::remove_if( keypoints.begin(), keypoints.end(),
                    [ cutoff ]( keypoint const& one )
                    { return one.response < cutoff; } ),
    keypoints.end() );
}

} // namespace

// ----------------------------------------------------------------------------
int
descriptor_size( settings const& config )
{
  return config.wta_k == 2 ? 32 : 0;
}

// ----------------------------------------------------------------------------
void
detect_and_compute( viame::image_of< uint8_t > const& image,
                    settings const& config,
                    std::vector< keypoint >& keypoints,
                    std::vector< uint8_t >* descriptors,
                    bool use_provided_keypoints )
{
  if( config.wta_k != 2 )
  {
    throw std::invalid_argument(
      "orb: only wta_k of 2 is implemented; 3 and 4 draw their sampling "
      "pattern from cv::RNG" );
  }

  if( config.patch_size != 31 )
  {
    throw std::invalid_argument(
      "orb: only a patch_size of 31 is implemented; any other size draws "
      "its sampling pattern from cv::RNG" );
  }

  if( config.patch_size < 2 || config.n_levels < 1 || config.first_level < 0 )
  { throw std::invalid_argument( "orb: the pyramid settings are not usable" ); }

  if( image.depth() != 1 )
  { throw std::invalid_argument( "orb wants a single plane" ); }

  if( !use_provided_keypoints ) { keypoints.clear(); }

  if( image.width() == 0 || image.height() == 0 )
  {
    keypoints.clear();
    if( descriptors ) { descriptors->clear(); }
    return;
  }

  auto const half_patch = config.patch_size / 2;
  // sqrt(2) because the patch is sampled rotated, so its corner sweeps out
  // to the circumscribing circle.
  auto const descriptor_patch =
    static_cast< int >( std::ceil( half_patch * std::sqrt( 2.0 ) ) );
  auto const border = std::max(
    config.edge_threshold,
    std::max( descriptor_patch, harris_block_size / 2 ) ) + 1;

  int n_levels = config.n_levels;
  if( use_provided_keypoints )
  {
    // Provided keypoints may name levels the settings do not reach.
    n_levels = 0;
    for( auto const& one : keypoints )
    {
      if( one.octave < 0 )
      { throw std::invalid_argument( "orb: a keypoint has a negative octave" ); }
      n_levels = std::max( n_levels, one.octave );
    }
    ++n_levels;
  }

  // ---- the pyramid -------------------------------------------------------
  std::vector< level_image > levels( static_cast< size_t >( n_levels ) );
  std::vector< float > layer_scale( static_cast< size_t >( n_levels ) );

  {
    viame::image_of< uint8_t > previous = image;
    for( int level = 0; level < n_levels; ++level )
    {
      auto const scale =
        level_scale( level, config.first_level, config.scale_factor );
      layer_scale[ level ] = scale;
      auto const inverse = 1.0f / scale;
      auto const width = round_to_even(
        static_cast< float >( image.width() ) * inverse );
      auto const height = round_to_even(
        static_cast< float >( image.height() ) * inverse );

      if( width <= 0 || height <= 0 )
      {
        throw std::invalid_argument(
          "orb: the pyramid runs out of image before its last level" );
      }

      viame::image_of< uint8_t > current;
      if( level != config.first_level )
      {
        // Levels cascade from the one below rather than from the source, so
        // the resampling compounds -- that is OpenCV's, and reading the
        // source each time gives a visibly different pyramid. Levels below
        // `first_level` are enlargements, and they do read the source,
        // because `prevImg` is not advanced until the loop passes it.
        current = viame::image_kernels::resize(
          previous, static_cast< size_t >( width ),
          static_cast< size_t >( height ), interpolation::BILINEAR_EXACT );
      }
      else
      {
        current = viame::image_of< uint8_t >( image.width(), image.height(), 1 );
        current.copy_from( image );
      }

      levels[ level ] = with_border( current, border );
      if( level > config.first_level ) { previous = current; }
    }
  }

  // ---- the circular patch's row extents ----------------------------------
  // `umax[v]` is how far the patch reaches on row v. The first loop is the
  // circle; the second makes it symmetric about the diagonal, since a
  // quarter-circle sampled by rows is not the same set as one sampled by
  // columns and the moments have to see a shape, not a staircase.
  std::vector< int > u_max( static_cast< size_t >( half_patch ) + 2, 0 );
  {
    auto const diagonal =
      half_patch * std::sqrt( 2.0f ) / 2.0f;
    auto const v_max = static_cast< int >( std::floor( diagonal + 1 ) );
    auto const v_min = static_cast< int >( std::ceil( diagonal ) );

    for( int v = 0; v <= v_max; ++v )
    {
      u_max[ v ] = round_to_even( std::sqrt(
        static_cast< double >( half_patch ) * half_patch - v * v ) );
    }

    for( int v = half_patch, v0 = 0; v >= v_min; --v )
    {
      while( u_max[ v0 ] == u_max[ v0 + 1 ] ) { ++v0; }
      u_max[ v ] = v0;
      ++v0;
    }
  }

  if( !use_provided_keypoints )
  {
    // ---- the budget, shared out as a geometric series -------------------
    std::vector< int > per_level( static_cast< size_t >( n_levels ), 0 );
    {
      auto const factor =
        static_cast< float >( 1.0 / static_cast< double >(
                                     config.scale_factor ) );
      auto wanted = config.n_features * ( 1 - factor ) /
        ( 1 - static_cast< float >(
            std::pow( static_cast< double >( factor ),
                      static_cast< double >( n_levels ) ) ) );
      int total = 0;
      for( int level = 0; level < n_levels - 1; ++level )
      {
        per_level[ level ] = round_to_even( wanted );
        total += per_level[ level ];
        wanted *= factor;
      }
      per_level[ n_levels - 1 ] = std::max( config.n_features - total, 0 );
    }

    // ---- detect, per level ----------------------------------------------
    std::vector< size_t > counts( static_cast< size_t >( n_levels ), 0 );

    for( int level = 0; level < n_levels; ++level )
    {
      auto const& here = levels[ level ];
      auto const found = viame::image_kernels::fast_corners(
        here.interior(), config.fast_threshold, true );

      std::vector< keypoint > mine;
      mine.reserve( found.size() );

      auto const edge = config.edge_threshold;
      auto const wide = static_cast< long >( here.width ) > 2 * edge;
      auto const tall = static_cast< long >( here.height ) > 2 * edge;

      if( wide && tall )
      {
        for( auto const& corner : found )
        {
          if( corner.x < edge || corner.x >= here.width - edge ||
              corner.y < edge || corner.y >= here.height - edge )
          { continue; }

          keypoint one;
          one.x = corner.x;
          one.y = corner.y;
          one.response = corner.response;
          one.octave = level;
          one.size = config.patch_size * layer_scale[ level ];
          mine.push_back( one );
        }
      }

      // Twice the budget when Harris is about to re-rank them, because FAST
      // does not order corners well enough to cull on directly.
      retain_best( mine, config.harris_score ? 2 * per_level[ level ]
                                             : per_level[ level ] );

      counts[ level ] = mine.size();
      keypoints.insert( keypoints.end(), mine.begin(), mine.end() );
    }

    if( keypoints.empty() )
    {
      if( descriptors ) { descriptors->clear(); }
      return;
    }

    // ---- re-rank on Harris cornerness and cull again ---------------------
    if( config.harris_score )
    {
      for( auto& one : keypoints )
      {
        one.response = harris_response(
          levels[ one.octave ], round_to_even( one.x ),
          round_to_even( one.y ) );
      }

      std::vector< keypoint > kept;
      kept.reserve( keypoints.size() );
      size_t offset = 0;
      for( int level = 0; level < n_levels; ++level )
      {
        std::vector< keypoint > mine(
          keypoints.begin() + offset,
          keypoints.begin() + offset + counts[ level ] );
        offset += counts[ level ];

        retain_best( mine, per_level[ level ] );
        kept.insert( kept.end(), mine.begin(), mine.end() );
      }
      keypoints.swap( kept );
    }

    // ---- orientation, then back to the source's coordinates --------------
    for( auto& one : keypoints )
    {
      one.angle = centroid_angle(
        levels[ one.octave ], round_to_even( one.x ), round_to_even( one.y ),
        u_max, half_patch );
    }

    for( auto& one : keypoints )
    {
      auto const scale = layer_scale[ one.octave ];
      one.x *= scale;
      one.y *= scale;
    }
  }
  else
  {
    // Provided keypoints are filtered against the **source** image's
    // bounds, not their own level's, which is what `detectAndCompute` does
    // before it reorders them by level.
    auto const edge = config.edge_threshold;
    if( edge > 0 )
    {
      if( static_cast< long >( image.height() ) <= 2 * edge ||
          static_cast< long >( image.width() ) <= 2 * edge )
      { keypoints.clear(); }
      else
      {
        keypoints.erase(
          std::remove_if(
            keypoints.begin(), keypoints.end(),
            [ & ]( keypoint const& one )
            {
              return one.x < edge || one.x >= image.width() - edge ||
                     one.y < edge || one.y >= image.height() - edge;
            } ),
          keypoints.end() );
      }
    }

    std::stable_sort( keypoints.begin(), keypoints.end(),
                      []( keypoint const& a, keypoint const& b )
                      { return a.octave < b.octave; } );
  }

  if( !descriptors ) { return; }

  if( keypoints.empty() ) { descriptors->clear(); return; }

  // ---- describe ----------------------------------------------------------
  // The detector ran on the unblurred pyramid and the descriptor runs on a
  // blurred one; OpenCV blurs in place between the two, which is why this
  // cannot be done any earlier. rBRIEF compares single pixels, so without
  // the blur the descriptor would be a sample of the noise.
  for( auto& here : levels )
  {
    auto const interior = here.interior();
    // **Not** `gaussian_blur`. ORB blurs a region of its packed pyramid
    // buffer with `BORDER_REFLECT_101` and no `BORDER_ISOLATED`, which is
    // exactly the guard that sends `cv::GaussianBlur` past its bit-exact
    // fixed-point path and into `sepFilter2D` with the float kernel. The
    // two differ by a count on about a fifth of the pixels, and rBRIEF
    // compares single pixels, so a count is a bit of the descriptor.
    auto const blurred = viame::image_kernels::gaussian_blur_float_taps(
      interior, 7, 2.0, border_mode::REFLECT_101 );
    for( size_t y = 0; y < here.height; ++y )
    {
      for( size_t x = 0; x < here.width; ++x )
      {
        here.padded( x + here.border, y + here.border ) = blurred( x, y );
      }
    }
  }

  auto const size = descriptor_size( config );
  descriptors->assign( keypoints.size() * static_cast< size_t >( size ), 0 );

  for( size_t i = 0; i < keypoints.size(); ++i )
  {
    auto const& one = keypoints[ i ];
    if( one.octave < 0 || one.octave >= n_levels )
    { throw std::invalid_argument( "orb: a keypoint names a level that is "
                                   "not in the pyramid" ); }

    // Back from the source's coordinates to the level's. Not a division:
    // OpenCV multiplies by the reciprocal in float and rounds, and the two
    // differ by a count often enough to move a sampling point.
    auto const inverse = 1.0f / layer_scale[ one.octave ];
    describe( levels[ one.octave ], round_to_even( one.x * inverse ),
              round_to_even( one.y * inverse ), one.angle,
              descriptors->data() + i * static_cast< size_t >( size ) );
  }
}

} // namespace orb

} // namespace viame
