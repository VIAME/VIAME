/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief SIFT, ported from OpenCV's `features/src/sift.dispatch.cpp` and
///        `sift.simd.hpp`
///
/// The algorithm, its constants and its layout are OpenCV's, which took them
/// in turn from Rob Hess's implementation. OpenCV's copyright and licence for
/// those files:
///
///   Copyright (c) 2006-2010, Rob Hess <hess@eecs.oregonstate.edu>
///   Copyright (C) 2009, Willow Garage Inc., all rights reserved.
///   Copyright (C) 2020, Intel Corporation, all rights reserved.
///
///   Redistribution and use in source and binary forms, with or without
///   modification, are permitted provided that the following conditions are
///   met: redistributions of source code must retain the above copyright and
///   patent notices, this list of conditions and the following disclaimer;
///   redistributions in binary form must reproduce the above copyright notice,
///   this list of conditions and the following disclaimer in the documentation
///   and/or other materials provided with the distribution; neither the name
///   of Oregon State University nor the names of its contributors may be used
///   to endorse or promote products derived from this software without
///   specific prior written permission.
///
///   This software is provided by the copyright holders and contributors "as
///   is" and any express or implied warranties, including, but not limited to,
///   the implied warranties of merchantability and fitness for a particular
///   purpose are disclaimed. In no event shall the copyright holder or
///   contributors be liable for any direct, indirect, incidental, special,
///   exemplary, or consequential damages however caused and on any theory of
///   liability, whether in contract, strict liability, or tort arising in any
///   way out of the use of this software, even if advised of the possibility
///   of such damage.
///
///   Patent US6711293 expired in March 2020.
///
/// What changed in the port, and nothing else did:
///
/// - `cv::Mat` became `viame::image_of< float >`, and the pyramid a vector of
///   those. The extrema search does read across rows, but by index rather than
///   by pointer, which reads the same and bounds the same.
/// - `cv::parallel_for_` became plain loops, and with it the `TLSData`
///   accumulator that gathered keypoints per thread. **That changes the order
///   keypoints come out in**: OpenCV's order depends on how its thread pool
///   split the rows, and `removeDuplicatedSorted` does not re-sort, it only
///   masks duplicates and compacts in place. So a comparison against cv2 has
///   to match keypoints up rather than zip them, which is what
///   `tests/golden/sift` does.
/// - `cv::solve( ..., DECOMP_LU )` on the 3x3 Hessian is **not** an LU solve:
///   `Matx::solve` has a fast path for three by three with one right hand
///   side, and it is Cramer's rule in float. Reproduced as such below --
///   getting this wrong moves the sub-pixel offset, which decides both whether
///   a keypoint survives and where it lands.
/// - `cv::hal::exp32f` and `magnitude32f` became `std::exp` and
///   `std::sqrt( x * x + y * y )`. OpenCV's vectorised `exp` is a polynomial
///   over a table and differs from `std::exp` by a few 1e-07, which is the
///   only knowing approximation in this file; `fastAtan2` **is** reproduced,
///   because it is off by up to a third of a degree and the descriptor is
///   sampled along the angle it returns.
/// - OpenCL, CUDA, the mask argument, the `CV_8U` descriptor type and the
///   `enable_precise_upscale` branch are dropped. No shipped config selects
///   any of them; `cv2.SIFT_create` with five arguments leaves the upscale in
///   its deprecated mode, which is the `resize` path taken here.

#include "sift.h"

#include <image_kernels/color.h>
#include <image_kernels/filter.h>
#include <image_kernels/warp.h>

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>

namespace viame {

namespace sift {

namespace {

using plane = viame::image_of< float >;

// OpenCV's constants, from sift.simd.hpp, with its names in the comments.
int const descr_width = 4;             // SIFT_DESCR_WIDTH
int const descr_hist_bins = 8;         // SIFT_DESCR_HIST_BINS
float const init_sigma = 0.5f;         // SIFT_INIT_SIGMA
int const img_border = 5;              // SIFT_IMG_BORDER
int const max_interp_steps = 5;        // SIFT_MAX_INTERP_STEPS
int const ori_hist_bins = 36;          // SIFT_ORI_HIST_BINS
float const ori_sig_fctr = 1.5f;       // SIFT_ORI_SIG_FCTR
float const ori_radius = 4.5f;         // SIFT_ORI_RADIUS
float const ori_peak_ratio = 0.8f;     // SIFT_ORI_PEAK_RATIO
float const descr_scl_fctr = 3.0f;     // SIFT_DESCR_SCL_FCTR
float const descr_mag_thr = 0.2f;      // SIFT_DESCR_MAG_THR
float const int_descr_fctr = 512.0f;   // SIFT_INT_DESCR_FCTR

/// `cvRound`: half to even, which is what the FPU does.
inline int
round_half_even( float value )
{
  return static_cast< int >( std::nearbyint( value ) );
}

inline int
round_half_even( double value )
{
  return static_cast< int >( std::nearbyint( value ) );
}

/// `cvFloor`.
inline int
floor_of( float value )
{
  return static_cast< int >( std::floor( value ) );
}

/// `cv::fastAtan2`: degrees in [0, 360), to about a third of a degree.
///
/// The same polynomial as `surf.cxx`'s, and reproduced for the same reason:
/// the descriptor is sampled along the angle this returns, so a more accurate
/// answer here is a different answer.
inline float
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

/// The kernel width `cv::GaussianBlur` picks for a float image given `Size()`.
///
/// Four sigmas either side rather than three, which is what a non-byte depth
/// gets, forced odd.
inline size_t
blur_size( double sigma )
{
  auto const width = round_half_even( sigma * 4.0 * 2.0 + 1.0 ) | 1;

  return static_cast< size_t >( std::max( width, 1 ) );
}

plane blurred ( plane const &source, double sigma,
                viame::image_kernels::gaussian_workspace *workspace = nullptr )
{
  return viame::image_kernels::gaussian_blur (
      source, blur_size ( sigma ), sigma, viame::image_kernels::border_mode::REFLECT_101,
      workspace );
}

/// `Matx33f::solve( b, DECOMP_LU )` for one right hand side, which OpenCV
/// answers with Cramer's rule in float rather than with a factorisation.
///
/// Returns false on a singular matrix, as OpenCV does, leaving \p x alone.
bool
solve_3x3( float const a[ 9 ], float const b[ 3 ], float x[ 3 ] )
{
  auto const determinant =
    a[ 0 ] * ( a[ 4 ] * a[ 8 ] - a[ 5 ] * a[ 7 ] ) -
    a[ 1 ] * ( a[ 3 ] * a[ 8 ] - a[ 5 ] * a[ 6 ] ) +
    a[ 2 ] * ( a[ 3 ] * a[ 7 ] - a[ 4 ] * a[ 6 ] );

  if( determinant == 0.0f )
  {
    return false;
  }

  auto const inverse = 1.0f / determinant;

  x[ 0 ] = ( b[ 0 ] * ( a[ 4 ] * a[ 8 ] - a[ 5 ] * a[ 7 ] ) -
             a[ 1 ] * ( b[ 1 ] * a[ 8 ] - a[ 5 ] * b[ 2 ] ) +
             a[ 2 ] * ( b[ 1 ] * a[ 7 ] - a[ 4 ] * b[ 2 ] ) ) * inverse;
  x[ 1 ] = ( a[ 0 ] * ( b[ 1 ] * a[ 8 ] - a[ 5 ] * b[ 2 ] ) -
             b[ 0 ] * ( a[ 3 ] * a[ 8 ] - a[ 5 ] * a[ 6 ] ) +
             a[ 2 ] * ( a[ 3 ] * b[ 2 ] - b[ 1 ] * a[ 6 ] ) ) * inverse;
  x[ 2 ] = ( a[ 0 ] * ( a[ 4 ] * b[ 2 ] - b[ 1 ] * a[ 7 ] ) -
             a[ 1 ] * ( a[ 3 ] * b[ 2 ] - b[ 1 ] * a[ 6 ] ) +
             b[ 0 ] * ( a[ 3 ] * a[ 7 ] - a[ 4 ] * a[ 6 ] ) ) * inverse;

  return true;
}

/// The octave, layer and sub-layer offset a packed `octave` field holds.
void
unpack_octave( int packed, int& octave, int& layer, float& scale )
{
  octave = packed & 255;
  layer = ( packed >> 8 ) & 255;
  octave = octave < 128 ? octave : ( -128 | octave );
  scale = octave >= 0 ? 1.0f / ( 1 << octave )
                      : static_cast< float >( 1 << -octave );
}

// --------------------------------------------------------------------------
/// The base of the pyramid: grey, float, doubled, and blurred up to \p sigma.
plane
initial_image( viame::image_of< uint8_t > const& image, bool double_size,
               float sigma )
{
  auto const grey = image.depth() >= 3
                    ? viame::image_kernels::rgb_to_gray( image )
                    : image;

  plane real( grey.width(), grey.height(), 1 );

  for( size_t j = 0; j < grey.height(); ++j )
  {
    for( size_t i = 0; i < grey.width(); ++i )
    {
      real( i, j, 0 ) = static_cast< float >( grey( i, j, 0 ) );
    }
  }

  if( !double_size )
  {
    auto const difference = std::sqrt(
      std::max( sigma * sigma - init_sigma * init_sigma, 0.01f ) );

    return blurred( real, difference );
  }

  auto const difference = std::sqrt(
    std::max( sigma * sigma - init_sigma * init_sigma * 4.0f, 0.01f ) );

  auto const doubled = viame::image_kernels::resize(
    real, real.width() * 2, real.height() * 2,
    viame::image_kernels::interpolation::BILINEAR );

  return blurred( doubled, difference );
}

/// One Gaussian per octave and layer, `n_octave_layers + 3` to an octave.
std::vector< plane >
gaussian_pyramid( plane const& base, int octaves, int layers, double sigma )
{
  auto const per_octave = layers + 3;

  // sigma[i] is what takes layer i-1 up to layer i, since a Gaussian of
  // sigma_prev followed by one of sigma[i] is a Gaussian of sigma_total.
  std::vector< double > step( static_cast< size_t >( per_octave ) );
  step[ 0 ] = sigma;

  auto const k = std::pow( 2.0, 1.0 / layers );

  for( int i = 1; i < per_octave; ++i )
  {
    auto const previous = std::pow( k, i - 1 ) * sigma;
    auto const total = previous * k;

    step[ static_cast< size_t >( i ) ] =
      std::sqrt( total * total - previous * previous );
  }

  viame::image_kernels::gaussian_workspace workspace;
  std::vector< plane > pyramid(
    static_cast< size_t >( octaves * per_octave ) );

  for( int o = 0; o < octaves; ++o )
  {
    for( int i = 0; i < per_octave; ++i )
    {
      auto& destination = pyramid[ static_cast< size_t >( o * per_octave + i ) ];

      if( o == 0 && i == 0 )
      {
        destination = base;
      }
      else if( i == 0 )
      {
        // A new octave starts on the layer `layers` image of the last one,
        // halved -- and halved by **nearest**, not by area.
        auto const& source =
          pyramid[ static_cast< size_t >( ( o - 1 ) * per_octave + layers ) ];

        destination = viame::image_kernels::resize(
          source, source.width() / 2, source.height() / 2,
          viame::image_kernels::interpolation::NEAREST );
      }
      else
      {
        destination = blurred ( pyramid[static_cast<size_t> ( o * per_octave + i - 1 )],
                                step[static_cast<size_t> ( i )], &workspace );
      }
    }
  }

  return pyramid;
}

std::vector< plane >
difference_pyramid( std::vector< plane > const& gaussian, int layers )
{
  auto const per_octave = layers + 2;
  auto const octaves =
    static_cast< int >( gaussian.size() ) / ( layers + 3 );

  std::vector< plane > pyramid(
    static_cast< size_t >( octaves * per_octave ) );

  for( int o = 0; o < octaves; ++o )
  {
    for( int i = 0; i < per_octave; ++i )
    {
      auto const& lower =
        gaussian[ static_cast< size_t >( o * ( layers + 3 ) + i ) ];
      auto const& upper =
        gaussian[ static_cast< size_t >( o * ( layers + 3 ) + i + 1 ) ];

      plane out( lower.width(), lower.height(), 1 );

      for( size_t j = 0; j < lower.height(); ++j )
      {
        for( size_t x = 0; x < lower.width(); ++x )
        {
          out( x, j, 0 ) = upper( x, j, 0 ) - lower( x, j, 0 );
        }
      }

      pyramid[ static_cast< size_t >( o * per_octave + i ) ] = out;
    }
  }

  return pyramid;
}

// --------------------------------------------------------------------------
/// The orientation histogram around \p x, \p y, and its largest bin.
float
orientation_histogram( plane const& image, int x, int y, int radius,
                       float sigma, std::vector< float >& hist, int bins )
{
  auto const rows = static_cast< int >( image.height() );
  auto const cols = static_cast< int >( image.width() );
  auto const scale = -1.0f / ( 2.0f * sigma * sigma );

  std::vector< float > gradient_x;
  std::vector< float > gradient_y;
  std::vector< float > weight;

  for( int i = -radius; i <= radius; ++i )
  {
    auto const row = y + i;

    if( row <= 0 || row >= rows - 1 ) { continue; }

    for( int j = -radius; j <= radius; ++j )
    {
      auto const col = x + j;

      if( col <= 0 || col >= cols - 1 ) { continue; }

      gradient_x.push_back(
        image( static_cast< size_t >( col + 1 ),
               static_cast< size_t >( row ), 0 ) -
        image( static_cast< size_t >( col - 1 ),
               static_cast< size_t >( row ), 0 ) );
      gradient_y.push_back(
        image( static_cast< size_t >( col ),
               static_cast< size_t >( row - 1 ), 0 ) -
        image( static_cast< size_t >( col ),
               static_cast< size_t >( row + 1 ), 0 ) );
      weight.push_back(
        static_cast< float >( i * i + j * j ) * scale );
    }
  }

  auto const count = gradient_x.size();

  // Two beyond each end, so that the smoothing below can read across the
  // wrap without a branch -- which is what OpenCV's `temphist += 2` buys.
  std::vector< float > raw( static_cast< size_t >( bins ) + 4, 0.0f );
  auto const offset = size_t{ 2 };

  for( size_t k = 0; k < count; ++k )
  {
    auto const magnitude = std::sqrt( gradient_x[ k ] * gradient_x[ k ] +
                                      gradient_y[ k ] * gradient_y[ k ] );
    auto const angle = fast_atan2( gradient_y[ k ], gradient_x[ k ] );

    auto bin = round_half_even( ( bins / 360.0f ) * angle );

    if( bin >= bins ) { bin -= bins; }
    if( bin < 0 ) { bin += bins; }

    raw[ offset + static_cast< size_t >( bin ) ] +=
      std::exp( weight[ k ] ) * magnitude;
  }

  raw[ offset - 1 ] = raw[ offset + static_cast< size_t >( bins ) - 1 ];
  raw[ offset - 2 ] = raw[ offset + static_cast< size_t >( bins ) - 2 ];
  raw[ offset + static_cast< size_t >( bins ) ] = raw[ offset ];
  raw[ offset + static_cast< size_t >( bins ) + 1 ] = raw[ offset + 1 ];

  hist.assign( static_cast< size_t >( bins ), 0.0f );

  auto highest = 0.0f;

  for( int i = 0; i < bins; ++i )
  {
    auto const at = offset + static_cast< size_t >( i );

    hist[ static_cast< size_t >( i ) ] =
      ( raw[ at - 2 ] + raw[ at + 2 ] ) * ( 1.0f / 16.0f ) +
      ( raw[ at - 1 ] + raw[ at + 1 ] ) * ( 4.0f / 16.0f ) +
      raw[ at ] * ( 6.0f / 16.0f );

    highest = i == 0 ? hist[ 0 ]
                     : std::max( highest, hist[ static_cast< size_t >( i ) ] );
  }

  return highest;
}

// --------------------------------------------------------------------------
/// Refine an extremum to sub-pixel, and reject it if it is weak or on an edge.
bool
adjust_extremum( std::vector< plane > const& dog, keypoint& kpt, int octave,
                 int& layer, int& row, int& col, int layers,
                 float contrast_threshold, float edge_threshold, float sigma )
{
  auto const per_octave = layers + 2;

  // The DoG holds differences of values on 0..255, and every derivative below
  // is taken on the 0..1 scale OpenCV works the thresholds against.
  auto const image_scale = 1.0f / 255.0f;
  auto const first_scale = image_scale * 0.5f;
  auto const second_scale = image_scale;
  auto const cross_scale = image_scale * 0.25f;

  auto offset_layer = 0.0f;
  auto offset_row = 0.0f;
  auto offset_col = 0.0f;
  auto contrast = 0.0f;

  auto const at = []( plane const& image, int y, int x ) -> float
  {
    return image( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 );
  };

  int step = 0;

  for( ; step < max_interp_steps; ++step )
  {
    auto const index = static_cast< size_t >( octave * per_octave + layer );
    auto const& image = dog[ index ];
    auto const& below = dog[ index - 1 ];
    auto const& above = dog[ index + 1 ];

    float const gradient[ 3 ] = {
      ( at( image, row, col + 1 ) - at( image, row, col - 1 ) ) * first_scale,
      ( at( image, row + 1, col ) - at( image, row - 1, col ) ) * first_scale,
      ( at( above, row, col ) - at( below, row, col ) ) * first_scale };

    auto const twice = at( image, row, col ) * 2.0f;
    auto const dxx =
      ( at( image, row, col + 1 ) + at( image, row, col - 1 ) - twice ) *
      second_scale;
    auto const dyy =
      ( at( image, row + 1, col ) + at( image, row - 1, col ) - twice ) *
      second_scale;
    auto const dss =
      ( at( above, row, col ) + at( below, row, col ) - twice ) * second_scale;
    auto const dxy =
      ( at( image, row + 1, col + 1 ) - at( image, row + 1, col - 1 ) -
        at( image, row - 1, col + 1 ) + at( image, row - 1, col - 1 ) ) *
      cross_scale;
    auto const dxs =
      ( at( above, row, col + 1 ) - at( above, row, col - 1 ) -
        at( below, row, col + 1 ) + at( below, row, col - 1 ) ) * cross_scale;
    auto const dys =
      ( at( above, row + 1, col ) - at( above, row - 1, col ) -
        at( below, row + 1, col ) + at( below, row - 1, col ) ) * cross_scale;

    float const hessian[ 9 ] = { dxx, dxy, dxs,
                                 dxy, dyy, dys,
                                 dxs, dys, dss };
    float solution[ 3 ] = { 0.0f, 0.0f, 0.0f };

    solve_3x3( hessian, gradient, solution );

    offset_layer = -solution[ 2 ];
    offset_row = -solution[ 1 ];
    offset_col = -solution[ 0 ];

    if( std::abs( offset_layer ) < 0.5f && std::abs( offset_row ) < 0.5f &&
        std::abs( offset_col ) < 0.5f )
    {
      break;
    }

    auto const runaway = static_cast< float >( INT_MAX / 3 );

    if( std::abs( offset_layer ) > runaway ||
        std::abs( offset_row ) > runaway ||
        std::abs( offset_col ) > runaway )
    {
      return false;
    }

    col += round_half_even( offset_col );
    row += round_half_even( offset_row );
    layer += round_half_even( offset_layer );

    auto const width = static_cast< int >( image.width() );
    auto const height = static_cast< int >( image.height() );

    if( layer < 1 || layer > layers ||
        col < img_border || col >= width - img_border ||
        row < img_border || row >= height - img_border )
    {
      return false;
    }
  }

  // Five steps without settling is a rejection, not a best effort.
  if( step >= max_interp_steps )
  {
    return false;
  }

  {
    auto const index = static_cast< size_t >( octave * per_octave + layer );
    auto const& image = dog[ index ];
    auto const& below = dog[ index - 1 ];
    auto const& above = dog[ index + 1 ];

    float const gradient[ 3 ] = {
      ( at( image, row, col + 1 ) - at( image, row, col - 1 ) ) * first_scale,
      ( at( image, row + 1, col ) - at( image, row - 1, col ) ) * first_scale,
      ( at( above, row, col ) - at( below, row, col ) ) * first_scale };

    auto const along = gradient[ 0 ] * offset_col +
                       gradient[ 1 ] * offset_row +
                       gradient[ 2 ] * offset_layer;

    contrast = at( image, row, col ) * image_scale + along * 0.5f;

    if( std::abs( contrast ) * layers < contrast_threshold )
    {
      return false;
    }

    auto const twice = at( image, row, col ) * 2.0f;
    auto const dxx =
      ( at( image, row, col + 1 ) + at( image, row, col - 1 ) - twice ) *
      second_scale;
    auto const dyy =
      ( at( image, row + 1, col ) + at( image, row - 1, col ) - twice ) *
      second_scale;
    auto const dxy =
      ( at( image, row + 1, col + 1 ) - at( image, row + 1, col - 1 ) -
        at( image, row - 1, col + 1 ) + at( image, row - 1, col - 1 ) ) *
      cross_scale;

    auto const trace = dxx + dyy;
    auto const determinant = dxx * dyy - dxy * dxy;

    // The ratio of principal curvatures, without taking the ratio: an edge
    // has a large trace against a small determinant.
    if( determinant <= 0.0f ||
        trace * trace * edge_threshold >=
          ( edge_threshold + 1.0f ) * ( edge_threshold + 1.0f ) *
          determinant )
    {
      return false;
    }
  }

  kpt.x = ( col + offset_col ) * ( 1 << octave );
  kpt.y = ( row + offset_row ) * ( 1 << octave );
  kpt.octave = octave + ( layer << 8 ) +
               ( round_half_even( ( offset_layer + 0.5 ) * 255 ) << 16 );
  kpt.size = sigma *
             std::pow( 2.0f, ( layer + offset_layer ) /
                             static_cast< float >( layers ) ) *
             ( 1 << octave ) * 2;
  kpt.response = std::abs( contrast );

  return true;
}

// --------------------------------------------------------------------------
/// Every scale-space extremum, as one keypoint per dominant orientation.
std::vector< keypoint >
scale_space_extrema( std::vector< plane > const& gaussian,
                     std::vector< plane > const& dog, int layers,
                     double contrast_threshold, double edge_threshold,
                     double sigma )
{
  auto const per_octave = layers + 2;
  auto const octaves = static_cast< int >( dog.size() ) / per_octave;

  // `cvFloor`, so a contrast threshold of 0.04 over three layers is 1 rather
  // than 1.7 -- the truncation is OpenCV's and it is generous by design.
  auto const threshold = static_cast< float >( static_cast< int >(
    std::floor( 0.5 * contrast_threshold / layers * 255 ) ) );

  std::vector< keypoint > keypoints;
  std::vector< float > hist;

  auto const at = []( plane const& image, int y, int x ) -> float
  {
    return image( static_cast< size_t >( x ), static_cast< size_t >( y ), 0 );
  };

  for( int o = 0; o < octaves; ++o )
  {
    for( int i = 1; i <= layers; ++i )
    {
      auto const index = static_cast< size_t >( o * per_octave + i );
      auto const& image = dog[ index ];
      auto const& below = dog[ index - 1 ];
      auto const& above = dog[ index + 1 ];

      auto const rows = static_cast< int >( image.height() );
      auto const cols = static_cast< int >( image.width() );

      for( int r = img_border; r < rows - img_border; ++r )
      {
        for( int c = img_border; c < cols - img_border; ++c )
        {
          auto const value = at( image, r, c );

          if( !( std::abs( value ) > threshold ) )
          {
            continue;
          }

          auto const extreme = [ & ]( plane const& other, bool centre ) -> bool
          {
            for( int dy = -1; dy <= 1; ++dy )
            {
              for( int dx = -1; dx <= 1; ++dx )
              {
                if( !centre && dy == 0 && dx == 0 ) { continue; }

                auto const neighbour = at( other, r + dy, c + dx );

                if( value > 0.0f ? value < neighbour : value > neighbour )
                {
                  return false;
                }
              }
            }

            return true;
          };

          if( !extreme( image, false ) || !extreme( below, true ) ||
              !extreme( above, true ) )
          {
            continue;
          }

          keypoint kpt;
          auto layer = i;
          auto row = r;
          auto col = c;

          if( !adjust_extremum( dog, kpt, o, layer, row, col, layers,
                                static_cast< float >( contrast_threshold ),
                                static_cast< float >( edge_threshold ),
                                static_cast< float >( sigma ) ) )
          {
            continue;
          }

          auto const octave_scale = kpt.size * 0.5f / ( 1 << o );
          auto const highest = orientation_histogram(
            gaussian[ static_cast< size_t >( o * ( layers + 3 ) + layer ) ],
            col, row, round_half_even( ori_radius * octave_scale ),
            ori_sig_fctr * octave_scale, hist, ori_hist_bins );
          auto const floor_magnitude = highest * ori_peak_ratio;

          for( int j = 0; j < ori_hist_bins; ++j )
          {
            auto const left = static_cast< size_t >(
              j > 0 ? j - 1 : ori_hist_bins - 1 );
            auto const right = static_cast< size_t >(
              j < ori_hist_bins - 1 ? j + 1 : 0 );
            auto const here = static_cast< size_t >( j );

            if( hist[ here ] > hist[ left ] && hist[ here ] > hist[ right ] &&
                hist[ here ] >= floor_magnitude )
            {
              // A parabola through the peak and its two neighbours.
              auto bin = j + 0.5f * ( hist[ left ] - hist[ right ] ) /
                             ( hist[ left ] - 2 * hist[ here ] +
                               hist[ right ] );

              bin = bin < 0 ? ori_hist_bins + bin
                            : bin >= ori_hist_bins ? bin - ori_hist_bins : bin;

              auto found = kpt;

              found.angle = 360.0f - ( 360.0f / ori_hist_bins ) * bin;

              if( std::abs( found.angle - 360.0f ) < FLT_EPSILON )
              {
                found.angle = 0.0f;
              }

              keypoints.push_back( found );
            }
          }
        }
      }
    }
  }

  return keypoints;
}

// --------------------------------------------------------------------------
/// Sort, then drop keypoints that agree on position, size and angle.
///
/// **This reorders**, and the order is part of the contract. `cv::SIFT`
/// returns its keypoints sorted by x -- not in the order it found them -- and
/// nothing downstream can tell where that came from: it is
/// `KeyPointsFilter::removeDuplicatedSorted`, which sorts the keypoints in
/// place before compacting them. Written from a memory of an older OpenCV,
/// where it sorted an *index* and compacted in the original order, this file
/// emitted the same keypoints in detection order -- and every recorded
/// descriptor matrix, every FLANN match set and every homography built on one
/// disagreed, because a matcher pairs row i with keypoint i.
void
remove_duplicates( std::vector< keypoint >& keypoints )
{
  if( keypoints.size() < 2 ) { return; }

  std::sort( keypoints.begin(), keypoints.end(),
             []( keypoint const& a, keypoint const& b )
             {
               if( a.x != b.x ) { return a.x < b.x; }
               if( a.y != b.y ) { return a.y < b.y; }
               if( a.size != b.size ) { return a.size > b.size; }
               if( a.angle != b.angle ) { return a.angle < b.angle; }
               if( a.response != b.response ) { return a.response > b.response; }

               return a.octave > b.octave;
             } );

  size_t kept = 0;

  for( size_t j = 1; j < keypoints.size(); ++j )
  {
    auto const& a = keypoints[ kept ];
    auto const& b = keypoints[ j ];

    if( a.x != b.x || a.y != b.y || a.size != b.size || a.angle != b.angle )
    {
      keypoints[ ++kept ] = keypoints[ j ];
    }
  }

  keypoints.resize( kept + 1 );
}

/// The \p wanted strongest by response, plus everyone tied with the last.
void
retain_best( std::vector< keypoint >& keypoints, int wanted )
{
  if( wanted < 0 || keypoints.size() <= static_cast< size_t >( wanted ) )
  {
    return;
  }

  if( wanted == 0 )
  {
    keypoints.clear();
    return;
  }

  auto const stronger = []( keypoint const& a, keypoint const& b )
  {
    return a.response > b.response;
  };

  std::nth_element( keypoints.begin(), keypoints.begin() + wanted - 1,
                    keypoints.end(), stronger );

  auto const boundary = keypoints[ static_cast< size_t >( wanted ) - 1 ].response;

  // Greater than **or equal**: everyone tied with the boundary is kept, so the
  // result can be longer than `wanted`. Strictly greater drops the boundary
  // keypoints themselves.
  auto const end = std::partition(
    keypoints.begin() + wanted, keypoints.end(),
    [ boundary ]( keypoint const& k ) { return k.response >= boundary; } );

  keypoints.erase( end, keypoints.end() );
}

// --------------------------------------------------------------------------
/// One 128-float descriptor, written into \p out at \p row.
void
describe( plane const& image, float px, float py, float orientation,
          float scale, std::vector< float >& out, size_t row )
{
  auto const d = descr_width;
  auto const n = descr_hist_bins;

  auto const x = round_half_even( px );
  auto const y = round_half_even( py );

  auto cos_t = std::cos( orientation * static_cast< float >( M_PI / 180.0 ) );
  auto sin_t = std::sin( orientation * static_cast< float >( M_PI / 180.0 ) );

  auto const bins_per_rad = n / 360.0f;
  auto const exp_scale = -1.0f / ( d * d * 0.5f );
  auto const hist_width = descr_scl_fctr * scale;

  auto radius = round_half_even(
    hist_width * 1.4142135623730951f * ( d + 1 ) * 0.5f );

  auto const rows = static_cast< int >( image.height() );
  auto const cols = static_cast< int >( image.width() );

  // Clipped to the image diagonal, which is OpenCV guarding its own buffer
  // rather than a property of the descriptor.
  radius = std::min( radius, static_cast< int >( std::sqrt(
    static_cast< double >( cols ) * cols +
    static_cast< double >( rows ) * rows ) ) );

  cos_t /= hist_width;
  sin_t /= hist_width;

  auto const hist_size = static_cast< size_t >( ( d + 2 ) * ( d + 2 ) *
                                                ( n + 2 ) );
  std::vector< float > hist( hist_size, 0.0f );

  for( int i = -radius; i <= radius; ++i )
  {
    for( int j = -radius; j <= radius; ++j )
    {
      // The sample's place in the histogram grid, rotated to the keypoint's
      // orientation. Half a bin comes off so that a sample in the middle of
      // row one lands wholly in row one.
      auto const c_rot = j * cos_t - i * sin_t;
      auto const r_rot = j * sin_t + i * cos_t;
      auto rbin = r_rot + d / 2 - 0.5f;
      auto cbin = c_rot + d / 2 - 0.5f;

      auto const r = y + i;
      auto const c = x + j;

      if( !( rbin > -1 && rbin < d && cbin > -1 && cbin < d &&
             r > 0 && r < rows - 1 && c > 0 && c < cols - 1 ) )
      {
        continue;
      }

      auto const dx = image( static_cast< size_t >( c + 1 ),
                             static_cast< size_t >( r ), 0 ) -
                      image( static_cast< size_t >( c - 1 ),
                             static_cast< size_t >( r ), 0 );
      auto const dy = image( static_cast< size_t >( c ),
                             static_cast< size_t >( r - 1 ), 0 ) -
                      image( static_cast< size_t >( c ),
                             static_cast< size_t >( r + 1 ), 0 );

      auto const angle = fast_atan2( dy, dx );
      auto const magnitude = std::sqrt( dx * dx + dy * dy ) *
                             std::exp( ( c_rot * c_rot + r_rot * r_rot ) *
                                       exp_scale );

      auto obin = ( angle - orientation ) * bins_per_rad;

      auto r0 = floor_of( rbin );
      auto c0 = floor_of( cbin );
      auto o0 = floor_of( obin );

      rbin -= r0;
      cbin -= c0;
      obin -= o0;

      if( o0 < 0 ) { o0 += n; }
      if( o0 >= n ) { o0 -= n; }

      // Trilinear, as eight fractions of one magnitude.
      auto const v_r1 = magnitude * rbin;
      auto const v_r0 = magnitude - v_r1;
      auto const v_rc11 = v_r1 * cbin;
      auto const v_rc10 = v_r1 - v_rc11;
      auto const v_rc01 = v_r0 * cbin;
      auto const v_rc00 = v_r0 - v_rc01;
      auto const v_rco111 = v_rc11 * obin;
      auto const v_rco110 = v_rc11 - v_rco111;
      auto const v_rco101 = v_rc10 * obin;
      auto const v_rco100 = v_rc10 - v_rco101;
      auto const v_rco011 = v_rc01 * obin;
      auto const v_rco010 = v_rc01 - v_rco011;
      auto const v_rco001 = v_rc00 * obin;
      auto const v_rco000 = v_rc00 - v_rco001;

      auto const at = static_cast< size_t >(
        ( ( r0 + 1 ) * ( d + 2 ) + c0 + 1 ) * ( n + 2 ) + o0 );

      hist[ at ] += v_rco000;
      hist[ at + 1 ] += v_rco001;
      hist[ at + static_cast< size_t >( n + 2 ) ] += v_rco010;
      hist[ at + static_cast< size_t >( n + 3 ) ] += v_rco011;
      hist[ at + static_cast< size_t >( ( d + 2 ) * ( n + 2 ) ) ] += v_rco100;
      hist[ at + static_cast< size_t >( ( d + 2 ) * ( n + 2 ) + 1 ) ] +=
        v_rco101;
      hist[ at + static_cast< size_t >( ( d + 3 ) * ( n + 2 ) ) ] += v_rco110;
      hist[ at + static_cast< size_t >( ( d + 3 ) * ( n + 2 ) + 1 ) ] +=
        v_rco111;
    }
  }

  auto const length = static_cast< size_t >( d * d * n );
  std::vector< float > raw( length, 0.0f );

  // The orientation histograms are circular, so the two bins past the end
  // fold back onto the two at the start.
  for( int i = 0; i < d; ++i )
  {
    for( int j = 0; j < d; ++j )
    {
      auto const at = static_cast< size_t >(
        ( ( i + 1 ) * ( d + 2 ) + ( j + 1 ) ) * ( n + 2 ) );

      hist[ at ] += hist[ at + static_cast< size_t >( n ) ];
      hist[ at + 1 ] += hist[ at + static_cast< size_t >( n ) + 1 ];

      for( int k = 0; k < n; ++k )
      {
        raw[ static_cast< size_t >( ( i * d + j ) * n + k ) ] =
          hist[ at + static_cast< size_t >( k ) ];
      }
    }
  }

  // Normalise, clip at a fifth of the norm, normalise again, and scale so the
  // result fits a byte. The clip is what makes the descriptor robust to a
  // change of illumination; the second normalisation is what makes the clip
  // not change the overall magnitude.
  auto squared = 0.0f;

  for( auto const value : raw ) { squared += value * value; }

  auto const clip = std::sqrt( squared ) * descr_mag_thr;

  squared = 0.0f;

  for( auto& value : raw )
  {
    value = std::min( value, clip );
    squared += value * value;
  }

  auto const factor =
    int_descr_fctr / std::max( std::sqrt( squared ), FLT_EPSILON );

  for( size_t k = 0; k < length; ++k )
  {
    // `saturate_cast< uchar >`, kept in a float: the descriptor type is
    // CV_32F but the values are the byte ones.
    auto const value = std::nearbyint( raw[ k ] * factor );

    out[ row * length + k ] =
      static_cast< float >( std::min( std::max( value, 0.0f ), 255.0f ) );
  }
}

} // namespace

// --------------------------------------------------------------------------
int
descriptor_size()
{
  return descr_width * descr_width * descr_hist_bins;
}

// --------------------------------------------------------------------------
void
detect_and_compute( viame::image_of< uint8_t > const& image,
                    settings const& config,
                    std::vector< keypoint >& keypoints,
                    std::vector< float >* descriptors,
                    bool use_provided_keypoints )
{
  auto const layers = config.n_octave_layers;

  auto first_octave = -1;
  auto actual_octaves = 0;

  if( use_provided_keypoints )
  {
    first_octave = 0;
    auto highest_octave = INT_MIN;
    auto actual_layers = 0;

    for( auto const& kpt : keypoints )
    {
      int octave = 0;
      int layer = 0;
      float scale = 0.0f;

      unpack_octave( kpt.octave, octave, layer, scale );
      first_octave = std::min( first_octave, octave );
      highest_octave = std::max( highest_octave, octave );
      actual_layers = std::max( actual_layers, layer - 2 );
    }

    first_octave = std::min( first_octave, 0 );
    actual_octaves = highest_octave - first_octave + 1;
  }

  auto const base = initial_image( image, first_octave < 0,
                                   static_cast< float >( config.sigma ) );

  auto const octaves = actual_octaves > 0
    ? actual_octaves
    : round_half_even(
        std::log( static_cast< double >(
          std::min( base.width(), base.height() ) ) ) / std::log( 2.0 ) - 2 ) -
      first_octave;

  auto const gaussian = gaussian_pyramid( base, octaves, layers,
                                          config.sigma );

  if( !use_provided_keypoints )
  {
    auto const dog = difference_pyramid( gaussian, layers );

    keypoints = scale_space_extrema( gaussian, dog, layers,
                                     config.contrast_threshold,
                                     config.edge_threshold, config.sigma );
    remove_duplicates( keypoints );

    if( config.n_features > 0 )
    {
      retain_best( keypoints, config.n_features );
    }

    if( first_octave < 0 )
    {
      auto const scale = 1.0f / static_cast< float >( 1 << -first_octave );

      for( auto& kpt : keypoints )
      {
        kpt.octave = ( kpt.octave & ~255 ) |
                     ( ( kpt.octave + first_octave ) & 255 );
        kpt.x *= scale;
        kpt.y *= scale;
        kpt.size *= scale;
      }
    }
  }

  if( descriptors == nullptr )
  {
    return;
  }

  auto const width = static_cast< size_t >( descriptor_size() );

  descriptors->assign( keypoints.size() * width, 0.0f );

  viame::image_kernels::parallel_rows (
      0, keypoints.size (), 32,
      [&] ( std::size_t begin, std::size_t end )
      {
        for ( size_t k = begin; k < end; ++k )
        {
          auto const &kpt = keypoints[k];

          int octave = 0;
          int layer = 0;
          float scale = 0.0f;

          unpack_octave ( kpt.octave, octave, layer, scale );

          auto const index =
              static_cast<size_t> ( ( octave - first_octave ) * ( layers + 3 ) + layer );

          auto angle = 360.0f - kpt.angle;

          if ( std::abs ( angle - 360.0f ) < FLT_EPSILON )
          {
            angle = 0.0f;
          }

          describe ( gaussian[index], kpt.x * scale, kpt.y * scale, angle,
                     kpt.size * scale * 0.5f, *descriptors, k );
        }
      } );
}

} // namespace sift

} // namespace viame
