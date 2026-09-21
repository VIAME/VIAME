/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief SURF, ported from OpenCV's `xfeatures2d/src/surf.cpp`
///
/// The algorithm, its constants and its layout are OpenCV's, originally
/// contributed by Liu Liu and substantially revised by Ian Mahon. OpenCV's
/// copyright and BSD licence for that file:
///
///   Copyright (C) 2008, Liu Liu, all rights reserved.
///
///   Redistribution and use in source and binary forms, with or without
///   modification, are permitted provided that the following conditions are
///   met: redistributions of source code must retain the above copyright
///   notice, this list of conditions and the following disclaimer;
///   redistributions in binary form must reproduce it in the documentation
///   and/or other materials provided with the distribution; the name of
///   Contributor may not be used to endorse or promote products derived from
///   this software without specific prior written permission.
///
///   This software is provided by the copyright holders and contributors "as
///   is" and any express or implied warranties, including, but not limited
///   to, the implied warranties of merchantability and fitness for a
///   particular purpose are disclaimed. In no event shall the contributors be
///   liable for any direct, indirect, incidental, special, exemplary, or
///   consequential damages however caused and on any theory of liability,
///   whether in contract, strict liability, or tort arising in any way out of
///   the use of this software, even if advised of the possibility of such
///   damage.
///
/// What changed in the port, and nothing else did:
///
/// - `cv::Mat` became flat `std::vector` buffers with an explicit step. The
///   detector does raw pointer arithmetic across rows (`det1[-step-1]`), and
///   a buffer with one stride is closer to that than an `image_of`, which
///   carries three.
/// - `cv::parallel_for_` became plain loops. The `Mutex` around the keypoint
///   vector went with it.
/// - `cvRound`, `fastAtan2`, `getGaussianKernel`, `integral` and the
///   `INTER_AREA` case of `resize` are reproduced below, because the results
///   have to match OpenCV's and not merely be correct: `fastAtan2` is a
///   polynomial with its own error, and the orientation it returns is what
///   the descriptor is sampled along.
/// - OpenCL, CUDA and the mask argument are dropped. No shipped config passes
///   a mask to this algorithm.

#include "surf.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>

namespace viame {

namespace surf {

namespace {

// ----------------------------------------------------------------------------
// The pieces of OpenCV this leans on, reproduced so the numbers match
// ----------------------------------------------------------------------------

/// `cvRound`. Ties go to even, which is what `lrint` does under the default
/// rounding mode and what OpenCV documents.
inline int
cv_round( double value )
{
  return static_cast< int >( std::lrint( value ) );
}

inline int
cv_floor( double value )
{
  return static_cast< int >( std::floor( value ) );
}

inline int
cv_ceil( double value )
{
  return static_cast< int >( std::ceil( value ) );
}

/// `cv::fastAtan2`: degrees in [0, 360), to about a third of a degree.
///
/// Reproduced rather than replaced with `std::atan2`. The descriptor is
/// sampled along the angle this returns, so a more accurate answer here is a
/// different answer, and the recorded reference is OpenCV's.
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

/// `cv::getGaussianKernel( n, sigma, CV_32F )` with sigma given, which is the
/// closed form rather than the small-kernel table.
std::vector< float >
gaussian_kernel( int n, double sigma )
{
  std::vector< float > kernel( static_cast< size_t >( n ) );

  double const scale = -0.5 / ( sigma * sigma );
  double total = 0.0;

  for( int i = 0; i < n; ++i )
  {
    double const x = i - ( n - 1 ) * 0.5;
    double const value = std::exp( scale * x * x );
    kernel[ i ] = static_cast< float >( value );
    total += value;
  }

  double const norm = 1.0 / total;
  for( int i = 0; i < n; ++i )
  {
    kernel[ i ] = static_cast< float >( kernel[ i ] * norm );
  }

  return kernel;
}

/// `cv::integral( image, sum, CV_32S )`: one row and column bigger than the
/// image, zero along the top and left.
void
integral_image( viame::image_of< uint8_t > const& image,
                std::vector< int >& sum, int& sum_w, int& sum_h )
{
  int const width = static_cast< int >( image.width() );
  int const height = static_cast< int >( image.height() );

  sum_w = width + 1;
  sum_h = height + 1;
  sum.assign( static_cast< size_t >( sum_w ) * sum_h, 0 );

  for( int i = 0; i < height; ++i )
  {
    int row_total = 0;
    int const* previous = &sum[ static_cast< size_t >( i ) * sum_w ];
    int* current = &sum[ static_cast< size_t >( i + 1 ) * sum_w ];
    current[ 0 ] = 0;

    for( int j = 0; j < width; ++j )
    {
      row_total += image( static_cast< size_t >( j ),
                          static_cast< size_t >( i ), 0 );
      current[ j + 1 ] = previous[ j + 1 ] + row_total;
    }
  }
}

/// The `INTER_AREA` case of `cv::resize`, for shrinking only, on one plane of
/// bytes. Each output pixel is the mean of the source rectangle it covers,
/// with the partial rows and columns at its edges weighted by how much of
/// them falls inside.
void
resize_area( uint8_t const* src, int src_w, int src_h,
             uint8_t* dst, int dst_w, int dst_h )
{
  double const scale_x = static_cast< double >( src_w ) / dst_w;
  double const scale_y = static_cast< double >( src_h ) / dst_h;

  // The source columns each output column draws on, and by how much. Computed
  // once and walked for every row.
  struct span { int first; int last; };
  std::vector< span > columns( static_cast< size_t >( dst_w ) );
  std::vector< std::vector< double > > column_weights(
    static_cast< size_t >( dst_w ) );

  for( int dx = 0; dx < dst_w; ++dx )
  {
    double const from = dx * scale_x;
    double const to = std::min( from + scale_x, static_cast< double >( src_w ) );

    int const first = static_cast< int >( std::floor( from ) );
    int const last = std::min( static_cast< int >( std::ceil( to ) ), src_w );

    columns[ dx ] = { first, last };
    auto& weights = column_weights[ dx ];
    weights.resize( static_cast< size_t >( last - first ) );

    for( int sx = first; sx < last; ++sx )
    {
      double const overlap =
        std::min( to, static_cast< double >( sx + 1 ) ) -
        std::max( from, static_cast< double >( sx ) );
      weights[ static_cast< size_t >( sx - first ) ] = overlap;
    }
  }

  for( int dy = 0; dy < dst_h; ++dy )
  {
    double const from_y = dy * scale_y;
    double const to_y =
      std::min( from_y + scale_y, static_cast< double >( src_h ) );

    int const first_y = static_cast< int >( std::floor( from_y ) );
    int const last_y = std::min( static_cast< int >( std::ceil( to_y ) ), src_h );

    for( int dx = 0; dx < dst_w; ++dx )
    {
      auto const& col = columns[ dx ];
      auto const& weights = column_weights[ dx ];

      double total = 0.0;
      double area = 0.0;

      for( int sy = first_y; sy < last_y; ++sy )
      {
        double const wy =
          std::min( to_y, static_cast< double >( sy + 1 ) ) -
          std::max( from_y, static_cast< double >( sy ) );
        uint8_t const* row = src + static_cast< size_t >( sy ) * src_w;

        for( int sx = col.first; sx < col.last; ++sx )
        {
          double const w =
            wy * weights[ static_cast< size_t >( sx - col.first ) ];
          total += row[ sx ] * w;
          area += w;
        }
      }

      int value = area > 0.0 ? cv_round( total / area ) : 0;
      value = std::min( 255, std::max( 0, value ) );
      dst[ static_cast< size_t >( dy ) * dst_w + dx ] =
        static_cast< uint8_t >( value );
    }
  }
}

// ----------------------------------------------------------------------------
// The detector
// ----------------------------------------------------------------------------

int const ORI_SEARCH_INC = 5;
float const ORI_SIGMA = 2.5f;
float const DESC_SIGMA = 3.3f;

/// Wavelet size at the first layer of the first octave.
int const HAAR_SIZE0 = 9;

/// Wavelet size increment between layers. Even, so that the sizes in an
/// octave are all even or all odd and a sample has neighbours above and below.
int const HAAR_SIZE_INC = 6;

struct haar_filter
{
  int p0 = 0, p1 = 0, p2 = 0, p3 = 0;
  float w = 0.0f;
};

inline float
calc_haar_pattern( int const* origin, haar_filter const* f, int n )
{
  double d = 0;
  for( int k = 0; k < n; ++k )
  {
    d += ( origin[ f[ k ].p0 ] + origin[ f[ k ].p3 ] -
           origin[ f[ k ].p1 ] - origin[ f[ k ].p2 ] ) * f[ k ].w;
  }
  return static_cast< float >( d );
}

void
resize_haar_pattern( int const src[][ 5 ], haar_filter* dst, int n,
                     int old_size, int new_size, int width_step )
{
  float const ratio = static_cast< float >( new_size ) / old_size;

  for( int k = 0; k < n; ++k )
  {
    int const dx1 = cv_round( ratio * src[ k ][ 0 ] );
    int const dy1 = cv_round( ratio * src[ k ][ 1 ] );
    int const dx2 = cv_round( ratio * src[ k ][ 2 ] );
    int const dy2 = cv_round( ratio * src[ k ][ 3 ] );

    dst[ k ].p0 = dy1 * width_step + dx1;
    dst[ k ].p1 = dy2 * width_step + dx1;
    dst[ k ].p2 = dy1 * width_step + dx2;
    dst[ k ].p3 = dy2 * width_step + dx2;
    dst[ k ].w = src[ k ][ 4 ] /
                 ( static_cast< float >( dx2 - dx1 ) * ( dy2 - dy1 ) );
  }
}

/// Determinant and trace of the Hessian for one layer of the pyramid.
void
calc_layer_det_and_trace( std::vector< int > const& sum, int sum_w, int sum_h,
                          int size, int sample_step,
                          std::vector< float >& det,
                          std::vector< float >& trace, int layer_w )
{
  int const NX = 3, NY = 3, NXY = 4;
  int const dx_s[ NX ][ 5 ] =
    { { 0, 2, 3, 7, 1 }, { 3, 2, 6, 7, -2 }, { 6, 2, 9, 7, 1 } };
  int const dy_s[ NY ][ 5 ] =
    { { 2, 0, 7, 3, 1 }, { 2, 3, 7, 6, -2 }, { 2, 6, 7, 9, 1 } };
  int const dxy_s[ NXY ][ 5 ] =
    { { 1, 1, 4, 4, 1 }, { 5, 1, 8, 4, -1 },
      { 1, 5, 4, 8, -1 }, { 5, 5, 8, 8, 1 } };

  haar_filter Dx[ NX ], Dy[ NY ], Dxy[ NXY ];

  if( size > sum_h - 1 || size > sum_w - 1 )
  {
    return;
  }

  resize_haar_pattern( dx_s, Dx, NX, 9, size, sum_w );
  resize_haar_pattern( dy_s, Dy, NY, 9, size, sum_w );
  resize_haar_pattern( dxy_s, Dxy, NXY, 9, size, sum_w );

  int const samples_i = 1 + ( sum_h - 1 - size ) / sample_step;
  int const samples_j = 1 + ( sum_w - 1 - size ) / sample_step;

  // Ignore pixels where part of the kernel falls outside the image
  int const margin = ( size / 2 ) / sample_step;

  for( int i = 0; i < samples_i; ++i )
  {
    int const* sum_ptr = &sum[ static_cast< size_t >( i * sample_step ) * sum_w ];
    float* det_ptr =
      &det[ static_cast< size_t >( i + margin ) * layer_w + margin ];
    float* trace_ptr =
      &trace[ static_cast< size_t >( i + margin ) * layer_w + margin ];

    for( int j = 0; j < samples_j; ++j )
    {
      float const dx = calc_haar_pattern( sum_ptr, Dx, 3 );
      float const dy = calc_haar_pattern( sum_ptr, Dy, 3 );
      float const dxy = calc_haar_pattern( sum_ptr, Dxy, 4 );
      sum_ptr += sample_step;
      det_ptr[ j ] = dx * dy - 0.81f * dxy * dxy;
      trace_ptr[ j ] = dx + dy;
    }
  }
}

/// `Matx33f::solve( b, DECOMP_LU )`: Gaussian elimination with partial
/// pivoting, and zeros when the matrix is singular, which is what OpenCV
/// returns and what the caller reads as failure.
bool
solve_3x3( double A[ 3 ][ 3 ], double b[ 3 ], double x[ 3 ] )
{
  for( int i = 0; i < 3; ++i )
  {
    int pivot = i;
    for( int k = i + 1; k < 3; ++k )
    {
      if( std::abs( A[ k ][ i ] ) > std::abs( A[ pivot ][ i ] ) )
      {
        pivot = k;
      }
    }

    if( std::abs( A[ pivot ][ i ] ) < 1e-20 )
    {
      x[ 0 ] = x[ 1 ] = x[ 2 ] = 0.0;
      return false;
    }

    if( pivot != i )
    {
      for( int k = 0; k < 3; ++k ) { std::swap( A[ i ][ k ], A[ pivot ][ k ] ); }
      std::swap( b[ i ], b[ pivot ] );
    }

    for( int k = i + 1; k < 3; ++k )
    {
      double const factor = A[ k ][ i ] / A[ i ][ i ];
      for( int c = i; c < 3; ++c ) { A[ k ][ c ] -= factor * A[ i ][ c ]; }
      b[ k ] -= factor * b[ i ];
    }
  }

  for( int i = 2; i >= 0; --i )
  {
    double value = b[ i ];
    for( int k = i + 1; k < 3; ++k ) { value -= A[ i ][ k ] * x[ k ]; }
    x[ i ] = value / A[ i ][ i ];
  }

  return true;
}

/// Maxima interpolation, after Brown and Lowe: fit a quadratic to the 3x3x3
/// neighbourhood and step to its peak.
bool
interpolate_keypoint( float N9[ 3 ][ 9 ], int dx, int dy, int ds, keypoint& kpt )
{
  double b[ 3 ] =
    { -( N9[ 1 ][ 5 ] - N9[ 1 ][ 3 ] ) / 2.0,
      -( N9[ 1 ][ 7 ] - N9[ 1 ][ 1 ] ) / 2.0,
      -( N9[ 2 ][ 4 ] - N9[ 0 ][ 4 ] ) / 2.0 };

  double A[ 3 ][ 3 ] =
    { { N9[ 1 ][ 3 ] - 2 * N9[ 1 ][ 4 ] + N9[ 1 ][ 5 ],
        ( N9[ 1 ][ 8 ] - N9[ 1 ][ 6 ] - N9[ 1 ][ 2 ] + N9[ 1 ][ 0 ] ) / 4.0,
        ( N9[ 2 ][ 5 ] - N9[ 2 ][ 3 ] - N9[ 0 ][ 5 ] + N9[ 0 ][ 3 ] ) / 4.0 },
      { ( N9[ 1 ][ 8 ] - N9[ 1 ][ 6 ] - N9[ 1 ][ 2 ] + N9[ 1 ][ 0 ] ) / 4.0,
        N9[ 1 ][ 1 ] - 2 * N9[ 1 ][ 4 ] + N9[ 1 ][ 7 ],
        ( N9[ 2 ][ 7 ] - N9[ 2 ][ 1 ] - N9[ 0 ][ 7 ] + N9[ 0 ][ 1 ] ) / 4.0 },
      { ( N9[ 2 ][ 5 ] - N9[ 2 ][ 3 ] - N9[ 0 ][ 5 ] + N9[ 0 ][ 3 ] ) / 4.0,
        ( N9[ 2 ][ 7 ] - N9[ 2 ][ 1 ] - N9[ 0 ][ 7 ] + N9[ 0 ][ 1 ] ) / 4.0,
        N9[ 0 ][ 4 ] - 2 * N9[ 1 ][ 4 ] + N9[ 2 ][ 4 ] } };

  double x[ 3 ] = { 0.0, 0.0, 0.0 };
  solve_3x3( A, b, x );

  bool const ok = ( x[ 0 ] != 0 || x[ 1 ] != 0 || x[ 2 ] != 0 ) &&
                  std::abs( x[ 0 ] ) <= 1 &&
                  std::abs( x[ 1 ] ) <= 1 &&
                  std::abs( x[ 2 ] ) <= 1;

  if( ok )
  {
    kpt.x += static_cast< float >( x[ 0 ] * dx );
    kpt.y += static_cast< float >( x[ 1 ] * dy );
    kpt.size = static_cast< float >(
      cv_round( kpt.size + static_cast< float >( x[ 2 ] * ds ) ) );
  }

  return ok;
}

/// Find the maxima of the Hessian determinant in one layer.
void
find_maxima_in_layer( int sum_w, int sum_h,
                      std::vector< std::vector< float > > const& dets,
                      std::vector< std::vector< float > > const& traces,
                      std::vector< int > const& sizes,
                      std::vector< int > const& layer_widths,
                      std::vector< keypoint >& keypoints,
                      int octave, int layer, float hessian_threshold,
                      int sample_step )
{
  int const size = sizes[ layer ];

  // The integral image is one pixel bigger than the source
  int const layer_rows = ( sum_h - 1 ) / sample_step;
  int const layer_cols = ( sum_w - 1 ) / sample_step;

  // Ignore pixels without a 3x3x3 neighbourhood in the layer above
  int const margin = ( sizes[ layer + 1 ] / 2 ) / sample_step + 1;

  int const step = layer_widths[ layer ];

  for( int i = margin; i < layer_rows - margin; ++i )
  {
    float const* det_ptr = &dets[ layer ][ static_cast< size_t >( i ) * step ];
    float const* trace_ptr =
      &traces[ layer ][ static_cast< size_t >( i ) * step ];

    for( int j = margin; j < layer_cols - margin; ++j )
    {
      float const val0 = det_ptr[ j ];

      if( val0 > hessian_threshold )
      {
        // Start of the wavelet in the integral image. The integer division
        // does not cancel with sample_step -- do not simplify it.
        int const sum_i = sample_step * ( i - ( size / 2 ) / sample_step );
        int const sum_j = sample_step * ( j - ( size / 2 ) / sample_step );

        // The 3x3x3 samples around the maxima, which sits at N9[1][4]
        float const* det1 =
          &dets[ layer - 1 ][ static_cast< size_t >( i ) * step + j ];
        float const* det2 =
          &dets[ layer ][ static_cast< size_t >( i ) * step + j ];
        float const* det3 =
          &dets[ layer + 1 ][ static_cast< size_t >( i ) * step + j ];

        float N9[ 3 ][ 9 ] =
          { { det1[ -step - 1 ], det1[ -step ], det1[ -step + 1 ],
              det1[ -1 ], det1[ 0 ], det1[ 1 ],
              det1[ step - 1 ], det1[ step ], det1[ step + 1 ] },
            { det2[ -step - 1 ], det2[ -step ], det2[ -step + 1 ],
              det2[ -1 ], det2[ 0 ], det2[ 1 ],
              det2[ step - 1 ], det2[ step ], det2[ step + 1 ] },
            { det3[ -step - 1 ], det3[ -step ], det3[ -step + 1 ],
              det3[ -1 ], det3[ 0 ], det3[ 1 ],
              det3[ step - 1 ], det3[ step ], det3[ step + 1 ] } };

        // Non-maxima suppression; val0 is N9[1][4]
        if( val0 > N9[ 0 ][ 0 ] && val0 > N9[ 0 ][ 1 ] && val0 > N9[ 0 ][ 2 ] &&
            val0 > N9[ 0 ][ 3 ] && val0 > N9[ 0 ][ 4 ] && val0 > N9[ 0 ][ 5 ] &&
            val0 > N9[ 0 ][ 6 ] && val0 > N9[ 0 ][ 7 ] && val0 > N9[ 0 ][ 8 ] &&
            val0 > N9[ 1 ][ 0 ] && val0 > N9[ 1 ][ 1 ] && val0 > N9[ 1 ][ 2 ] &&
            val0 > N9[ 1 ][ 3 ] && val0 > N9[ 1 ][ 5 ] &&
            val0 > N9[ 1 ][ 6 ] && val0 > N9[ 1 ][ 7 ] && val0 > N9[ 1 ][ 8 ] &&
            val0 > N9[ 2 ][ 0 ] && val0 > N9[ 2 ][ 1 ] && val0 > N9[ 2 ][ 2 ] &&
            val0 > N9[ 2 ][ 3 ] && val0 > N9[ 2 ][ 4 ] && val0 > N9[ 2 ][ 5 ] &&
            val0 > N9[ 2 ][ 6 ] && val0 > N9[ 2 ][ 7 ] && val0 > N9[ 2 ][ 8 ] )
        {
          float const center_i = sum_i + ( size - 1 ) * 0.5f;
          float const center_j = sum_j + ( size - 1 ) * 0.5f;

          keypoint kpt;
          kpt.x = center_j;
          kpt.y = center_i;
          kpt.size = static_cast< float >( sizes[ layer ] );
          kpt.angle = -1.0f;
          kpt.response = val0;
          kpt.octave = octave;
          kpt.laplacian = ( trace_ptr[ j ] > 0 ) - ( trace_ptr[ j ] < 0 );

          int const ds = size - sizes[ layer - 1 ];

          if( interpolate_keypoint( N9, sample_step, sample_step, ds, kpt ) )
          {
            keypoints.push_back( kpt );
          }
        }
      }
    }
  }
}

/// The order OpenCV sorts keypoints into, which decides which survive a
/// downstream cap and so has to match.
struct keypoint_greater
{
  bool operator()( keypoint const& a, keypoint const& b ) const
  {
    if( a.response > b.response ) { return true; }
    if( a.response < b.response ) { return false; }
    if( a.size > b.size ) { return true; }
    if( a.size < b.size ) { return false; }
    if( a.octave > b.octave ) { return true; }
    if( a.octave < b.octave ) { return false; }
    if( a.y < b.y ) { return false; }
    if( a.y > b.y ) { return true; }
    return a.x < b.x;
  }
};

void
fast_hessian_detector( std::vector< int > const& sum, int sum_w, int sum_h,
                       std::vector< keypoint >& keypoints,
                       int n_octaves, int n_octave_layers,
                       float hessian_threshold )
{
  // Sampling step at the first octave, doubled each octave after.
  int const SAMPLE_STEP0 = 1;

  int const total_layers = ( n_octave_layers + 2 ) * n_octaves;
  int const middle_layers = n_octave_layers * n_octaves;

  std::vector< std::vector< float > > dets( total_layers );
  std::vector< std::vector< float > > traces( total_layers );
  std::vector< int > sizes( total_layers );
  std::vector< int > sample_steps( total_layers );
  std::vector< int > layer_widths( total_layers );
  std::vector< int > middle_indices( middle_layers );

  keypoints.clear();

  int index = 0, middle_index = 0, step = SAMPLE_STEP0;

  for( int octave = 0; octave < n_octaves; ++octave )
  {
    for( int layer = 0; layer < n_octave_layers + 2; ++layer )
    {
      int const rows = ( sum_h - 1 ) / step;
      int const cols = ( sum_w - 1 ) / step;

      dets[ index ].assign( static_cast< size_t >( rows ) * cols, 0.0f );
      traces[ index ].assign( static_cast< size_t >( rows ) * cols, 0.0f );
      layer_widths[ index ] = cols;
      sizes[ index ] = ( HAAR_SIZE0 + HAAR_SIZE_INC * layer ) << octave;
      sample_steps[ index ] = step;

      if( 0 < layer && layer <= n_octave_layers )
      {
        middle_indices[ middle_index++ ] = index;
      }
      ++index;
    }
    step *= 2;
  }

  for( int i = 0; i < total_layers; ++i )
  {
    calc_layer_det_and_trace( sum, sum_w, sum_h, sizes[ i ], sample_steps[ i ],
                              dets[ i ], traces[ i ], layer_widths[ i ] );
  }

  for( int i = 0; i < middle_layers; ++i )
  {
    int const layer = middle_indices[ i ];
    int const octave = i / n_octave_layers;
    find_maxima_in_layer( sum_w, sum_h, dets, traces, sizes, layer_widths,
                          keypoints, octave, layer, hessian_threshold,
                          sample_steps[ layer ] );
  }

  std::sort( keypoints.begin(), keypoints.end(), keypoint_greater() );
}

// ----------------------------------------------------------------------------
// Orientation and descriptor
// ----------------------------------------------------------------------------

int const ORI_RADIUS = 6;
int const ORI_WIN = 60;
int const PATCH_SZ = 20;

void
describe_keypoints( viame::image_of< uint8_t > const& image,
                    std::vector< int > const& sum, int sum_w, int sum_h,
                    std::vector< keypoint >& keypoints,
                    std::vector< float >* descriptors,
                    bool extended, bool upright, int dsize )
{
  int const ori_sample_bound = ( 2 * ORI_RADIUS + 1 ) * ( 2 * ORI_RADIUS + 1 );

  // Sample positions and weights for the orientation pass
  std::vector< std::pair< int, int > > apt( ori_sample_bound );
  std::vector< float > aptw( ori_sample_bound );
  std::vector< float > DW( PATCH_SZ * PATCH_SZ );

  std::vector< float > const g_ori =
    gaussian_kernel( 2 * ORI_RADIUS + 1, ORI_SIGMA );

  int ori_samples = 0;
  for( int i = -ORI_RADIUS; i <= ORI_RADIUS; ++i )
  {
    for( int j = -ORI_RADIUS; j <= ORI_RADIUS; ++j )
    {
      if( i * i + j * j <= ORI_RADIUS * ORI_RADIUS )
      {
        apt[ ori_samples ] = { i, j };
        aptw[ ori_samples++ ] =
          g_ori[ i + ORI_RADIUS ] * g_ori[ j + ORI_RADIUS ];
      }
    }
  }

  std::vector< float > const g_desc = gaussian_kernel( PATCH_SZ, DESC_SIGMA );
  for( int i = 0; i < PATCH_SZ; ++i )
  {
    for( int j = 0; j < PATCH_SZ; ++j )
    {
      DW[ i * PATCH_SZ + j ] = g_desc[ i ] * g_desc[ j ];
    }
  }

  int const NX = 2, NY = 2;
  int const dx_s[ NX ][ 5 ] = { { 0, 0, 2, 4, -1 }, { 2, 0, 4, 4, 1 } };
  int const dy_s[ NY ][ 5 ] = { { 0, 0, 4, 2, 1 }, { 0, 2, 4, 4, -1 } };

  int const width = static_cast< int >( image.width() );
  int const height = static_cast< int >( image.height() );

  std::vector< float > X( ori_sample_bound ), Y( ori_sample_bound );
  std::vector< float > angle( ori_sample_bound );
  std::vector< uint8_t > PATCH( ( PATCH_SZ + 1 ) * ( PATCH_SZ + 1 ) );
  std::vector< float > DX( PATCH_SZ * PATCH_SZ ), DY( PATCH_SZ * PATCH_SZ );
  std::vector< uint8_t > window;

  for( size_t k = 0; k < keypoints.size(); ++k )
  {
    keypoint& kp = keypoints[ k ];
    float const size = kp.size;

    // The sampling intervals and wavelet sizes are relative to s
    float const s = size * 1.2f / 9.0f;

    // Gradients are sampled in a circle of radius 6s with wavelets of size
    // 4s, kept even so the pattern stays symmetric about its centre.
    int const grad_wav_size = 2 * cv_round( 2 * s );

    if( sum_h < grad_wav_size || sum_w < grad_wav_size )
    {
      // The gradient sampling would be meaningless; mark for deletion.
      kp.size = -1.0f;
      continue;
    }

    float descriptor_dir = 360.0f - 90.0f;

    if( !upright )
    {
      haar_filter dx_t[ NX ], dy_t[ NY ];
      resize_haar_pattern( dx_s, dx_t, NX, 4, grad_wav_size, sum_w );
      resize_haar_pattern( dy_s, dy_t, NY, 4, grad_wav_size, sum_w );

      int nangle = 0;
      for( int kk = 0; kk < ori_samples; ++kk )
      {
        int const x =
          cv_round( kp.x + apt[ kk ].first * s -
                    static_cast< float >( grad_wav_size - 1 ) / 2 );
        int const y =
          cv_round( kp.y + apt[ kk ].second * s -
                    static_cast< float >( grad_wav_size - 1 ) / 2 );

        if( y < 0 || y >= sum_h - grad_wav_size ||
            x < 0 || x >= sum_w - grad_wav_size )
        {
          continue;
        }

        int const* ptr = &sum[ static_cast< size_t >( y ) * sum_w + x ];
        float const vx = calc_haar_pattern( ptr, dx_t, 2 );
        float const vy = calc_haar_pattern( ptr, dy_t, 2 );
        X[ nangle ] = vx * aptw[ kk ];
        Y[ nangle ] = vy * aptw[ kk ];
        ++nangle;
      }

      if( nangle == 0 )
      {
        // Too near a border to sample a gradient, so there is no dominant
        // direction; mark for deletion.
        kp.size = -1.0f;
        continue;
      }

      for( int i = 0; i < nangle; ++i )
      {
        angle[ i ] = fast_atan2( Y[ i ], X[ i ] );
      }

      float bestx = 0, besty = 0, descriptor_mod = 0;
      for( int i = 0; i < 360; i += ORI_SEARCH_INC )
      {
        float sumx = 0, sumy = 0;
        for( int j = 0; j < nangle; ++j )
        {
          int const d = std::abs( cv_round( angle[ j ] ) - i );
          if( d < ORI_WIN / 2 || d > 360 - ORI_WIN / 2 )
          {
            sumx += X[ j ];
            sumy += Y[ j ];
          }
        }

        float const temp_mod = sumx * sumx + sumy * sumy;
        if( temp_mod > descriptor_mod )
        {
          descriptor_mod = temp_mod;
          bestx = sumx;
          besty = sumy;
        }
      }

      descriptor_dir = fast_atan2( -besty, bestx );
    }

    kp.angle = descriptor_dir;

    if( !descriptors )
    {
      continue;
    }

    // A window of 20s pixels about the keypoint
    int const win_size = static_cast< int >( ( PATCH_SZ + 1 ) * s );
    if( win_size <= 0 )
    {
      kp.size = -1.0f;
      continue;
    }
    window.resize( static_cast< size_t >( win_size ) * win_size );

    if( !upright )
    {
      float const radians =
        descriptor_dir * static_cast< float >( M_PI / 180.0 );
      float const sin_dir = -std::sin( radians );
      float const cos_dir = std::cos( radians );

      float const win_offset = -static_cast< float >( win_size - 1 ) / 2;
      float start_x = kp.x + win_offset * cos_dir + win_offset * sin_dir;
      float start_y = kp.y - win_offset * sin_dir + win_offset * cos_dir;

      int const ncols1 = width - 1, nrows1 = height - 1;

      for( int i = 0; i < win_size; ++i, start_x += sin_dir, start_y += cos_dir )
      {
        double pixel_x = start_x;
        double pixel_y = start_y;

        for( int j = 0; j < win_size; ++j, pixel_x += cos_dir, pixel_y -= sin_dir )
        {
          int const ix = cv_floor( pixel_x ), iy = cv_floor( pixel_y );

          if( static_cast< unsigned >( ix ) < static_cast< unsigned >( ncols1 ) &&
              static_cast< unsigned >( iy ) < static_cast< unsigned >( nrows1 ) )
          {
            float const a = static_cast< float >( pixel_x - ix );
            float const b = static_cast< float >( pixel_y - iy );
            auto const p00 = image( ix, iy, 0 );
            auto const p01 = image( ix + 1, iy, 0 );
            auto const p10 = image( ix, iy + 1, 0 );
            auto const p11 = image( ix + 1, iy + 1, 0 );
            window[ static_cast< size_t >( i ) * win_size + j ] =
              static_cast< uint8_t >( cv_round(
                p00 * ( 1.f - a ) * ( 1.f - b ) + p01 * a * ( 1.f - b ) +
                p10 * ( 1.f - a ) * b + p11 * a * b ) );
          }
          else
          {
            int const x = std::min( std::max( cv_round( pixel_x ), 0 ), ncols1 );
            int const y = std::min( std::max( cv_round( pixel_y ), 0 ), nrows1 );
            window[ static_cast< size_t >( i ) * win_size + j ] =
              image( x, y, 0 );
          }
        }
      }
    }
    else
    {
      // An axis-aligned rectangle; descriptor_dir is 90 degrees, so sin is 1
      // and cos is 0 and the general path above reduces to this.
      float const win_offset = -static_cast< float >( win_size - 1 ) / 2;
      int start_x = cv_round( kp.x + win_offset );
      int start_y = cv_round( kp.y - win_offset );

      for( int i = 0; i < win_size; ++i, ++start_x )
      {
        int pixel_x = start_x;
        int pixel_y = start_y;

        for( int j = 0; j < win_size; ++j, --pixel_y )
        {
          int const x = std::min( std::max( pixel_x, 0 ), width - 1 );
          int const y = std::min( std::max( pixel_y, 0 ), height - 1 );
          window[ static_cast< size_t >( i ) * win_size + j ] = image( x, y, 0 );
        }
      }
    }

    // Scale the window to PATCH_SZ so each pixel is s across, which makes the
    // gradients wavelets of size 2s.
    resize_area( window.data(), win_size, win_size,
                 PATCH.data(), PATCH_SZ + 1, PATCH_SZ + 1 );

    int const pstep = PATCH_SZ + 1;
    for( int i = 0; i < PATCH_SZ; ++i )
    {
      for( int j = 0; j < PATCH_SZ; ++j )
      {
        float const dw = DW[ i * PATCH_SZ + j ];
        float const vx =
          ( PATCH[ i * pstep + j + 1 ] - PATCH[ i * pstep + j ] +
            PATCH[ ( i + 1 ) * pstep + j + 1 ] -
            PATCH[ ( i + 1 ) * pstep + j ] ) * dw;
        float const vy =
          ( PATCH[ ( i + 1 ) * pstep + j ] - PATCH[ i * pstep + j ] +
            PATCH[ ( i + 1 ) * pstep + j + 1 ] -
            PATCH[ i * pstep + j + 1 ] ) * dw;
        DX[ i * PATCH_SZ + j ] = vx;
        DY[ i * PATCH_SZ + j ] = vy;
      }
    }

    float* vec = &( *descriptors )[ k * static_cast< size_t >( dsize ) ];
    std::fill( vec, vec + dsize, 0.0f );
    double square_mag = 0;

    if( extended )
    {
      // 128 bins: the sums split by the sign of the other gradient
      for( int i = 0; i < 4; ++i )
      {
        for( int j = 0; j < 4; ++j )
        {
          for( int y = i * 5; y < i * 5 + 5; ++y )
          {
            for( int x = j * 5; x < j * 5 + 5; ++x )
            {
              float const tx = DX[ y * PATCH_SZ + x ];
              float const ty = DY[ y * PATCH_SZ + x ];

              if( ty >= 0 )
              {
                vec[ 0 ] += tx;
                vec[ 1 ] += std::abs( tx );
              }
              else
              {
                vec[ 2 ] += tx;
                vec[ 3 ] += std::abs( tx );
              }

              if( tx >= 0 )
              {
                vec[ 4 ] += ty;
                vec[ 5 ] += std::abs( ty );
              }
              else
              {
                vec[ 6 ] += ty;
                vec[ 7 ] += std::abs( ty );
              }
            }
          }

          for( int kk = 0; kk < 8; ++kk ) { square_mag += vec[ kk ] * vec[ kk ]; }
          vec += 8;
        }
      }
    }
    else
    {
      // 64 bins
      for( int i = 0; i < 4; ++i )
      {
        for( int j = 0; j < 4; ++j )
        {
          for( int y = i * 5; y < i * 5 + 5; ++y )
          {
            for( int x = j * 5; x < j * 5 + 5; ++x )
            {
              float const tx = DX[ y * PATCH_SZ + x ];
              float const ty = DY[ y * PATCH_SZ + x ];
              vec[ 0 ] += tx;
              vec[ 1 ] += ty;
              vec[ 2 ] += std::abs( tx );
              vec[ 3 ] += std::abs( ty );
            }
          }

          for( int kk = 0; kk < 4; ++kk ) { square_mag += vec[ kk ] * vec[ kk ]; }
          vec += 4;
        }
      }
    }

    // A unit vector, which is what makes the descriptor contrast invariant
    vec = &( *descriptors )[ k * static_cast< size_t >( dsize ) ];
    float const scale =
      static_cast< float >( 1.0 / ( std::sqrt( square_mag ) + FLT_EPSILON ) );
    for( int kk = 0; kk < dsize; ++kk ) { vec[ kk ] *= scale; }
  }
}

} // namespace

// ----------------------------------------------------------------------------

int
descriptor_size( settings const& config )
{
  return config.extended ? 128 : 64;
}

void
detect_and_compute( viame::image_of< uint8_t > const& image,
                    settings const& config,
                    std::vector< keypoint >& keypoints,
                    std::vector< float >* descriptors,
                    bool use_provided_keypoints )
{
  std::vector< int > sum;
  int sum_w = 0, sum_h = 0;
  integral_image( image, sum, sum_w, sum_h );

  if( !use_provided_keypoints )
  {
    fast_hessian_detector( sum, sum_w, sum_h, keypoints, config.n_octaves,
                           config.n_octaves_layers,
                           static_cast< float >( config.hessian_threshold ) );
  }

  int const dsize = descriptor_size( config );
  size_t const n = keypoints.size();

  if( n == 0 )
  {
    if( descriptors ) { descriptors->clear(); }
    return;
  }

  if( descriptors )
  {
    descriptors->assign( n * static_cast< size_t >( dsize ), 0.0f );
  }

  // Called even when no descriptor is wanted: it is what assigns each
  // keypoint its orientation.
  describe_keypoints( image, sum, sum_w, sum_h, keypoints, descriptors,
                      config.extended, config.upright, dsize );

  // Drop the keypoints the orientation pass marked, keeping the descriptors
  // in step with them.
  size_t kept = 0;
  for( size_t i = 0; i < n; ++i )
  {
    if( keypoints[ i ].size > 0 )
    {
      if( i > kept )
      {
        keypoints[ kept ] = keypoints[ i ];
        if( descriptors )
        {
          std::memcpy( &( *descriptors )[ kept * dsize ],
                       &( *descriptors )[ i * dsize ],
                       static_cast< size_t >( dsize ) * sizeof( float ) );
        }
      }
      ++kept;
    }
  }

  if( kept < n )
  {
    keypoints.resize( kept );
    if( descriptors )
    {
      descriptors->resize( kept * static_cast< size_t >( dsize ) );
    }
  }
}

} // namespace surf

} // namespace viame
