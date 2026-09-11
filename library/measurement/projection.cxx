/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Pinhole projection, lens distortion and stereo rectification

#include "projection.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace kv = kwiver::vital;

namespace viame {

namespace measurement {

namespace {

// ----------------------------------------------------------------------------
/// The coefficients, padded to fourteen, with the unimplemented ones refused.
///
/// OpenCV reads the length to decide the model: four is the two radial and
/// the two tangential terms, five adds `k3`, eight adds the rational
/// denominator `k4 k5 k6`, twelve adds the thin prism and fourteen the tilt.
/// VIAME's calibrations write five; the two beyond eight are refused rather
/// than dropped, because dropping them would model a different lens without
/// saying so.
std::array< double, 14 >
padded( distortion_t const& coefficients )
{
  std::array< double, 14 > out{};

  if( coefficients.empty() )
  {
    return out;
  }

  auto const count = coefficients.size();

  if( count != 4 && count != 5 && count != 8 && count != 12 && count != 14 )
  {
    throw std::invalid_argument(
      "distortion has " + std::to_string( count ) + " coefficients; OpenCV's "
      "models are 4, 5, 8, 12 and 14" );
  }

  for( size_t i = 0; i < count && i < out.size(); ++i )
  {
    out[ i ] = coefficients[ i ];
  }

  for( size_t i = 8; i < count; ++i )
  {
    if( coefficients[ i ] != 0.0 )
    {
      throw std::invalid_argument(
        "the thin prism and tilt distortion terms are not implemented; "
        "coefficient " + std::to_string( i ) + " is non zero" );
    }
  }

  return out;
}

// ----------------------------------------------------------------------------
/// A normalised point through the lens, which is the forward distortion.
void
distort( double x, double y, std::array< double, 14 > const& k,
         double& out_x, double& out_y )
{
  double const r2 = x * x + y * y;
  double const r4 = r2 * r2;
  double const r6 = r4 * r2;

  double const numerator = 1.0 + k[ 0 ] * r2 + k[ 1 ] * r4 + k[ 4 ] * r6;
  double const denominator = 1.0 + k[ 5 ] * r2 + k[ 6 ] * r4 + k[ 7 ] * r6;

  double const radial = numerator / denominator;

  double const a1 = 2.0 * x * y;
  double const a2 = r2 + 2.0 * x * x;
  double const a3 = r2 + 2.0 * y * y;

  out_x = x * radial + k[ 2 ] * a1 + k[ 3 ] * a2;
  out_y = y * radial + k[ 2 ] * a3 + k[ 3 ] * a1;
}

// ----------------------------------------------------------------------------
/// OpenCV's fixed number of iterations, and its formula, for the inverse.
///
/// Five passes with no convergence test, which is what `cv::undistortPoints`
/// runs when it is given no term criteria -- and it is given none anywhere in
/// VIAME. It is a fixed-point iteration rather than a solve, so on a strong
/// lens it does not converge; that is OpenCV's behaviour and reproducing it
/// is the point.
constexpr int undistort_iterations = 5;

void
undistort( double u, double v, std::array< double, 14 > const& k,
           double& out_x, double& out_y )
{
  double x = u;
  double y = v;

  double const x0 = x;
  double const y0 = y;

  for( int pass = 0; pass < undistort_iterations; ++pass )
  {
    double const r2 = x * x + y * y;
    double const r4 = r2 * r2;
    double const r6 = r4 * r2;

    double const inverse_radial =
      ( 1.0 + k[ 5 ] * r2 + k[ 6 ] * r4 + k[ 7 ] * r6 ) /
      ( 1.0 + k[ 0 ] * r2 + k[ 1 ] * r4 + k[ 4 ] * r6 );

    // The iteration diverges on a strong lens far from the centre, and
    // OpenCV's guard against that is this: as soon as the radial factor goes
    // negative, give up and return the **distorted** normalised point. A
    // wide-angle rig hits it at the image corners, so reproducing it is not
    // optional -- `cv::undistortPoints` there returns its input unchanged.
    if( inverse_radial < 0.0 )
    {
      x = x0;
      y = y0;
      break;
    }

    double const delta_x = 2.0 * k[ 2 ] * x * y + k[ 3 ] * ( r2 + 2.0 * x * x );
    double const delta_y = k[ 2 ] * ( r2 + 2.0 * y * y ) + 2.0 * k[ 3 ] * x * y;

    x = ( x0 - delta_x ) * inverse_radial;
    y = ( y0 - delta_y ) * inverse_radial;
  }

  out_x = x;
  out_y = y;
}

// ----------------------------------------------------------------------------
/// A rotation vector to a rotation matrix, which is `cv::Rodrigues`.
kv::matrix_3x3d
rodrigues( kv::vector_3d const& vector )
{
  double const angle = vector.norm();

  if( angle == 0.0 )
  {
    return kv::matrix_3x3d::Identity();
  }

  kv::vector_3d const axis = vector / angle;

  kv::matrix_3x3d cross;
  cross( 0, 0 ) = 0.0;         cross( 0, 1 ) = -axis[ 2 ]; cross( 0, 2 ) = axis[ 1 ];
  cross( 1, 0 ) = axis[ 2 ];   cross( 1, 1 ) = 0.0;        cross( 1, 2 ) = -axis[ 0 ];
  cross( 2, 0 ) = -axis[ 1 ];  cross( 2, 1 ) = axis[ 0 ];  cross( 2, 2 ) = 0.0;

  return kv::matrix_3x3d::Identity() + std::sin( angle ) * cross +
         ( 1.0 - std::cos( angle ) ) * ( cross * cross );
}

// ----------------------------------------------------------------------------
/// The inverse, which is `cv::Rodrigues` the other way.
///
/// The branch-free form OpenCV uses: the axis comes from the antisymmetric
/// part and the angle from the trace, with the degenerate cases handled by
/// clamping rather than by a special case, since a calibration's relative
/// rotation is always small.
kv::vector_3d
inverse_rodrigues( kv::matrix_3x3d const& matrix )
{
  kv::vector_3d axis( matrix( 2, 1 ) - matrix( 1, 2 ),
                      matrix( 0, 2 ) - matrix( 2, 0 ),
                      matrix( 1, 0 ) - matrix( 0, 1 ) );

  double const s = axis.norm() * 0.5;
  double const c =
    ( matrix( 0, 0 ) + matrix( 1, 1 ) + matrix( 2, 2 ) - 1.0 ) * 0.5;

  double const clamped = std::max( -1.0, std::min( 1.0, c ) );
  double const angle = std::acos( clamped );

  if( s < 1e-10 )
  {
    return kv::vector_3d( 0.0, 0.0, 0.0 );
  }

  return axis * ( angle / ( 2.0 * s ) );
}

// ----------------------------------------------------------------------------
kv::matrix_3x4d
widened( kv::matrix_3x3d const& intrinsics )
{
  kv::matrix_3x4d out;

  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 3; ++c )
    {
      out( r, c ) = intrinsics( r, c );
    }

    out( r, 3 ) = 0.0;
  }

  return out;
}

// ----------------------------------------------------------------------------
/// `icvGetRectangles`: the largest rectangle inside the rectified border and
/// the smallest one containing it.
///
/// A nine by nine grid over the source image, mapped through the
/// rectification, exactly as OpenCV samples it. **In double rather than
/// float**, which is the one deliberate difference in this file: OpenCV
/// stores those eighty-one points as `CV_32FC2`, so its rectangles carry
/// about seven digits and the rectified focal length below inherits that.
void
rectangles( kv::matrix_3x3d const& intrinsics,
            distortion_t const& coefficients,
            kv::matrix_3x3d const& rotation,
            kv::matrix_3x4d const& projection,
            size_t width, size_t height,
            double inner[ 4 ], double outer[ 4 ] )
{
  constexpr int n = 9;

  double inner_x0 = -std::numeric_limits< double >::max();
  double inner_x1 = std::numeric_limits< double >::max();
  double inner_y0 = -std::numeric_limits< double >::max();
  double inner_y1 = std::numeric_limits< double >::max();

  double outer_x0 = std::numeric_limits< double >::max();
  double outer_x1 = -std::numeric_limits< double >::max();
  double outer_y0 = std::numeric_limits< double >::max();
  double outer_y1 = -std::numeric_limits< double >::max();

  for( int y = 0; y < n; ++y )
  {
    for( int x = 0; x < n; ++x )
    {
      // The grid spans the last pixel, not the image extent: `width - 1`.
      // One off here moves the rectified focal length by half a per cent,
      // which is the kind of thing only a check against OpenCV finds.
      kv::vector_2d const source(
        static_cast< double >( x ) * ( static_cast< double >( width ) - 1.0 ) /
          ( n - 1 ),
        static_cast< double >( y ) * ( static_cast< double >( height ) - 1.0 ) /
          ( n - 1 ) );

      auto const point = undistort_point( source, intrinsics, coefficients,
                                          rotation, projection );

      outer_x0 = std::min( outer_x0, point[ 0 ] );
      outer_x1 = std::max( outer_x1, point[ 0 ] );
      outer_y0 = std::min( outer_y0, point[ 1 ] );
      outer_y1 = std::max( outer_y1, point[ 1 ] );

      if( x == 0 )     { inner_x0 = std::max( inner_x0, point[ 0 ] ); }
      if( x == n - 1 ) { inner_x1 = std::min( inner_x1, point[ 0 ] ); }
      if( y == 0 )     { inner_y0 = std::max( inner_y0, point[ 1 ] ); }
      if( y == n - 1 ) { inner_y1 = std::min( inner_y1, point[ 1 ] ); }
    }
  }

  inner[ 0 ] = inner_x0;
  inner[ 1 ] = inner_y0;
  inner[ 2 ] = inner_x1 - inner_x0;
  inner[ 3 ] = inner_y1 - inner_y0;

  outer[ 0 ] = outer_x0;
  outer[ 1 ] = outer_y0;
  outer[ 2 ] = outer_x1 - outer_x0;
  outer[ 3 ] = outer_y1 - outer_y0;
}

} // namespace

// ----------------------------------------------------------------------------
kv::vector_2d
project_point( kv::vector_3d const& point,
               kv::matrix_3x3d const& intrinsics,
               distortion_t const& coefficients )
{
  return project_point( point, kv::matrix_3x3d::Identity(), intrinsics,
                        coefficients );
}

// ----------------------------------------------------------------------------
kv::vector_2d
project_point( kv::vector_3d const& point,
               kv::matrix_3x3d const& rotation,
               kv::matrix_3x3d const& intrinsics,
               distortion_t const& coefficients )
{
  kv::vector_3d const rotated = rotation * point;

  double const z = rotated[ 2 ] != 0.0 ? 1.0 / rotated[ 2 ] : 1.0;
  double const x = rotated[ 0 ] * z;
  double const y = rotated[ 1 ] * z;

  double distorted_x = x;
  double distorted_y = y;

  if( !coefficients.empty() )
  {
    distort( x, y, padded( coefficients ), distorted_x, distorted_y );
  }

  return kv::vector_2d(
    intrinsics( 0, 0 ) * distorted_x + intrinsics( 0, 1 ) * distorted_y +
      intrinsics( 0, 2 ),
    intrinsics( 1, 1 ) * distorted_y + intrinsics( 1, 2 ) );
}

// ----------------------------------------------------------------------------
kv::vector_2d
undistort_point( kv::vector_2d const& point,
                 kv::matrix_3x3d const& intrinsics,
                 distortion_t const& coefficients,
                 kv::matrix_3x3d const& rotation,
                 kv::matrix_3x4d const& projection )
{
  double const fx = intrinsics( 0, 0 );
  double const fy = intrinsics( 1, 1 );
  double const cx = intrinsics( 0, 2 );
  double const cy = intrinsics( 1, 2 );
  double const skew = intrinsics( 0, 1 );

  double y = ( point[ 1 ] - cy ) / fy;
  double x = ( point[ 0 ] - cx - skew * y ) / fx;

  if( !coefficients.empty() )
  {
    undistort( x, y, padded( coefficients ), x, y );
  }

  // The **first three columns only**. `cv::undistortPoints` takes a 3 by 4
  // `P` and then uses `cvGetCols( matP, 0, 3 )`, so the baseline term in a
  // rectified `P2` is ignored -- which is right, since this maps a direction
  // rather than a point. Passing the whole thing instead puts every sample
  // of the right camera tens of thousands of pixels away, and the inscribed
  // rectangle `stereo_rectify` scales to then comes out of the wrong camera.
  kv::vector_3d const ray = rotation * kv::vector_3d( x, y, 1.0 );

  double const w = projection( 2, 0 ) * ray[ 0 ] +
                   projection( 2, 1 ) * ray[ 1 ] +
                   projection( 2, 2 ) * ray[ 2 ];

  double const scale = w != 0.0 ? 1.0 / w : 1.0;

  return kv::vector_2d(
    ( projection( 0, 0 ) * ray[ 0 ] + projection( 0, 1 ) * ray[ 1 ] +
      projection( 0, 2 ) * ray[ 2 ] ) * scale,
    ( projection( 1, 0 ) * ray[ 0 ] + projection( 1, 1 ) * ray[ 1 ] +
      projection( 1, 2 ) * ray[ 2 ] ) * scale );
}

// ----------------------------------------------------------------------------
rectification
stereo_rectify( kv::matrix_3x3d const& left_intrinsics,
                distortion_t const& left_distortion,
                kv::matrix_3x3d const& right_intrinsics,
                distortion_t const& right_distortion,
                size_t width, size_t height,
                kv::matrix_3x3d const& rotation,
                kv::vector_3d const& translation )
{
  auto const nx = static_cast< double >( width );
  auto const ny = static_cast< double >( height );

  // Rotate each camera half way toward the other, so the pair ends up
  // symmetric about the original baseline.
  kv::vector_3d const half = inverse_rodrigues( rotation ) * -0.5;
  kv::matrix_3x3d const half_rotation = rodrigues( half );

  kv::vector_3d t = half_rotation * translation;

  // Which axis the baseline runs along decides whether the rig rectifies
  // horizontally or vertically.
  unsigned const idx = std::abs( t[ 0 ] ) > std::abs( t[ 1 ] ) ? 0u : 1u;

  double const c = t[ idx ];
  double const nt = t.norm();

  if( nt == 0.0 )
  {
    throw std::invalid_argument( "stereo_rectify: the two cameras coincide" );
  }

  kv::vector_3d uu( 0.0, 0.0, 0.0 );
  uu[ idx ] = c > 0.0 ? 1.0 : -1.0;

  // The rotation about the optical axis that brings the baseline onto it.
  kv::vector_3d ww = t.cross( uu );
  double const nw = ww.norm();

  if( nw > 0.0 )
  {
    ww *= std::acos( std::abs( c ) / nt ) / nw;
  }

  kv::matrix_3x3d const w_rotation = rodrigues( ww );

  rectification out;
  out.left_rotation = w_rotation * half_rotation.transpose();
  out.right_rotation = w_rotation * half_rotation;

  t = out.right_rotation * translation;

  // One focal length for both, from the axis the baseline does **not** run
  // along -- `idx ^ 1` -- which is the one the rectification leaves alone.
  double focal =
    ( left_intrinsics( idx ^ 1, idx ^ 1 ) +
      right_intrinsics( idx ^ 1, idx ^ 1 ) ) * 0.5;

  // Where each rectified image's centre has to be for the four corners of
  // the original to sit symmetrically about it.
  double centre_x[ 2 ] = { 0.0, 0.0 };
  double centre_y[ 2 ] = { 0.0, 0.0 };

  for( unsigned k = 0; k < 2; ++k )
  {
    auto const& intrinsics = k == 0 ? left_intrinsics : right_intrinsics;
    auto const& coefficients = k == 0 ? left_distortion : right_distortion;
    auto const& camera_rotation = k == 0 ? out.left_rotation
                                         : out.right_rotation;

    kv::matrix_3x3d centred;
    centred.setZero();
    centred( 0, 0 ) = focal;
    centred( 1, 1 ) = focal;
    centred( 2, 2 ) = 1.0;

    double sum_x = 0.0;
    double sum_y = 0.0;

    for( unsigned i = 0; i < 4; ++i )
    {
      kv::vector_2d const corner( ( i % 2 ) * ( nx - 1 ),
                                  ( i < 2 ? 0.0 : 1.0 ) * ( ny - 1 ) );

      auto const normalised = undistort_point(
        corner, intrinsics, coefficients, kv::matrix_3x3d::Identity(),
        widened( kv::matrix_3x3d::Identity() ) );

      auto const projected = project_point(
        kv::vector_3d( normalised[ 0 ], normalised[ 1 ], 1.0 ),
        camera_rotation, centred, {} );

      sum_x += projected[ 0 ];
      sum_y += projected[ 1 ];
    }

    centre_x[ k ] = ( nx - 1 ) * 0.5 - sum_x * 0.25;
    centre_y[ k ] = ( ny - 1 ) * 0.5 - sum_y * 0.25;
  }

  // `CALIB_ZERO_DISPARITY`: both principal points at the same place, so a
  // correspondence is a pure shift along one axis.
  centre_x[ 0 ] = centre_x[ 1 ] = ( centre_x[ 0 ] + centre_x[ 1 ] ) * 0.5;
  centre_y[ 0 ] = centre_y[ 1 ] = ( centre_y[ 0 ] + centre_y[ 1 ] ) * 0.5;

  auto build = [ & ]( unsigned k )
  {
    kv::matrix_3x4d out_projection;
    out_projection.setZero();
    out_projection( 0, 0 ) = focal;
    out_projection( 1, 1 ) = focal;
    out_projection( 0, 2 ) = centre_x[ k ];
    out_projection( 1, 2 ) = centre_y[ k ];
    out_projection( 2, 2 ) = 1.0;
    return out_projection;
  };

  out.left_projection = build( 0 );
  out.right_projection = build( 1 );
  out.right_projection( idx, 3 ) = t[ idx ] * focal;

  // `alpha = 0`: scale until the rectified image is entirely valid, which is
  // the inscribed rectangle rather than the bounding one.
  double inner1[ 4 ], outer1[ 4 ], inner2[ 4 ], outer2[ 4 ];

  rectangles( left_intrinsics, left_distortion, out.left_rotation,
              out.left_projection, width, height, inner1, outer1 );
  rectangles( right_intrinsics, right_distortion, out.right_rotation,
              out.right_projection, width, height, inner2, outer2 );

  double const cx1_0 = centre_x[ 0 ];
  double const cy1_0 = centre_y[ 0 ];
  double const cx2_0 = centre_x[ 1 ];
  double const cy2_0 = centre_y[ 1 ];

  double scale = std::max(
    std::max( std::max( cx1_0 / ( cx1_0 - inner1[ 0 ] ),
                        cy1_0 / ( cy1_0 - inner1[ 1 ] ) ),
              ( nx - 1 - cx1_0 ) / ( inner1[ 0 ] + inner1[ 2 ] - cx1_0 ) ),
    ( ny - 1 - cy1_0 ) / ( inner1[ 1 ] + inner1[ 3 ] - cy1_0 ) );

  scale = std::max(
    std::max(
      std::max( std::max( cx2_0 / ( cx2_0 - inner2[ 0 ] ),
                          cy2_0 / ( cy2_0 - inner2[ 1 ] ) ),
                ( nx - 1 - cx2_0 ) / ( inner2[ 0 ] + inner2[ 2 ] - cx2_0 ) ),
      ( ny - 1 - cy2_0 ) / ( inner2[ 1 ] + inner2[ 3 ] - cy2_0 ) ),
    scale );

  focal *= scale;

  out.left_projection( 0, 0 ) = focal;
  out.left_projection( 1, 1 ) = focal;
  out.right_projection( 0, 0 ) = focal;
  out.right_projection( 1, 1 ) = focal;
  out.right_projection( idx, 3 ) *= scale;

  // Disparity to depth. Column two whichever way the rig is oriented -- that
  // is OpenCV's layout, not a horizontal special case -- and the divisor is
  // the **unscaled** rectified translation along the baseline axis. Its last
  // entry is zero whenever the two principal points agree, which
  // `CALIB_ZERO_DISPARITY` guarantees.
  double const offset = idx == 0 ? cx1_0 - cx2_0 : cy1_0 - cy2_0;

  out.disparity_to_depth.setZero();
  out.disparity_to_depth( 0, 0 ) = 1.0;
  out.disparity_to_depth( 0, 3 ) = -cx1_0;
  out.disparity_to_depth( 1, 1 ) = 1.0;
  out.disparity_to_depth( 1, 3 ) = -cy1_0;
  out.disparity_to_depth( 2, 3 ) = focal;
  out.disparity_to_depth( 3, 2 ) = -1.0 / t[ idx ];
  out.disparity_to_depth( 3, 3 ) = offset / t[ idx ];

  return out;
}

// ----------------------------------------------------------------------------
void
rectification_maps( kv::matrix_3x3d const& intrinsics,
                    distortion_t const& coefficients,
                    kv::matrix_3x3d const& rotation,
                    kv::matrix_3x4d const& projection,
                    size_t width, size_t height,
                    kv::image_of< float >& map_x,
                    kv::image_of< float >& map_y )
{
  map_x = kv::image_of< float >( width, height, 1 );
  map_y = kv::image_of< float >( width, height, 1 );

  // From the rectified image back to the original: undo the new projection,
  // undo the rectifying rotation, distort, and apply the original intrinsics.
  kv::matrix_3x3d rectified;
  for( unsigned r = 0; r < 3; ++r )
  {
    for( unsigned c = 0; c < 3; ++c )
    {
      rectified( r, c ) = projection( r, c );
    }
  }

  kv::matrix_3x3d const inverse = ( rotation.transpose() *
                                    rectified.inverse() );

  auto const k = padded( coefficients );
  bool const has_distortion = !coefficients.empty();

  for( size_t j = 0; j < height; ++j )
  {
    for( size_t i = 0; i < width; ++i )
    {
      kv::vector_3d const ray =
        inverse * kv::vector_3d( static_cast< double >( i ),
                                 static_cast< double >( j ), 1.0 );

      double const w = ray[ 2 ] != 0.0 ? 1.0 / ray[ 2 ] : 1.0;
      double const x = ray[ 0 ] * w;
      double const y = ray[ 1 ] * w;

      double distorted_x = x;
      double distorted_y = y;

      if( has_distortion )
      {
        distort( x, y, k, distorted_x, distorted_y );
      }

      map_x( i, j, 0 ) = static_cast< float >(
        intrinsics( 0, 0 ) * distorted_x + intrinsics( 0, 1 ) * distorted_y +
        intrinsics( 0, 2 ) );
      map_y( i, j, 0 ) = static_cast< float >(
        intrinsics( 1, 1 ) * distorted_y + intrinsics( 1, 2 ) );
    }
  }
}

} // namespace measurement

} // namespace viame
