// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of \link kwiver::vital::camera_intrinsics_
 *        camera_intrinsics_<T> \endlink class
 *        for \c T = { \c float, \c double }
 */

#include <vital/types/camera_intrinsics.h>
#include <vital/io/eigen_io.h>
#include <vital/math_constants.h>
#include <Eigen/Dense>

#include <cmath>
#include <iomanip>

namespace kwiver {
namespace vital {

/// Convert to a 3x3 calibration matrix
matrix_3x3d
camera_intrinsics
::as_matrix() const
{
  matrix_3x3d K;
  const double f = this->focal_length();
  const vector_2d pp = this->principal_point();
  K << f, this->skew(), pp.x(),
    0, f / this->aspect_ratio(), pp.y(),
    0, 0, 1;
  return K;
}

/// Map normalized image coordinates into actual image coordinates
vector_2d
camera_intrinsics
::map( const vector_2d& point ) const
{
  // apply radial and tangential distortion if coefficients are provided
  const vector_2d pt = this->distort( point );
  const vector_2d pp = this->principal_point();
  const double f = this->focal_length();

  return vector_2d( pt.x() * f + pt.y() * this->skew() + pp.x(),
                    pt.y() * f / this->aspect_ratio() + pp.y() );
}

/// Map a 3D point in camera coordinates into actual image coordinates
vector_2d
camera_intrinsics
::map( const vector_3d& norm_hpt ) const
{
  return this->map( vector_2d( norm_hpt[0] / norm_hpt[2],
                               norm_hpt[1] / norm_hpt[2] ) );
}

/// Unmap actual image coordinates back into normalized image coordinates
vector_2d
camera_intrinsics
::unmap( const vector_2d& pt ) const
{
  const double f = this->focal_length();
  const vector_2d p0 = pt - this->principal_point();
  const double y = p0.y() * this->aspect_ratio() / f;
  const double x = ( p0.x() - y * this->skew() ) / f;

  return this->undistort( vector_2d( x, y ) );
}

/// Check if a 3D point in camera coordinates can map into image coordinates
bool
camera_intrinsics
::is_map_valid(const vector_3d& norm_hpt) const
{
  return this->is_map_valid(vector_2d(norm_hpt[0] / norm_hpt[2],
                                      norm_hpt[1] / norm_hpt[2]));
}

namespace // anonymous namespace
{

/// Compute the radial distortion scaling
/** Distortion scaling is a function of the squared radius \p r2
 *  and the distortion parameters \p d
 */
template < typename T >
T
radial_distortion_scale( const T                r2,
                         const Eigen::VectorXd&  d )
{
  T scale = T( 1 );

  if ( d.rows() > 0 )
  {
    scale += r2 * d[0];
    if ( d.rows() > 1 )
    {
      const T r4 = r2 * r2;
      scale += r4 * d[1];
      if ( d.rows() > 4 )
      {
        const T r6 = r2 * r4;
        scale += r6 * d[4];
        if ( d.rows() > 7 )
        {
          scale /= T( 1 ) + r2 * d[5] + r4 * d[6] + r6 * d[7];
        }
      }
    }
  }
  return scale;
}

/// Compute radial distortion as a scaling and offset
/** For a point \p pt and distortion coefficients \p d compute
 *  a scale and offset such that distortion can be applied as
 *  \code
 *    distorted_pt = pt * scale + offset;
 *  \endcode
 */
template < typename T >
void
distortion_scale_offset( const Eigen::Matrix< T, 2, 1 >& pt,
                         const Eigen::VectorXd& d,
                         T& scale, Eigen::Matrix< T, 2, 1 >& offset )
{
  const T x2 = pt.x() * pt.x();
  const T y2 = pt.y() * pt.y();
  const T r2 = x2 + y2;

  scale = radial_distortion_scale( r2, d );
  offset = Eigen::Matrix< T, 2, 1 > ( T( 0 ), T( 0 ) );
  if ( d.rows() > 3 )
  {
    const T two_xy = 2 * pt.x() * pt.y();
    offset = Eigen::Matrix< T, 2, 1 > ( d[2] * two_xy + d[3] * ( r2 + 2 * x2 ),
                                        d[3] * two_xy + d[2] * ( r2 + 2 * y2 ) );
  }
}

/// Compute the derivative of the radial distortion as a function of \p r2
template < typename T >
T
radial_distortion_deriv( const T                r2,
                         const Eigen::VectorXd&  d )
{
  T deriv = T( 0 );

  if ( d.rows() > 0 )
  {
    deriv += d[0];
    if ( d.rows() > 1 )
    {
      deriv += 2 * d[1] * r2;
      if ( d.rows() > 4 )
      {
        const T r4 = r2 * r2;
        deriv += 3 * d[4] * r4;
        if ( d.rows() > 7 )
        {
          const T r6 = r4 * r2;
          const T a1 = T( 1 ) / ( d[5] * r2 + d[6] * r4 + d[7] * r6 + T( 1 ) );
          const T a2 = d[5] + 2 * d[6] * r2 + 3 * d[7] * r4;
          deriv -= a2 * a1 * ( d[0] * r2 + d[1] * r4 + d[4] * r6 + T( 1 ) );
          deriv *= a1;
        }
      }
    }
  }
  return deriv;
}

/// Compute the Jacobian of the distortion at a point
template < typename T >
Eigen::Matrix< T, 2, 2 >
distortion_jacobian( const Eigen::Matrix< T, 2, 1 >& pt,
                     const Eigen::VectorXd& d )
{
  const T x2 = pt.x() * pt.x();
  const T y2 = pt.y() * pt.y();
  const T xy = pt.x() * pt.y();
  const T r2 = x2 + y2;
  const T d_scale = 2 * radial_distortion_deriv( r2, d );
  const T scale = radial_distortion_scale( r2, d );
  Eigen::Matrix< T, 2, 2 > J;

  J << d_scale * x2 + scale, d_scale * xy,
    d_scale * xy, d_scale * y2 + scale;
  // add tangential distortion jacobian
  if ( d.rows() > 3 )
  {
    const T axy = 2 * ( d[2] * pt.x() + d[3] * pt.y() );
    const T ay = 2 * d[2] * pt.y();
    const T ax = 2 * d[3] * pt.x();
    J( 0, 0 ) += ay + 3 * ax;
    J( 0, 1 ) += axy;
    J( 1, 0 ) += axy;
    J( 0, 0 ) += 3 * ay + ax;
  }
  return J;
}

} // end anonymous namespace

/// Constructor - from a calibration matrix
simple_camera_intrinsics
::simple_camera_intrinsics( const matrix_3x3d& K,
                            const vector_t& d )
  : focal_length_( K( 0, 0 ) ),
  principal_point_( K( 0, 2 ), K( 1, 2 ) ),
  aspect_ratio_( K( 0, 0 ) / K( 1, 1 ) ),
  skew_( K( 0, 1 ) ),
  dist_coeffs_( d ),
  max_distort_radius_sq_(compute_max_distort_radius_sq())
{
}

/// Map normalized image coordinates into distorted coordinates
vector_2d
simple_camera_intrinsics
::distort( const vector_2d& norm_pt ) const
{
  double scale;
  vector_2d offset;

  distortion_scale_offset( norm_pt, dist_coeffs_, scale, offset );
  return scale * norm_pt + offset;
}

/// Unnap distorted normalized coordinates into normalized coordinates
vector_2d
simple_camera_intrinsics
::undistort( const vector_2d& dist_pt ) const
{
  double scale;
  vector_2d offset, residual;
  vector_2d norm_pt = dist_pt;

  // iteratively solve for the undistorted point
  for ( unsigned int i = 0; i < 5; ++i )
  {
    distortion_scale_offset( norm_pt, dist_coeffs_, scale, offset );
    // This is a Gauss-Newton update
    // an alternative is a fixed point iteration as used by OpenCV:
    //   norm_pt = (dist_pt - offset) / scale;
    // Gauss-Newton seems to have faster convergence
    matrix_2x2d J = distortion_jacobian( norm_pt, dist_coeffs_ );
    residual = norm_pt * scale + offset - dist_pt;
    // check the maximum absolution residual to test convergence
    if ( residual.cwiseAbs().maxCoeff() < 1e-12 )
    {
      break;
    }
    norm_pt -= J.ldlt().solve( residual );
  }
  return norm_pt;
}

/// Check if a normalized image coordinate can map into image coordinates
bool
simple_camera_intrinsics
::is_map_valid(const vector_2d& norm_pt) const
{
  return norm_pt.squaredNorm() < this->max_distort_radius_sq_;
}

/// Compute the maximum distortion radius from dist_coeffs_
double
simple_camera_intrinsics
::compute_max_distort_radius_sq() const
{
  double a = 0.0;
  double b = 0.0;
  double c = 0.0;
  if (dist_coeffs_.rows() > 0)
  {
    a = dist_coeffs_[0];
    if (dist_coeffs_.rows() > 1)
    {
      b = dist_coeffs_[1];
      if (dist_coeffs_.rows() > 4)
      {
        c = dist_coeffs_[4];
        if (dist_coeffs_.rows() > 5)
        {
          // the rational polynomial case is not yet handled
          // TODO: implement something or throw a warning
        }
      }
    }
  }
  return max_distort_radius_sq(a, b, c);
}

/// Compute the maximum radius for radial distortion given coefficients
double
simple_camera_intrinsics
::max_distort_radius_sq(double a, double b, double c)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  using kwiver::vital::pi;
  // this function finds the smallest positive root of
  //   (1 + a r^2 + b r^4 + c r^6) r
  // the derivative with respect to r is
  //   1 + 3 a r^2 + 5 b r^4 + 7 c r^6
  // substituting x = r^2 we can solve this cubic polynomial
  //   1 + 3 a x + 5 b x^2 + 7 c r^3
  // taking the square root of the smallest positive x gives r

  // fold the constants from the derivative into a, b, and c
  a *= 3;
  b *= 5;
  c *= 7;

  // if all non-negative there is no root
  if (a >= 0.0 && b >= 0.0 && c >= 0.0)
  {
    return inf;
  }
  // general solution for non-zero cubic term
  if (c != 0.0)
  {
    // an array for the three possible solutions
    double solns[3] = { inf, inf, inf };
    // precompute commonly used terms (boc is "b over c")
    double boc = b / c;
    double boc2 = boc * boc;
    double t1 = (9 * a * boc - 2 * b * boc2 - 27) / c;
    double t2 = 3 * a / c - boc2;
    double discrim = t1 * t1 + 4 * t2 * t2 * t2;
    if (discrim > 0.0)
    {
      discrim = std::cbrt((std::sqrt(discrim) + t1) / 2.0);
      solns[0] = (discrim - (t2 / discrim) - boc) / 3;
    }
    else
    {
      double theta = (std::atan2(std::sqrt(-discrim), t1)) / 3;
      constexpr double twothirdpi = 2.0 * pi / 3.0;
      // by construction, if discrim < 0 then t2 < 0, so the sqrt is safe
      double t3 = 2 * std::sqrt(-t2);
      solns[0] = (t3 * std::cos(theta) - boc) / 3;
      solns[1] = (t3 * std::cos(theta + twothirdpi) - boc) / 3;
      solns[2] = (t3 * std::cos(theta - twothirdpi) - boc) / 3;
    }
    // find the minimum positive solution
    double min_soln = inf;
    for (auto const& s : solns)
    {
      if (s > 0.0 && s < min_soln)
      {
        min_soln = s;
      }
    }
    return min_soln;
  }
  // simplified solution for the quadratic case
  else if (b != 0.0)
  {
    double discrim = a * a - 4 * b;
    // if less than zero both solutions are not real
    if (discrim >= 0.0)
    {
      // we only need the smaller of the two possible solutions
      // solutions must be either both positive or both negative
      discrim = std::sqrt(discrim) - a;
      // if less than zero both solutions are negative
      if (discrim > 0.0)
      {
        return 2.0 / discrim;
      }
    }
  }
  // simple linear case for b = c = 0
  else if (a < 0.0)
  {
    return 1.0 / -a;
  }
  return inf;
}

/// output stream operator for a base class camera_intrinsics
std::ostream&
operator<<( std::ostream& s, const camera_intrinsics& k )
{
  using std::setprecision;
  std::vector<double> d = k.dist_coeffs();
  // if no distortion coefficients, create a zero entry as a place holder
  if ( d.empty() )
  {
    d.push_back(0.0);
  }
  s << setprecision( 12 ) << k.as_matrix() << "\n\n";
  for(unsigned i=0; i<d.size(); ++i)
  {
    s << setprecision( 12 ) << d[i] << " ";
  }
  s << "\n";

  return s;
}

/// input stream operator for a camera intrinsics
std::istream&
operator>>( std::istream& s, simple_camera_intrinsics& k )
{
  matrix_3x3d K;
  Eigen::VectorXd d;

  s >> K >> d;
  // a single 0 in d is used as a place holder,
  // if a single 0 was loaded then clear d
  if ( ( d.rows() == 1 ) && ( d[0] == 0.0 ) )
  {
    d.resize( 0 );
  }
  k = simple_camera_intrinsics( K, d );
  return s;
}

} } // end namespace
