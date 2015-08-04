/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Implementation of \link vital::camera_ camera_<T> \endlink class
 *        for \c T = { \c float, \c double }
 */

#include <vital/types/camera.h>
#include <vital/io/eigen_io.h>
#include <vital/types/matrix.h>
#include <Eigen/Geometry>

#include <iomanip>

namespace kwiver {
namespace vital {

/// output stream operator for a base class camera
std::ostream&
operator<<( std::ostream& s, const camera& c )
{
  using std::setprecision;
  std::vector<double> dc = c.intrinsics()->dist_coeffs();
  Eigen::VectorXd d = Eigen::VectorXd::Map(dc.data(), dc.size());
  // if no distortion coefficients, create a zero entry as a place holder
  if ( d.rows() == 0 )
  {
    d.resize( 1 );
    d[0] = 0.0;
  }
  s << setprecision( 12 ) << c.intrinsics()->as_matrix() << "\n\n"
    << setprecision( 12 ) << matrix_3x3d( c.rotation() ) << "\n\n"
    << setprecision( 12 ) << c.translation().transpose() << "\n\n"
    << setprecision( 12 ) << d.transpose() << "\n";
  return s;
}


/// Rotate the camera about its center such that it looks at the given point.
template < typename T >
void
camera_< T >
::look_at( const Eigen::Matrix< T, 3, 1 >& stare_point,
             const Eigen::Matrix< T, 3, 1 >& up_direction )
{
  // a unit vector in the up direction
  const Eigen::Matrix< T, 3, 1 > up = up_direction.normalized();
  // a unit vector in the look direction (camera Z-axis)
  const Eigen::Matrix< T, 3, 1 > z = ( stare_point - get_center() ).normalized();

  // the X-axis of the camera is perpendicular to up and z
  Eigen::Matrix< T, 3, 1 > x = -up.cross( z );
  T x_mag = x.norm();

  // if the cross product magnitude is small then the up and z vectors are
  // nearly parallel and the up direction is poorly defined.
  if ( x_mag < 1e-4 )
  {
    std::cerr << "WARNING: camera_::look_at up_direction is "
              << "nearly parallel with the look direction" << std::endl;
  }

  x /= x_mag;
  Eigen::Matrix< T, 3, 1 > y = z.cross( x ).normalized();

  Eigen::Matrix< T, 3, 3 > R;
  R << x.x(), x.y(), x.z(),
    y.x(), y.y(), y.z(),
    z.x(), z.y(), z.z();

  this->set_rotation( rotation_< T > ( R ) );
}


/// Convert to a 3x4 homogeneous projection matrix
template < typename T >
camera_< T >
::operator Eigen::Matrix< T, 3, 4 > () const
{
  Eigen::Matrix< T, 3, 4 > P;
  Eigen::Matrix< T, 3, 3 > R( this->get_rotation() );
  Eigen::Matrix< T, 3, 3 > K( this->get_intrinsics()->as_matrix().template cast<T>() );
  Eigen::Matrix< T, 3, 1 > t( this->get_translation() );
  P.template block< 3, 3 > ( 0, 0 ) = R;
  P.template block< 3, 1 > ( 0, 3 ) = t;
  return K * P;
}


/// Project a 3D point into a 2D image point
template < typename T >
Eigen::Matrix< T, 2, 1 >
camera_< T >
::project( const Eigen::Matrix< T, 3, 1 >& pt ) const
{
  vector_3d p3 = (this->orientation_ * ( pt - this->center_ )).template cast<double>();
  return this->intrinsics_->map( p3 ).template cast<T>();
}


/// Compute the distance of the 3D point to the image plane
template <typename T>
T
camera_<T>
::depth(const Eigen::Matrix<T, 3, 1>& pt) const
{
  return (this->orientation_ * (pt - this->center_)).z();
}


template < typename T >
std::ostream&
operator<<( std::ostream& s, const camera_< T >& c )
{
  using std::setprecision;
  std::vector<double> dc = c.get_intrinsics()->dist_coeffs();
  Eigen::VectorXd d = Eigen::VectorXd::Map(dc.data(), dc.size());
  // if no distortion coefficients, create a zero entry as a place holder
  if ( d.rows() == 0 )
  {
    d.resize( 1 );
    d[0] = T( 0 );
  }
  s << setprecision( 12 ) << c.get_intrinsics()->as_matrix() << "\n\n"
    << setprecision( 12 ) << Eigen::Matrix< T, 3, 3 > ( c.get_rotation() ) << "\n\n"
    << setprecision( 12 ) << c.get_translation().transpose() << "\n\n"
    << setprecision( 12 ) << d.transpose() << "\n";
  return s;
}


/// input stream operator for a camera intrinsics
template < typename T >
std::istream&
operator>>( std::istream& s, camera_< T >& k )
{
  matrix_3x3d K;
  Eigen::Matrix< T, 3, 3 > R;
  Eigen::Matrix< T, 3, 1 > t;
  Eigen::VectorXd d;

  s >> K >> R >> t >> d;
  // a single 0 in d is used as a place holder,
  // if a single 0 was loaded then clear d
  if ( ( d.rows() == 1 ) && ( d[0] ==  0.0 ) )
  {
    d.resize( 0 );
  }
  k.set_intrinsics( camera_intrinsics_sptr(new simple_camera_intrinsics( K, d ) ) );
  k.set_rotation( rotation_< T > ( R ) );
  k.set_translation( t );
  return s;
}


/// \cond DoxygenSuppress
#define INSTANTIATE_CAMERA( T )                                                      \
  template class VITAL_EXPORT camera_< T >;                                      \
  template VITAL_EXPORT std::ostream&                                            \
  operator<<( std::ostream& s, const camera_< T >& c );                              \
  template VITAL_EXPORT std::istream&                                            \
  operator>>( std::istream& s, camera_< T >& c );

INSTANTIATE_CAMERA( double );
INSTANTIATE_CAMERA( float );

#undef INSTANTIATE_CAMERA
/// \endcond

} } // end namespace
