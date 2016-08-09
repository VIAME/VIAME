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
 * \brief Implementation of \link kwiver::vital::camera camera \endlink class
 */

#include <vital/types/camera.h>
#include <vital/io/eigen_io.h>
#include <vital/types/matrix.h>
#include <Eigen/Geometry>

#include <iomanip>

namespace kwiver {
namespace vital {

camera
::camera()
  : m_logger( kwiver::vital::get_logger( "vital.camera" ) )
{
}

/// Convert to a 3x4 homogeneous projection matrix
matrix_3x4d
camera
::as_matrix() const
{
  matrix_3x4d P;
  matrix_3x3d R( this->rotation() );
  matrix_3x3d K( this->intrinsics()->as_matrix() );
  vector_3d t( this->translation() );
  P.block< 3, 3 > ( 0, 0 ) = R;
  P.block< 3, 1 > ( 0, 3 ) = t;
  return K * P;
}


/// Project a 3D point into a 2D image point
vector_2d
camera
::project( const vector_3d& pt ) const
{
  return this->intrinsics()->map( this->rotation() * ( pt - this->center() ));
}


/// Compute the distance of the 3D point to the image plane
double
camera
::depth(const vector_3d& pt) const
{
  return (this->rotation() * (pt - this->center())).z();
}


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
void
simple_camera
::look_at( const vector_3d& stare_point,
           const vector_3d& up_direction )
{
  // a unit vector in the up direction
  const vector_3d up = up_direction.normalized();
  // a unit vector in the look direction (camera Z-axis)
  const vector_3d z = ( stare_point - get_center() ).normalized();

  // the X-axis of the camera is perpendicular to up and z
  vector_3d x = -up.cross( z );
  double x_mag = x.norm();

  // if the cross product magnitude is small then the up and z vectors are
  // nearly parallel and the up direction is poorly defined.
  if ( x_mag < 1e-4 )
  {
    LOG_WARN( m_logger,  "camera_::look_at up_direction nearly parallel with the look direction" );
  }

  x /= x_mag;
  vector_3d y = z.cross( x ).normalized();

  matrix_3x3d R;
  R << x.x(), x.y(), x.z(),
    y.x(), y.y(), y.z(),
    z.x(), z.y(), z.z();

  this->set_rotation( rotation_d ( R ) );
}


/// input stream operator for a camera intrinsics
std::istream&
operator>>( std::istream& s, simple_camera& k )
{
  matrix_3x3d K, R;
  vector_3d t;
  s >> K >> R >> t;

  double dVal;
  std::vector<double> dValues;

  while (s >> dVal)
  {
    dValues.push_back(dVal);
  }

  Eigen::VectorXd d(dValues.size());

  for (size_t i = 0; i < dValues.size(); ++i)
  {
    d(i) =  dValues[i];
  }

  // a single 0 in d is used as a place holder,
  // if a single 0 was loaded then clear d
  if ( ( d.rows() == 1 ) && ( d[0] ==  0.0 ) )
  {
    d.resize( 0 );
  }
  k.set_intrinsics( camera_intrinsics_sptr(new simple_camera_intrinsics( K, d ) ) );
  k.set_rotation( rotation_d ( R ) );
  k.set_translation( t );
  return s;
}


} } // end namespace
