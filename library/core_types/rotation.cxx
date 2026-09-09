// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of \link kwiver::vital::rotation_ rotation_<T>
/// \endlink
///        for \c T = { \c float, \c double }

#include "rotation.h"

#include <viame/algorithm_framework/io/eigen_io.h>
#include <viame/core_types/math_constants.h>

#include <cmath>
#include <limits>

namespace kwiver {

namespace vital {

/// Constructor - from a Rodrigues vector
template < typename T >
rotation_< T >

::rotation_( const Eigen::Matrix< T, 3, 1 >& rvec )
{
  T mag = rvec.norm();

  if( mag == T( 0 ) )
  {
    // identity rotation is a special case
    q_.setIdentity();
  }
  else
  {
    q_ = Eigen::Quaternion< T >( Eigen::AngleAxis< T >( mag, rvec / mag ) );
  }
}

/// Constructor - from rotation angle (radians) and axis
template < typename T >
rotation_< T >

::rotation_( T angle, const Eigen::Matrix< T, 3, 1 >& axis )
  : q_( Eigen::Quaternion< T >(
      Eigen::AngleAxis< T >(
        angle,
        axis.normalized() ) ) )
{}

/// Constructor - from yaw, pitch, and roll (radians)
template < typename T >
rotation_< T >

::rotation_( const T& yaw, const T& pitch, const T& roll )
{
  // See
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Euler_angles_to_quaternion_conversion
  T const half_x = static_cast< T >( 0.5 ) * roll;
  T const half_y = static_cast< T >( 0.5 ) * pitch;
  T const half_z = static_cast< T >( 0.5 ) * yaw;
  T const sin_x = std::sin( half_x );
  T const cos_x = std::cos( half_x );
  T const sin_y = std::sin( half_y );
  T const cos_y = std::cos( half_y );
  T const sin_z = std::sin( half_z );
  T const cos_z = std::cos( half_z );
  *this = Eigen::Quaternion< T >{
    cos_x* cos_y* cos_z + sin_x * sin_y * sin_z,
    sin_x* cos_y* cos_z - cos_x * sin_y * sin_z,
    cos_x* sin_y* cos_z + sin_x * cos_y * sin_z,
    cos_x* cos_y* sin_z - sin_x * sin_y * cos_z };
}

/// Constructor - from a matrix
///
/// requires orthonormal matrix with +1 determinant
template < typename T >
rotation_< T >

::rotation_( const Eigen::Matrix< T, 3, 3 >& rot )
{
  q_ = Eigen::Quaternion< T >( rot );
}

/// Convert to a 3x3 matrix
template < typename T >
Eigen::Matrix< T, 3, 3 >
rotation_< T >
::matrix() const
{
  return q_.toRotationMatrix();
}

/// Returns the axis of rotation
template < typename T >
Eigen::Matrix< T, 3, 1 >
rotation_< T >
::axis() const
{
  Eigen::Matrix< T, 3, 1 > dir( q_.x(), q_.y(), q_.z() );
  T mag = dir.norm();

  if( mag == T( 0 ) )
  {
    return Eigen::Matrix< T, 3, 1 >( 0, 0, 1 );
  }
  return dir / mag;
}

/// Returns the angle of the rotation in radians about the axis
template < typename T >
T
rotation_< T >
::angle() const
{
  static const T _pi = static_cast< T >( pi );
  static const T _two_pi = static_cast< T >( 2.0 * pi );

  const double i = Eigen::Matrix< T, 3, 1 >( q_.x(), q_.y(), q_.z() ).norm();
  const double r = q_.w();
  T a = static_cast< T >( 2.0 * std::atan2( i, r ) );

  // make sure computed angle lies within a sensible range,
  // i.e. -pi/2 < a < pi/2
  if( a >= _pi )
  {
    a -= _two_pi;
  }
  if( a <= -_pi )
  {
    a += _two_pi;
  }
  return a;
}

/// Return the rotation as a Rodrigues vector
template < typename T >
Eigen::Matrix< T, 3, 1 >
rotation_< T >
::rodrigues() const
{
  T angle = this->angle();

  if( angle == 0.0 )
  {
    return Eigen::Matrix< T, 3, 1 >( 0, 0, 0 );
  }
  return this->axis() * angle;
}

/// Convert to yaw, pitch, and roll (radians)
template < typename T >
void
rotation_< T >
::get_yaw_pitch_roll( T& yaw, T& pitch, T& roll ) const
{
  // See
  // https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles#Quaternion_to_Euler_angles_conversion
  constexpr auto _1 = static_cast< T >( 1.0 );
  constexpr auto _2 = static_cast< T >( 2.0 );
  roll = std::atan2(
    _2 * ( q_.w() * q_.x() + q_.y() * q_.z() ),
    _1 - _2 * ( q_.x() * q_.x() + q_.y() * q_.y() ) );
  pitch = std::asin( _2 * ( q_.w() * q_.y() - q_.x() * q_.z() ) );
  yaw = std::atan2(
    _2 * ( q_.w() * q_.z() + q_.x() * q_.y() ),
    _1 - _2 * ( q_.y() * q_.y() + q_.z() * q_.z() ) );
}

/// Compose two rotations
template < typename T >
rotation_< T >
rotation_< T >
::operator*( const rotation_< T >& rhs ) const
{
  return q_ * rhs.q_;
}

/// Rotate a vector
///
/// \note for a large number of vectors, it is more efficient to
/// create a rotation matrix and use matrix multiplcation
template < typename T >
Eigen::Matrix< T, 3, 1 >
rotation_< T >
::operator*( const Eigen::Matrix< T, 3, 1 >& rhs ) const
{
  return q_ * rhs;
}

/// output stream operator for a rotation
template < typename T >
std::ostream&
operator<<( std::ostream& s, const rotation_< T >& r )
{
  s << r.quaternion().coeffs();
  return s;
}

/// input stream operator for a rotation
template < typename T >
std::istream&
operator>>( std::istream& s, rotation_< T >& r )
{
  Eigen::Matrix< T, 4, 1 > q;

  s >> q;
  r = rotation_< T >( q );
  return s;
}

/// Generate a rotation vector that, when applied to A N times, produces B.
template < typename T >
rotation_< T >
interpolate_rotation( rotation_< T > const& A, rotation_< T > const& B, T f )
{
  // rotation from A -> B
  rotation_< T > C = A.inverse() * B;
  // Reduce the angle of rotation by the fraction provided
  return A * rotation_< T >( C.angle() * f, C.axis() );
}

/// Generate N evenly interpolated rotations inbetween \c A and \c B.
template < typename T >
void
interpolated_rotations(
  rotation_< T > const& A, rotation_< T > const& B,
  size_t n, std::vector< rotation_< T > >& interp_rots )
{
  interp_rots.reserve( interp_rots.capacity() + n );

  size_t denom = n + 1;
  for( size_t i = 1; i < denom; ++i )
  {
    interp_rots.push_back(
      interpolate_rotation< T >(
        A, B,
        static_cast< T >( i ) / denom ) );
  }
}

template < typename T >
rotation_< T >
ned_to_enu( rotation_< T > const& r )
{
  auto axis = Eigen::Matrix< T, 3, 1 >{ 1, 1, 0 };
  auto angle = T{ -pi };
  auto adjustment = rotation_< T >{ angle, axis };
  return adjustment * r;
}

template < typename T >
rotation_< T >
enu_to_ned( rotation_< T > const& r )
{
  auto axis = Eigen::Matrix< T, 3, 1 >{ 1, 1, 0 };
  auto angle = T{ pi };
  auto adjustment = rotation_< T >{ angle, axis };
  return adjustment * r;
}

template < typename T >
rotation_< T >
sensor_to_camera( rotation_< T > const& r )
{
  static const rotation_< T > conversion{
    Eigen::Quaternion< T >{ 0.5, 0.5, 0.5, 0.5 } };
  return ( r * conversion ).inverse();
}

template < typename T >
rotation_< T >
camera_to_sensor( rotation_< T > const& r )
{
  static const rotation_< T > conversion{
    Eigen::Quaternion< T >{ 0.5, -0.5, -0.5, -0.5 } };
  return r.inverse() * conversion;
}

/// \cond DoxygenSuppress
#define INSTANTIATE_ROTATION( T )                                \
template class VITAL_TYPES_EXPORT rotation_< T >;                \
template VITAL_TYPES_EXPORT std::ostream&                        \
operator<<( std::ostream& s, const rotation_< T >& r );          \
template VITAL_TYPES_EXPORT std::istream&                        \
operator>>( std::istream& s, rotation_< T >& r );                \
template VITAL_TYPES_EXPORT rotation_< T > interpolate_rotation( \
  rotation_< T > const& A, rotation_< T > const& B, T f );       \
template VITAL_TYPES_EXPORT void                                 \
interpolated_rotations(                                          \
  rotation_< T > const& A, rotation_< T > const& B,              \
  size_t n, std::vector< rotation_< T > >& interp_rots );        \
template VITAL_TYPES_EXPORT rotation_< T > ned_to_enu(           \
  rotation_< T > const& r );                                     \
template VITAL_TYPES_EXPORT rotation_< T > enu_to_ned(           \
  rotation_< T > const& r );                                     \
template VITAL_TYPES_EXPORT rotation_< T > sensor_to_camera(     \
  rotation_< T > const& r );                                     \
template VITAL_TYPES_EXPORT rotation_< T > camera_to_sensor(     \
  rotation_< T > const& r );

INSTANTIATE_ROTATION( double );
INSTANTIATE_ROTATION( float );

#undef INSTANTIATE_ROTATION
/// \endcond

} // namespace vital

}   // end namespace vital
