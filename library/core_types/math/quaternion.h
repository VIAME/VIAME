/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Unit quaternions, for `rotation_`
///
/// The member names are Eigen's -- `w()`, `x()`, `y()`, `z()`, `coeffs()`,
/// `toRotationMatrix()`, `inverse()`, `slerp()`, `angularDistance()`,
/// `setFromTwoVectors()` -- so that `core_types/rotation` changes its include
/// and nothing else.
///
/// `coeffs()` is (x, y, z, w), as Eigen's is: the scalar last. The
/// constructor takes (w, x, y, z), as Eigen's does: the scalar first. That
/// disagreement is Eigen's, kept because both spellings appear in the code
/// this replaces and in the files it is read against.

#ifndef VIAME_CORE_TYPES_MATH_QUATERNION_H_
#define VIAME_CORE_TYPES_MATH_QUATERNION_H_

#include "matrix.h"
#include "vector.h"

#include <cmath>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
template < typename T >
class quaternion_
{
public:
  using Scalar = T;

  /// The identity rotation.
  quaternion_() : c_( T( 0 ), T( 0 ), T( 0 ), T( 1 ) ) {}

  /// Scalar first, as Eigen's constructor is.
  quaternion_( T w, T x, T y, T z ) : c_( x, y, z, w ) {}

  /// From the (x, y, z, w) coefficient vector.
  explicit quaternion_( vector_< 4, T > const& coeffs ) : c_( coeffs ) {}

  template < typename U >
  explicit quaternion_( quaternion_< U > const& other )
    : c_( static_cast< T >( other.x() ), static_cast< T >( other.y() ),
          static_cast< T >( other.z() ), static_cast< T >( other.w() ) ) {}

  /// From a rotation matrix, by Shepperd's method: take the square root of
  /// whichever of the four diagonal combinations is largest, so that the
  /// division is never by something small.
  explicit quaternion_( matrix_< 3, 3, T > const& m ) { from_matrix( m ); }

  /// From an axis-angle pair; the axis need not be normalised.
  static quaternion_ from_axis_angle( T angle, vector_< 3, T > const& axis )
  {
    vector_< 3, T > const a = axis.normalized();
    T const half = angle / T( 2 );
    T const s = std::sin( half );
    return quaternion_( std::cos( half ), a[ 0 ] * s, a[ 1 ] * s, a[ 2 ] * s );
  }

  T x() const { return c_[ 0 ]; }
  T y() const { return c_[ 1 ]; }
  T z() const { return c_[ 2 ]; }
  T w() const { return c_[ 3 ]; }

  T& x() { return c_[ 0 ]; }
  T& y() { return c_[ 1 ]; }
  T& z() { return c_[ 2 ]; }
  T& w() { return c_[ 3 ]; }

  /// (x, y, z, w).
  vector_< 4, T > const& coeffs() const { return c_; }
  vector_< 4, T >& coeffs() { return c_; }

  /// The vector part.
  vector_< 3, T > vec() const { return vector_< 3, T >( c_[ 0 ], c_[ 1 ], c_[ 2 ] ); }

  void setIdentity() { c_ = vector_< 4, T >( T( 0 ), T( 0 ), T( 0 ), T( 1 ) ); }

  T norm() const { return c_.norm(); }
  T squaredNorm() const { return c_.squaredNorm(); }

  void normalize()
  {
    T const n = c_.norm();
    if( n > T( 0 ) ) { c_ /= n; }
  }

  quaternion_ normalized() const
  {
    quaternion_ out( *this );
    out.normalize();
    return out;
  }

  quaternion_ conjugate() const
  { return quaternion_( c_[ 3 ], -c_[ 0 ], -c_[ 1 ], -c_[ 2 ] ); }

  /// For a unit quaternion this is the conjugate; the general case divides
  /// by the squared norm, as Eigen's does.
  quaternion_ inverse() const
  {
    T const n2 = squaredNorm();
    quaternion_ const c = conjugate();
    if( n2 == T( 0 ) || n2 == T( 1 ) ) { return c; }
    return quaternion_( c.w() / n2, c.x() / n2, c.y() / n2, c.z() / n2 );
  }

  /// Composition: the rotation \p rhs followed by this one.
  quaternion_ operator*( quaternion_ const& rhs ) const
  {
    T const aw = w(), ax = x(), ay = y(), az = z();
    T const bw = rhs.w(), bx = rhs.x(), by = rhs.y(), bz = rhs.z();
    return quaternion_( aw * bw - ax * bx - ay * by - az * bz,
                        aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw );
  }

  /// Rotating a vector, by the two-cross-product form rather than by building
  /// the matrix.
  vector_< 3, T > operator*( vector_< 3, T > const& v ) const
  {
    vector_< 3, T > const u = vec();
    vector_< 3, T > const t = u.cross( v ) * T( 2 );
    return v + t * w() + u.cross( t );
  }

  matrix_< 3, 3, T > toRotationMatrix() const
  {
    T const xx = x() * x(), yy = y() * y(), zz = z() * z();
    T const xy = x() * y(), xz = x() * z(), yz = y() * z();
    T const wx = w() * x(), wy = w() * y(), wz = w() * z();

    matrix_< 3, 3, T > m;
    m( 0, 0 ) = T( 1 ) - T( 2 ) * ( yy + zz );
    m( 0, 1 ) = T( 2 ) * ( xy - wz );
    m( 0, 2 ) = T( 2 ) * ( xz + wy );
    m( 1, 0 ) = T( 2 ) * ( xy + wz );
    m( 1, 1 ) = T( 1 ) - T( 2 ) * ( xx + zz );
    m( 1, 2 ) = T( 2 ) * ( yz - wx );
    m( 2, 0 ) = T( 2 ) * ( xz - wy );
    m( 2, 1 ) = T( 2 ) * ( yz + wx );
    m( 2, 2 ) = T( 1 ) - T( 2 ) * ( xx + yy );
    return m;
  }

  T dot( quaternion_ const& o ) const { return c_.dot( o.c_ ); }

  /// The angle between two rotations, in radians, never more than pi.
  T angularDistance( quaternion_ const& o ) const
  {
    T d = std::abs( dot( o ) );
    if( d > T( 1 ) ) { d = T( 1 ); }
    return T( 2 ) * std::acos( d );
  }

  /// Spherical linear interpolation, taking the short way round.
  quaternion_ slerp( T t, quaternion_ const& o ) const
  {
    T d = dot( o );
    quaternion_ b = o;
    if( d < T( 0 ) )
    {
      d = -d;
      b.c_ = -b.c_;
    }

    // Close enough that the sine is unstable: interpolate straight and
    // renormalise, which is what Eigen does too.
    if( d > T( 1 ) - std::numeric_limits< T >::epsilon() * T( 10 ) )
    {
      quaternion_ out( *this );
      out.c_ = c_ * ( T( 1 ) - t ) + b.c_ * t;
      out.normalize();
      return out;
    }

    T const theta = std::acos( d );
    T const s = std::sin( theta );
    T const wa = std::sin( ( T( 1 ) - t ) * theta ) / s;
    T const wb = std::sin( t * theta ) / s;

    quaternion_ out;
    out.c_ = c_ * wa + b.c_ * wb;
    return out;
  }

  /// The shortest rotation taking \p from to \p to.
  void setFromTwoVectors( vector_< 3, T > const& from,
                          vector_< 3, T > const& to )
  {
    vector_< 3, T > const a = from.normalized();
    vector_< 3, T > const b = to.normalized();
    T const d = a.dot( b );

    if( d >= T( 1 ) - std::numeric_limits< T >::epsilon() )
    {
      setIdentity();
      return;
    }

    if( d <= T( -1 ) + std::numeric_limits< T >::epsilon() )
    {
      // Antiparallel: any perpendicular axis will do; take the one that is
      // furthest from parallel with a.
      vector_< 3, T > axis =
        std::abs( a[ 0 ] ) < std::abs( a[ 2 ] )
          ? vector_< 3, T >( T( 0 ), -a[ 2 ], a[ 1 ] )
          : vector_< 3, T >( -a[ 1 ], a[ 0 ], T( 0 ) );
      axis = axis.normalized();
      c_ = vector_< 4, T >( axis[ 0 ], axis[ 1 ], axis[ 2 ], T( 0 ) );
      return;
    }

    vector_< 3, T > const axis = a.cross( b );
    T const s = std::sqrt( ( T( 1 ) + d ) * T( 2 ) );
    c_ = vector_< 4, T >( axis[ 0 ] / s, axis[ 1 ] / s, axis[ 2 ] / s,
                          s / T( 2 ) );
    normalize();
  }

  static quaternion_ Identity() { return quaternion_(); }

private:
  void from_matrix( matrix_< 3, 3, T > const& m )
  {
    T const trace = m( 0, 0 ) + m( 1, 1 ) + m( 2, 2 );

    if( trace > T( 0 ) )
    {
      T const s = std::sqrt( trace + T( 1 ) ) * T( 2 );
      c_ = vector_< 4, T >( ( m( 2, 1 ) - m( 1, 2 ) ) / s,
                            ( m( 0, 2 ) - m( 2, 0 ) ) / s,
                            ( m( 1, 0 ) - m( 0, 1 ) ) / s,
                            s / T( 4 ) );
    }
    else if( m( 0, 0 ) > m( 1, 1 ) && m( 0, 0 ) > m( 2, 2 ) )
    {
      T const s = std::sqrt( T( 1 ) + m( 0, 0 ) - m( 1, 1 ) - m( 2, 2 ) ) * T( 2 );
      c_ = vector_< 4, T >( s / T( 4 ),
                            ( m( 0, 1 ) + m( 1, 0 ) ) / s,
                            ( m( 0, 2 ) + m( 2, 0 ) ) / s,
                            ( m( 2, 1 ) - m( 1, 2 ) ) / s );
    }
    else if( m( 1, 1 ) > m( 2, 2 ) )
    {
      T const s = std::sqrt( T( 1 ) + m( 1, 1 ) - m( 0, 0 ) - m( 2, 2 ) ) * T( 2 );
      c_ = vector_< 4, T >( ( m( 0, 1 ) + m( 1, 0 ) ) / s,
                            s / T( 4 ),
                            ( m( 1, 2 ) + m( 2, 1 ) ) / s,
                            ( m( 0, 2 ) - m( 2, 0 ) ) / s );
    }
    else
    {
      T const s = std::sqrt( T( 1 ) + m( 2, 2 ) - m( 0, 0 ) - m( 1, 1 ) ) * T( 2 );
      c_ = vector_< 4, T >( ( m( 0, 2 ) + m( 2, 0 ) ) / s,
                            ( m( 1, 2 ) + m( 2, 1 ) ) / s,
                            s / T( 4 ),
                            ( m( 1, 0 ) - m( 0, 1 ) ) / s );
    }
  }

  /// (x, y, z, w)
  vector_< 4, T > c_;
};

/// \cond DoxygenSuppress
typedef quaternion_< double > quaternion_d;
typedef quaternion_< float >  quaternion_f;
/// \endcond

} // namespace vital

} // namespace kwiver

#endif
