/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Fixed-size column vectors
///
/// Phase 6 replaces Eigen with this. The spelling is Eigen's, because 847
/// lines of VIAME use it and the point of the replacement is that they do not
/// change: `v.dot( w )`, `v.norm()`, `v.normalized()`, `v.cross( w )`,
/// `vector_3d::Zero()`, `v( i )`, `v[ i ]`, and the operators.
///
/// There are no expression templates. Every operation returns a value.
/// `design/lite-eigen-uses.txt` counts what VIAME asks for and the largest
/// vector in it is four elements, so the copies are four doubles and the
/// aliasing questions an expression template exists to answer do not arise.

#ifndef VIAME_CORE_TYPES_MATH_VECTOR_H_
#define VIAME_CORE_TYPES_MATH_VECTOR_H_

#include <cmath>
#include <cstddef>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <type_traits>

namespace kwiver {

namespace vital {

template < unsigned R, unsigned C, typename T > class matrix_;

// ----------------------------------------------------------------------------
/// The tolerance below which a value counts as zero, by element type.
///
/// Eigen calls this `NumTraits< T >::dummy_precision()` and the values are
/// its: 1e-12 for a double, 1e-5 for a float. Kept because the two callers
/// compare a homography's bottom-right entry against it, and changing the
/// threshold would change which homographies are called affine.
template < typename T > constexpr T math_dummy_precision();

template <> constexpr double math_dummy_precision< double >() { return 1e-12; }
template <> constexpr float math_dummy_precision< float >() { return 1e-5f; }

// ----------------------------------------------------------------------------
/// A column vector of \p N elements of \p T.
template < unsigned N, typename T >
class vector_
{
public:
  static_assert( N > 0, "a vector has at least one element" );

  using value_type = T;
  using Scalar = T;                    ///< Eigen's spelling, which callers use
  static constexpr unsigned size_value = N;

  constexpr vector_() : d_{} {}

  /// Every element the same, which is what `vector_3d::Constant( x )` means.
  explicit constexpr vector_( T value ) : d_{}
  {
    for( unsigned i = 0; i < N; ++i ) { d_[ i ] = value; }
  }

  constexpr vector_( T x, T y ) : d_{ x, y }
  { static_assert( N == 2, "two elements is a 2-vector" ); }

  constexpr vector_( T x, T y, T z ) : d_{ x, y, z }
  { static_assert( N == 3, "three elements is a 3-vector" ); }

  constexpr vector_( T x, T y, T z, T w ) : d_{ x, y, z, w }
  { static_assert( N == 4, "four elements is a 4-vector" ); }

  /// Widening or narrowing the element type, which is Eigen's `cast< T >()`.
  template < typename U >
  explicit vector_( vector_< N, U > const& other )
  {
    for( unsigned i = 0; i < N; ++i )
    {
      d_[ i ] = static_cast< T >( other[ i ] );
    }
  }

  template < typename U >
  vector_< N, U > cast() const
  {
    vector_< N, U > out;
    for( unsigned i = 0; i < N; ++i )
    {
      out[ i ] = static_cast< U >( d_[ i ] );
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Element access
  //
  // A vector answers to both `v( i )` and `v[ i ]`, as Eigen's does, and to
  // `v( i, 0 )`, which the code that treats a vector as an Nx1 matrix uses.
  T& operator()( unsigned i )             { return d_[ i ]; }
  T const& operator()( unsigned i ) const { return d_[ i ]; }
  T& operator()( unsigned i, unsigned )   { return d_[ i ]; }
  T const& operator()( unsigned i, unsigned ) const { return d_[ i ]; }
  T& operator[]( unsigned i )             { return d_[ i ]; }
  T const& operator[]( unsigned i ) const { return d_[ i ]; }

  T& x()             { return d_[ 0 ]; }
  T const& x() const { return d_[ 0 ]; }
  T& y()             { static_assert( N >= 2, "" ); return d_[ 1 ]; }
  T const& y() const { static_assert( N >= 2, "" ); return d_[ 1 ]; }
  T& z()             { static_assert( N >= 3, "" ); return d_[ 2 ]; }
  T const& z() const { static_assert( N >= 3, "" ); return d_[ 2 ]; }
  T& w()             { static_assert( N >= 4, "" ); return d_[ 3 ]; }
  T const& w() const { static_assert( N >= 4, "" ); return d_[ 3 ]; }

  T* data()             { return d_; }
  T const* data() const { return d_; }

  static constexpr unsigned size()  { return N; }
  static constexpr unsigned rows()  { return N; }
  static constexpr unsigned cols()  { return 1; }

  // --------------------------------------------------------------------------
  // The named constructors
  static vector_ Zero()     { return vector_(); }
  static vector_ Ones()     { return vector_( T( 1 ) ); }
  static vector_ Constant( T value ) { return vector_( value ); }

  static vector_ Unit( unsigned i )
  {
    vector_ out;
    out[ i ] = T( 1 );
    return out;
  }

  static vector_ UnitX() { return Unit( 0 ); }
  static vector_ UnitY() { return Unit( 1 ); }
  static vector_ UnitZ() { return Unit( 2 ); }

  void setZero() { for( unsigned i = 0; i < N; ++i ) { d_[ i ] = T( 0 ); } }
  void setConstant( T value )
  { for( unsigned i = 0; i < N; ++i ) { d_[ i ] = value; } }

  // --------------------------------------------------------------------------
  // Arithmetic
  vector_ operator-() const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i ) { out[ i ] = -d_[ i ]; }
    return out;
  }

  vector_& operator+=( vector_ const& o )
  { for( unsigned i = 0; i < N; ++i ) { d_[ i ] += o[ i ]; } return *this; }

  vector_& operator-=( vector_ const& o )
  { for( unsigned i = 0; i < N; ++i ) { d_[ i ] -= o[ i ]; } return *this; }

  vector_& operator*=( T s )
  { for( unsigned i = 0; i < N; ++i ) { d_[ i ] *= s; } return *this; }

  vector_& operator/=( T s )
  { for( unsigned i = 0; i < N; ++i ) { d_[ i ] /= s; } return *this; }

  // --------------------------------------------------------------------------
  // Products and norms
  T dot( vector_ const& o ) const
  {
    T sum = T( 0 );
    for( unsigned i = 0; i < N; ++i ) { sum += d_[ i ] * o[ i ]; }
    return sum;
  }

  T squaredNorm() const { return dot( *this ); }
  T norm() const { return static_cast< T >( std::sqrt( squaredNorm() ) ); }
  T sum() const
  {
    T total = T( 0 );
    for( unsigned i = 0; i < N; ++i ) { total += d_[ i ]; }
    return total;
  }

  T maxCoeff() const
  {
    T best = d_[ 0 ];
    for( unsigned i = 1; i < N; ++i ) { if( d_[ i ] > best ) { best = d_[ i ]; } }
    return best;
  }

  T minCoeff() const
  {
    T best = d_[ 0 ];
    for( unsigned i = 1; i < N; ++i ) { if( d_[ i ] < best ) { best = d_[ i ]; } }
    return best;
  }

  T mean() const { return sum() / static_cast< T >( N ); }

  vector_ normalized() const
  {
    T const n = norm();
    return n > T( 0 ) ? *this / n : *this;
  }

  void normalize() { *this = normalized(); }

  /// The cross product, for three elements only.
  vector_ cross( vector_ const& o ) const
  {
    static_assert( N == 3, "a cross product is three-dimensional" );
    return vector_( d_[ 1 ] * o[ 2 ] - d_[ 2 ] * o[ 1 ],
                    d_[ 2 ] * o[ 0 ] - d_[ 0 ] * o[ 2 ],
                    d_[ 0 ] * o[ 1 ] - d_[ 1 ] * o[ 0 ] );
  }

  /// One more element, set to one: the projective lift.
  vector_< N + 1, T > homogeneous() const
  {
    vector_< N + 1, T > out;
    for( unsigned i = 0; i < N; ++i ) { out[ i ] = d_[ i ]; }
    out[ N ] = T( 1 );
    return out;
  }

  /// One fewer element, divided through by the last: the projective drop.
  vector_< N - 1, T > hnormalized() const
  {
    static_assert( N >= 2, "there has to be something left" );
    vector_< N - 1, T > out;
    for( unsigned i = 0; i + 1 < N; ++i ) { out[ i ] = d_[ i ] / d_[ N - 1 ]; }
    return out;
  }

  // --------------------------------------------------------------------------
  // Coefficient-wise operations
  //
  // Eigen spells these through `.array()`, which turns a vector into an
  // expression whose operators are element-wise. There is no expression type
  // here, so the named forms are what remain.
  vector_ cwiseProduct( vector_ const& o ) const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i ) { out[ i ] = d_[ i ] * o[ i ]; }
    return out;
  }

  vector_ cwiseQuotient( vector_ const& o ) const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i ) { out[ i ] = d_[ i ] / o[ i ]; }
    return out;
  }

  vector_ cwiseAbs() const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i )
    {
      out[ i ] = d_[ i ] < T( 0 ) ? -d_[ i ] : d_[ i ];
    }
    return out;
  }

  vector_ cwiseMin( vector_ const& o ) const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i )
    {
      out[ i ] = d_[ i ] < o[ i ] ? d_[ i ] : o[ i ];
    }
    return out;
  }

  vector_ cwiseMax( vector_ const& o ) const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i )
    {
      out[ i ] = d_[ i ] > o[ i ] ? d_[ i ] : o[ i ];
    }
    return out;
  }

  /// Element-wise, with a scalar added to each.
  vector_ cwisePlus( T s ) const
  {
    vector_ out;
    for( unsigned i = 0; i < N; ++i ) { out[ i ] = d_[ i ] + s; }
    return out;
  }

  bool allFinite() const
  {
    for( unsigned i = 0; i < N; ++i )
    {
      if( !std::isfinite( d_[ i ] ) ) { return false; }
    }
    return true;
  }

  /// Every element within \p tolerance of zero.
  bool isZero( T tolerance = std::numeric_limits< T >::epsilon() *
                             T( 100 ) ) const
  {
    for( unsigned i = 0; i < N; ++i )
    {
      T const x = d_[ i ] < T( 0 ) ? -d_[ i ] : d_[ i ];
      if( x > tolerance ) { return false; }
    }
    return true;
  }

  /// The first \p M elements, with the count known at run time; the caller
  /// must ask for no more than there are.
  vector_ head( unsigned m ) const
  {
    vector_ out;
    for( unsigned i = 0; i < m && i < N; ++i ) { out[ i ] = d_[ i ]; }
    return out;
  }

  /// The first \p M elements.
  template < unsigned M >
  vector_< M, T > head() const
  {
    static_assert( M <= N, "there are not that many" );
    vector_< M, T > out;
    for( unsigned i = 0; i < M; ++i ) { out[ i ] = d_[ i ]; }
    return out;
  }

  /// The last \p M elements.
  template < unsigned M >
  vector_< M, T > tail() const
  {
    static_assert( M <= N, "there are not that many" );
    vector_< M, T > out;
    for( unsigned i = 0; i < M; ++i ) { out[ i ] = d_[ N - M + i ]; }
    return out;
  }

  /// \p M elements from \p start.
  template < unsigned M >
  vector_< M, T > segment( unsigned start ) const
  {
    vector_< M, T > out;
    for( unsigned i = 0; i < M; ++i ) { out[ i ] = d_[ start + i ]; }
    return out;
  }

  /// From a one-column matrix, which is the same thing.
  vector_( matrix_< N, 1, T > const& m );

  /// From the first \p N elements of a run-time-length vector.
  ///
  /// Eigen assigned a `VectorXd` straight into a `Vector3d` and checked the
  /// length at run time; this has to be asked for, and it is asked for in the
  /// three places that read a fixed-length thing out of metadata.
  template < typename Dynamic >
  static vector_ from_dynamic( Dynamic const& d )
  {
    vector_ out;
    for( unsigned i = 0; i < N && i < d.size(); ++i ) { out[ i ] = d[ i ]; }
    return out;
  }

  // --------------------------------------------------------------------------
  /// Filling a vector an element at a time: `v << a, b, c;`
  ///
  /// Eigen's comma initialiser. See `matrix.h` for why it is kept.
  class comma_initializer
  {
  public:
    comma_initializer( vector_& v, T first ) : v_( v ), at_( 0 )
    { operator,( first ); }

    comma_initializer& operator,( T value )
    {
      if( at_ < N ) { v_[ at_++ ] = value; }
      return *this;
    }

  private:
    vector_& v_;
    unsigned at_;
  };

  comma_initializer operator<<( T first )
  { return comma_initializer( *this, first ); }

  /// As a one-column matrix, which a product with a row vector needs.
  matrix_< N, 1, T > asMatrix() const;

  /// As a one-row matrix.
  matrix_< 1, N, T > transpose() const;

  bool isApprox( vector_ const& o,
                 T tolerance = std::numeric_limits< T >::epsilon() *
                               T( 100 ) ) const
  {
    return ( *this - o ).norm() <=
           tolerance * ( norm() > o.norm() ? norm() : o.norm() ) ||
           ( *this - o ).norm() <= tolerance;
  }

private:
  T d_[ N ];
};

// ----------------------------------------------------------------------------
template < unsigned N, typename T >
vector_< N, T > operator+( vector_< N, T > a, vector_< N, T > const& b )
{ a += b; return a; }

template < unsigned N, typename T >
vector_< N, T > operator-( vector_< N, T > a, vector_< N, T > const& b )
{ a -= b; return a; }

template < unsigned N, typename T >
vector_< N, T > operator*( vector_< N, T > a, T s )
{ a *= s; return a; }

template < unsigned N, typename T >
vector_< N, T > operator*( T s, vector_< N, T > a )
{ a *= s; return a; }

template < unsigned N, typename T >
vector_< N, T > operator/( vector_< N, T > a, T s )
{ a /= s; return a; }

template < unsigned N, typename T >
bool operator==( vector_< N, T > const& a, vector_< N, T > const& b )
{
  for( unsigned i = 0; i < N; ++i ) { if( !( a[ i ] == b[ i ] ) ) { return false; } }
  return true;
}

template < unsigned N, typename T >
bool operator!=( vector_< N, T > const& a, vector_< N, T > const& b )
{ return !( a == b ); }

template < unsigned N, typename T >
std::ostream& operator<<( std::ostream& os, vector_< N, T > const& v )
{
  for( unsigned i = 0; i < N; ++i )
  {
    os << v[ i ];
    if( i + 1 < N ) { os << "\n"; }
  }
  return os;
}


/// \cond DoxygenSuppress
typedef vector_< 2, int >    vector_2i;
typedef vector_< 2, double > vector_2d;
typedef vector_< 2, float >  vector_2f;
typedef vector_< 3, double > vector_3d;
typedef vector_< 3, float >  vector_3f;
typedef vector_< 4, double > vector_4d;
typedef vector_< 4, float >  vector_4f;
/// \endcond

} // namespace vital

} // namespace kwiver

#endif
