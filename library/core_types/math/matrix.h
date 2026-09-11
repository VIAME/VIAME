/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Fixed-size and dynamically-sized dense matrices
///
/// Column-major, as Eigen's default is, so that `data()` hands the same bytes
/// to numpy and to OpenCV as before. Every operation returns a value; see
/// `vector.h` for why there are no expression templates.

#ifndef VIAME_CORE_TYPES_MATH_MATRIX_H_
#define VIAME_CORE_TYPES_MATH_MATRIX_H_

#include "vector.h"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// An \p R by \p C matrix of \p T, stored column-major.
template < unsigned R, unsigned C, typename T >
class matrix_
{
public:
  static_assert( R > 0 && C > 0, "a matrix has at least one element" );

  using value_type = T;
  using Scalar = T;
  static constexpr unsigned row_count = R;
  static constexpr unsigned col_count = C;

  constexpr matrix_() : d_{} {}

  /// A column vector is a one-column matrix, and the code that came from
  /// Eigen moves between the two without saying so.
  matrix_( vector_< R, T > const& v )
  {
    static_assert( C == 1, "only a one-column matrix is a vector" );
    for( unsigned i = 0; i < R; ++i ) { d_[ i ] = v[ i ]; }
  }

  operator vector_< R, T >() const
  {
    static_assert( C == 1, "only a one-column matrix is a vector" );
    vector_< R, T > out;
    for( unsigned i = 0; i < R; ++i ) { out[ i ] = d_[ i ]; }
    return out;
  }

  explicit matrix_( T value )
  { for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] = value; } }

  template < typename U >
  matrix_< R, C, U > cast() const
  {
    matrix_< R, C, U > out;
    for( unsigned i = 0; i < R * C; ++i )
    {
      out.data()[ i ] = static_cast< U >( d_[ i ] );
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Element access. Column-major, so element (r, c) is at c * R + r.
  T& operator()( unsigned r, unsigned c )             { return d_[ c * R + r ]; }
  T const& operator()( unsigned r, unsigned c ) const { return d_[ c * R + r ]; }

  /// A one-column or one-row matrix indexes like a vector.
  T& operator()( unsigned i )
  { static_assert( R == 1 || C == 1, "not a vector" ); return d_[ i ]; }
  T const& operator()( unsigned i ) const
  { static_assert( R == 1 || C == 1, "not a vector" ); return d_[ i ]; }

  T* data()             { return d_; }
  T const* data() const { return d_; }

  static constexpr unsigned rows() { return R; }
  static constexpr unsigned cols() { return C; }
  static constexpr unsigned size() { return R * C; }

  // --------------------------------------------------------------------------
  // The named constructors
  static matrix_ Zero() { return matrix_(); }
  static matrix_ Constant( T value ) { return matrix_( value ); }

  /// A matrix of values drawn uniformly from [-1, 1].
  ///
  /// Eigen's `Random()`, which one binding exposes so that a python test can
  /// make an arbitrary homography. Deterministic across a process because the
  /// generator is a function-local static seeded once; Eigen's is too.
  static matrix_ Random()
  {
    static std::mt19937 rng( 20260910 );
    std::uniform_real_distribution< double > uniform( -1.0, 1.0 );

    matrix_ out;
    for( unsigned i = 0; i < R * C; ++i )
    {
      out.data()[ i ] = static_cast< T >( uniform( rng ) );
    }
    return out;
  }

  /// The size is fixed; the arguments say what it already is.
  static matrix_ Random( unsigned rows, unsigned cols )
  {
    ( void ) rows;
    ( void ) cols;
    return Random();
  }

  static matrix_ Identity()
  {
    matrix_ out;
    unsigned const n = R < C ? R : C;
    for( unsigned i = 0; i < n; ++i ) { out( i, i ) = T( 1 ); }
    return out;
  }


  // --------------------------------------------------------------------------
  /// Assignment, to an lvalue only.
  ///
  /// The `&` is the whole point. Eigen's `block()`, `row()`, `col()`,
  /// `head()` and their fellows return **writable proxies**, so code ported
  /// from Eigen is full of `m.block< 3, 3 >( 0, 0 ) = r;`. These return
  /// values, and assigning to a value compiles: it copy-assigns to a
  /// temporary which is then destroyed, so the statement does nothing at all
  /// and nothing says so. P6 shipped four of those, and one of them left
  /// `camera_perspective::pose_matrix()` returning a zero matrix -- which
  /// made every stereo measurement VIAME computed come out zero.
  ///
  /// With the ref qualifier, assigning to a temporary is a compile error.
  /// Use `set_block`, `set_row` and `set_col` instead. See finding 1.20.
  matrix_& operator=( matrix_ const& ) & = default;
  matrix_& operator=( matrix_&& ) & = default;
  matrix_( matrix_ const& ) = default;
  matrix_( matrix_&& ) = default;

  void setZero() { for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] = T( 0 ); } }

  void setIdentity()
  {
    setZero();
    unsigned const n = R < C ? R : C;
    for( unsigned i = 0; i < n; ++i ) { ( *this )( i, i ) = T( 1 ); }
  }

  // --------------------------------------------------------------------------
  // Rows, columns and blocks
  vector_< C, T > row( unsigned r ) const
  {
    vector_< C, T > out;
    for( unsigned c = 0; c < C; ++c ) { out[ c ] = ( *this )( r, c ); }
    return out;
  }

  vector_< R, T > col( unsigned c ) const
  {
    vector_< R, T > out;
    for( unsigned r = 0; r < R; ++r ) { out[ r ] = ( *this )( r, c ); }
    return out;
  }

  void set_row( unsigned r, vector_< C, T > const& v )
  { for( unsigned c = 0; c < C; ++c ) { ( *this )( r, c ) = v[ c ]; } }

  void set_col( unsigned c, vector_< R, T > const& v )
  { for( unsigned r = 0; r < R; ++r ) { ( *this )( r, c ) = v[ r ]; } }

  /// The \p BR by \p BC block whose top left corner is (\p r0, \p c0).
  template < unsigned BR, unsigned BC >
  matrix_< BR, BC, T > block( unsigned r0, unsigned c0 ) const
  {
    matrix_< BR, BC, T > out;
    for( unsigned c = 0; c < BC; ++c )
    {
      for( unsigned r = 0; r < BR; ++r )
      {
        out( r, c ) = ( *this )( r0 + r, c0 + c );
      }
    }
    return out;
  }

  template < unsigned BR, unsigned BC >
  void set_block( unsigned r0, unsigned c0, matrix_< BR, BC, T > const& b )
  {
    for( unsigned c = 0; c < BC; ++c )
    {
      for( unsigned r = 0; r < BR; ++r )
      {
        ( *this )( r0 + r, c0 + c ) = b( r, c );
      }
    }
  }

  matrix_< C, R, T > transpose() const
  {
    matrix_< C, R, T > out;
    for( unsigned c = 0; c < C; ++c )
    {
      for( unsigned r = 0; r < R; ++r ) { out( c, r ) = ( *this )( r, c ); }
    }
    return out;
  }

  // --------------------------------------------------------------------------
  // Arithmetic
  matrix_ operator-() const
  {
    matrix_ out;
    for( unsigned i = 0; i < R * C; ++i ) { out.data()[ i ] = -d_[ i ]; }
    return out;
  }

  matrix_& operator+=( matrix_ const& o ) &
  { for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] += o.data()[ i ]; } return *this; }

  matrix_& operator-=( matrix_ const& o ) &
  { for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] -= o.data()[ i ]; } return *this; }

  matrix_& operator*=( T s ) &
  { for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] *= s; } return *this; }

  matrix_& operator/=( T s ) &
  { for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] /= s; } return *this; }

  // --------------------------------------------------------------------------
  T sum() const
  {
    T total = T( 0 );
    for( unsigned i = 0; i < R * C; ++i ) { total += d_[ i ]; }
    return total;
  }

  T mean() const { return sum() / static_cast< T >( R * C ); }

  T squaredNorm() const
  {
    T total = T( 0 );
    for( unsigned i = 0; i < R * C; ++i ) { total += d_[ i ] * d_[ i ]; }
    return total;
  }

  /// The Frobenius norm, which is what Eigen's `norm()` on a matrix is.
  T norm() const { return static_cast< T >( std::sqrt( squaredNorm() ) ); }

  T trace() const
  {
    unsigned const n = R < C ? R : C;
    T total = T( 0 );
    for( unsigned i = 0; i < n; ++i ) { total += ( *this )( i, i ); }
    return total;
  }

  T maxCoeff() const
  {
    T best = d_[ 0 ];
    for( unsigned i = 1; i < R * C; ++i ) { if( d_[ i ] > best ) { best = d_[ i ]; } }
    return best;
  }

  T minCoeff() const
  {
    T best = d_[ 0 ];
    for( unsigned i = 1; i < R * C; ++i ) { if( d_[ i ] < best ) { best = d_[ i ]; } }
    return best;
  }

  T determinant() const;
  matrix_ inverse() const;

  /// The x with `*this * x == b`.
  ///
  /// Eigen's callers spell this `ldlt().solve( b )` or
  /// `colPivHouseholderQr().solve( b )`, choosing a factorisation. Every one
  /// of them in VIAME is two by two, where the inverse is a closed form and
  /// as accurate as any factorisation would be. `decomp.h` has the
  /// least-squares solve for the overdetermined case.
  vector_< R, T > solve( vector_< R, T > const& b ) const
  {
    static_assert( R == C, "a solve needs a square matrix" );
    return inverse() * b;
  }

  // --------------------------------------------------------------------------
  // Coefficient-wise operations, which Eigen spells through `.array()`
  matrix_ cwiseProduct( matrix_ const& o ) const
  {
    matrix_ out;
    for( unsigned i = 0; i < R * C; ++i )
    {
      out.data()[ i ] = d_[ i ] * o.data()[ i ];
    }
    return out;
  }

  matrix_ cwiseQuotient( matrix_ const& o ) const
  {
    matrix_ out;
    for( unsigned i = 0; i < R * C; ++i )
    {
      out.data()[ i ] = d_[ i ] / o.data()[ i ];
    }
    return out;
  }

  matrix_ cwiseAbs() const
  {
    matrix_ out;
    for( unsigned i = 0; i < R * C; ++i )
    {
      out.data()[ i ] = d_[ i ] < T( 0 ) ? -d_[ i ] : d_[ i ];
    }
    return out;
  }

  bool allFinite() const
  {
    for( unsigned i = 0; i < R * C; ++i )
    {
      if( !std::isfinite( d_[ i ] ) ) { return false; }
    }
    return true;
  }

  // --------------------------------------------------------------------------
  /// Filling a matrix a row at a time: `m << a, b, c, d;`
  ///
  /// Eigen's comma initialiser, in row order, kept because the alternative is
  /// rewriting every literal matrix in VIAME into nine assignments and losing
  /// the shape on the page. It is a proxy so that the commas can chain; it
  /// writes as it goes rather than at the end, so an incomplete list leaves
  /// the rest of the matrix as it was.
  class comma_initializer
  {
  public:
    comma_initializer( matrix_& m, T first ) : m_( m ), at_( 0 )
    { operator,( first ); }

    comma_initializer& operator,( T value )
    {
      if( at_ < R * C )
      {
        m_( at_ / C, at_ % C ) = value;
        ++at_;
      }
      return *this;
    }

  private:
    matrix_& m_;
    unsigned at_;
  };

  comma_initializer operator<<( T first )
  { return comma_initializer( *this, first ); }

  bool isApprox( matrix_ const& o,
                 T tolerance = std::numeric_limits< T >::epsilon() *
                               T( 100 ) ) const
  {
    T const diff = ( *this - o ).norm();
    T const scale = norm() > o.norm() ? norm() : o.norm();
    return diff <= tolerance * scale || diff <= tolerance;
  }

private:
  T d_[ R * C ];
};

// ----------------------------------------------------------------------------
template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > operator+( matrix_< R, C, T > a, matrix_< R, C, T > const& b )
{ a += b; return a; }

template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > operator-( matrix_< R, C, T > a, matrix_< R, C, T > const& b )
{ a -= b; return a; }

template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > operator*( matrix_< R, C, T > a, T s )
{ a *= s; return a; }

template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > operator*( T s, matrix_< R, C, T > a )
{ a *= s; return a; }

template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > operator/( matrix_< R, C, T > a, T s )
{ a /= s; return a; }

/// Matrix times matrix.
template < unsigned R, unsigned K, unsigned C, typename T >
matrix_< R, C, T > operator*( matrix_< R, K, T > const& a,
                              matrix_< K, C, T > const& b )
{
  matrix_< R, C, T > out;
  for( unsigned c = 0; c < C; ++c )
  {
    for( unsigned k = 0; k < K; ++k )
    {
      T const bkc = b( k, c );
      if( bkc == T( 0 ) ) { continue; }
      for( unsigned r = 0; r < R; ++r )
      {
        out( r, c ) += a( r, k ) * bkc;
      }
    }
  }
  return out;
}

/// Matrix times column vector.
template < unsigned R, unsigned C, typename T >
vector_< R, T > operator*( matrix_< R, C, T > const& m, vector_< C, T > const& v )
{
  vector_< R, T > out;
  for( unsigned c = 0; c < C; ++c )
  {
    T const vc = v[ c ];
    for( unsigned r = 0; r < R; ++r ) { out[ r ] += m( r, c ) * vc; }
  }
  return out;
}

template < unsigned R, unsigned C, typename T >
bool operator==( matrix_< R, C, T > const& a, matrix_< R, C, T > const& b )
{
  for( unsigned i = 0; i < R * C; ++i )
  {
    if( !( a.data()[ i ] == b.data()[ i ] ) ) { return false; }
  }
  return true;
}

template < unsigned R, unsigned C, typename T >
bool operator!=( matrix_< R, C, T > const& a, matrix_< R, C, T > const& b )
{ return !( a == b ); }

template < unsigned R, unsigned C, typename T >
std::ostream& operator<<( std::ostream& os, matrix_< R, C, T > const& m )
{
  for( unsigned r = 0; r < R; ++r )
  {
    for( unsigned c = 0; c < C; ++c )
    {
      os << m( r, c );
      if( c + 1 < C ) { os << " "; }
    }
    if( r + 1 < R ) { os << "\n"; }
  }
  return os;
}


// ----------------------------------------------------------------------------
// The vector members that need a matrix to state
template < unsigned N, typename T >
vector_< N, T >::vector_( matrix_< N, 1, T > const& m ) : d_{}
{
  for( unsigned i = 0; i < N; ++i ) { d_[ i ] = m( i, 0 ); }
}

template < unsigned N, typename T >
matrix_< N, 1, T > vector_< N, T >::asMatrix() const
{
  matrix_< N, 1, T > out;
  for( unsigned i = 0; i < N; ++i ) { out( i, 0 ) = d_[ i ]; }
  return out;
}

template < unsigned N, typename T >
matrix_< 1, N, T > vector_< N, T >::transpose() const
{
  matrix_< 1, N, T > out;
  for( unsigned i = 0; i < N; ++i ) { out( 0, i ) = d_[ i ]; }
  return out;
}

/// Column vector times row vector: the outer product.
template < unsigned R, unsigned C, typename T >
matrix_< R, C, T > operator*( vector_< R, T > const& v,
                              matrix_< 1, C, T > const& row )
{
  matrix_< R, C, T > out;
  for( unsigned c = 0; c < C; ++c )
  {
    for( unsigned r = 0; r < R; ++r ) { out( r, c ) = v[ r ] * row( 0, c ); }
  }
  return out;
}

/// Row vector times column vector: the inner product, as a scalar.
template < unsigned C, typename T >
T operator*( matrix_< 1, C, T > const& row, vector_< C, T > const& v )
{
  T total = T( 0 );
  for( unsigned c = 0; c < C; ++c ) { total += row( 0, c ) * v[ c ]; }
  return total;
}

/// \cond DoxygenSuppress
typedef matrix_< 2, 2, double > matrix_2x2d;
typedef matrix_< 2, 2, float >  matrix_2x2f;
typedef matrix_< 2, 3, double > matrix_2x3d;
typedef matrix_< 2, 3, float >  matrix_2x3f;
typedef matrix_< 3, 2, double > matrix_3x2d;
typedef matrix_< 3, 2, float >  matrix_3x2f;
typedef matrix_< 3, 3, double > matrix_3x3d;
typedef matrix_< 3, 3, float >  matrix_3x3f;
typedef matrix_< 3, 4, double > matrix_3x4d;
typedef matrix_< 3, 4, float >  matrix_3x4f;
typedef matrix_< 4, 3, double > matrix_4x3d;
typedef matrix_< 4, 3, float >  matrix_4x3f;
typedef matrix_< 4, 4, double > matrix_4x4d;
typedef matrix_< 4, 4, float >  matrix_4x4f;
/// \endcond

} // namespace vital

} // namespace kwiver

#include "matrix_detail.h"

#endif
