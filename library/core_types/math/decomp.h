/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief The decompositions VIAME asks Eigen for
///
/// Nine lines under `library/` reach for a decomposition and they want three
/// things: the null vector of a stacked DLT system, the symmetric eigenvectors
/// of a small covariance, and a least-squares solution. One-sided Jacobi
/// answers all three, is thirty lines, and is accurate to the last bit or two
/// on the sizes here -- the largest matrix in VIAME is 2n by 4 for n views.
///
/// Singular values come back in descending order and the columns of V with
/// them, which is the order Eigen's JacobiSVD produces and which the callers
/// depend on: they take the last column.

#ifndef VIAME_CORE_TYPES_MATH_DECOMP_H_
#define VIAME_CORE_TYPES_MATH_DECOMP_H_

#include "dynamic_matrix.h"
#include "matrix.h"
#include "vector.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
/// The thin singular value decomposition of an m by n matrix, m >= n.
///
/// `A == U * diag( s ) * V^T`. `U` is m by n, `s` has n entries in descending
/// order, `V` is n by n.
template < typename T >
class jacobi_svd
{
public:
  explicit jacobi_svd( dynamic_matrix< T > const& a,
                       unsigned max_sweeps = 60 )
    : u_( a ), v_( dynamic_matrix< T >::Identity( a.cols() ) ),
      s_( a.cols(), T( 0 ) )
  {
    unsigned const m = a.rows();
    unsigned const n = a.cols();

    // One-sided Jacobi: rotate pairs of columns of U until they are mutually
    // orthogonal. What is left is U * diag( s ), and V accumulates the
    // rotations.
    T const eps = std::numeric_limits< T >::epsilon();

    for( unsigned sweep = 0; sweep < max_sweeps; ++sweep )
    {
      T off = T( 0 );

      for( unsigned p = 0; p + 1 < n; ++p )
      {
        for( unsigned q = p + 1; q < n; ++q )
        {
          T alpha = T( 0 ), beta = T( 0 ), gamma = T( 0 );
          for( unsigned i = 0; i < m; ++i )
          {
            alpha += u_( i, p ) * u_( i, p );
            beta  += u_( i, q ) * u_( i, q );
            gamma += u_( i, p ) * u_( i, q );
          }

          if( gamma == T( 0 ) ) { continue; }

          T const scale = std::sqrt( alpha * beta );
          if( scale == T( 0 ) || std::abs( gamma ) <= eps * scale ) { continue; }

          off = std::max( off, std::abs( gamma ) / scale );

          // The rotation that zeroes gamma, by the stable half-angle form.
          T const zeta = ( beta - alpha ) / ( T( 2 ) * gamma );
          T const t = ( zeta >= T( 0 ) ? T( 1 ) : T( -1 ) ) /
                      ( std::abs( zeta ) + std::sqrt( T( 1 ) + zeta * zeta ) );
          T const c = T( 1 ) / std::sqrt( T( 1 ) + t * t );
          T const s = c * t;

          for( unsigned i = 0; i < m; ++i )
          {
            T const up = u_( i, p ), uq = u_( i, q );
            u_( i, p ) = c * up - s * uq;
            u_( i, q ) = s * up + c * uq;
          }

          for( unsigned i = 0; i < n; ++i )
          {
            T const vp = v_( i, p ), vq = v_( i, q );
            v_( i, p ) = c * vp - s * vq;
            v_( i, q ) = s * vp + c * vq;
          }
        }
      }

      if( off <= eps ) { break; }
    }

    // The column norms are the singular values; normalise U by them.
    for( unsigned j = 0; j < n; ++j )
    {
      T sum = T( 0 );
      for( unsigned i = 0; i < m; ++i ) { sum += u_( i, j ) * u_( i, j ); }
      s_[ j ] = std::sqrt( sum );

      if( s_[ j ] > T( 0 ) )
      {
        for( unsigned i = 0; i < m; ++i ) { u_( i, j ) /= s_[ j ]; }
      }
    }

    sort_descending();
  }

  dynamic_matrix< T > const& matrixU() const { return u_; }
  dynamic_matrix< T > const& matrixV() const { return v_; }
  std::vector< T > const& singularValues() const { return s_; }

  /// The column of V for the smallest singular value: the null vector.
  std::vector< T > null_vector() const
  {
    unsigned const n = v_.cols();
    std::vector< T > out( n );
    for( unsigned i = 0; i < n; ++i ) { out[ i ] = v_( i, n - 1 ); }
    return out;
  }

private:
  void sort_descending()
  {
    unsigned const n = v_.cols();
    std::vector< unsigned > order( n );
    for( unsigned i = 0; i < n; ++i ) { order[ i ] = i; }
    std::stable_sort( order.begin(), order.end(),
                      [ this ]( unsigned a, unsigned b )
                      { return s_[ a ] > s_[ b ]; } );

    dynamic_matrix< T > u( u_.rows(), n );
    dynamic_matrix< T > v( v_.rows(), n );
    std::vector< T > s( n );
    for( unsigned j = 0; j < n; ++j )
    {
      s[ j ] = s_[ order[ j ] ];
      for( unsigned i = 0; i < u_.rows(); ++i ) { u( i, j ) = u_( i, order[ j ] ); }
      for( unsigned i = 0; i < v_.rows(); ++i ) { v( i, j ) = v_( i, order[ j ] ); }
    }
    u_ = u;
    v_ = v;
    s_ = s;
  }

  dynamic_matrix< T > u_;
  dynamic_matrix< T > v_;
  std::vector< T > s_;
};

// ----------------------------------------------------------------------------
/// The eigenvalues and eigenvectors of a symmetric matrix, ascending.
///
/// The cyclic Jacobi method: rotate away the largest off-diagonal entry until
/// none is left. Symmetric input is assumed, not checked; the callers build
/// theirs from a covariance or a normal-equation matrix.
template < unsigned N, typename T >
class jacobi_eigen_symmetric
{
public:
  explicit jacobi_eigen_symmetric( matrix_< N, N, T > const& a,
                                   unsigned max_sweeps = 60 )
    : vectors_( matrix_< N, N, T >::Identity() )
  {
    matrix_< N, N, T > m = a;
    T const eps = std::numeric_limits< T >::epsilon();

    for( unsigned sweep = 0; sweep < max_sweeps; ++sweep )
    {
      T off = T( 0 );
      for( unsigned p = 0; p + 1 < N; ++p )
      {
        for( unsigned q = p + 1; q < N; ++q ) { off += m( p, q ) * m( p, q ); }
      }
      if( off <= eps * eps ) { break; }

      for( unsigned p = 0; p + 1 < N; ++p )
      {
        for( unsigned q = p + 1; q < N; ++q )
        {
          if( m( p, q ) == T( 0 ) ) { continue; }

          T const theta = ( m( q, q ) - m( p, p ) ) / ( T( 2 ) * m( p, q ) );
          T const t = ( theta >= T( 0 ) ? T( 1 ) : T( -1 ) ) /
                      ( std::abs( theta ) + std::sqrt( T( 1 ) + theta * theta ) );
          T const c = T( 1 ) / std::sqrt( T( 1 ) + t * t );
          T const s = c * t;

          for( unsigned i = 0; i < N; ++i )
          {
            T const mip = m( i, p ), miq = m( i, q );
            m( i, p ) = c * mip - s * miq;
            m( i, q ) = s * mip + c * miq;
          }
          for( unsigned i = 0; i < N; ++i )
          {
            T const mpi = m( p, i ), mqi = m( q, i );
            m( p, i ) = c * mpi - s * mqi;
            m( q, i ) = s * mpi + c * mqi;
          }
          for( unsigned i = 0; i < N; ++i )
          {
            T const vip = vectors_( i, p ), viq = vectors_( i, q );
            vectors_( i, p ) = c * vip - s * viq;
            vectors_( i, q ) = s * vip + c * viq;
          }
        }
      }
    }

    for( unsigned i = 0; i < N; ++i ) { values_[ i ] = m( i, i ); }
    sort_ascending();
  }

  vector_< N, T > const& eigenvalues() const  { return values_; }
  matrix_< N, N, T > const& eigenvectors() const { return vectors_; }

private:
  void sort_ascending()
  {
    for( unsigned i = 0; i + 1 < N; ++i )
    {
      unsigned smallest = i;
      for( unsigned j = i + 1; j < N; ++j )
      {
        if( values_[ j ] < values_[ smallest ] ) { smallest = j; }
      }
      if( smallest == i ) { continue; }

      T const tmp = values_[ i ];
      values_[ i ] = values_[ smallest ];
      values_[ smallest ] = tmp;

      for( unsigned r = 0; r < N; ++r )
      {
        T const v = vectors_( r, i );
        vectors_( r, i ) = vectors_( r, smallest );
        vectors_( r, smallest ) = v;
      }
    }
  }

  vector_< N, T > values_;
  matrix_< N, N, T > vectors_;
};

// ----------------------------------------------------------------------------
/// The Cholesky factor L of a symmetric positive definite A, so A == L * L^T.
///
/// Returns false and leaves \p l unspecified if \p a is not positive definite.
template < unsigned N, typename T >
bool cholesky( matrix_< N, N, T > const& a, matrix_< N, N, T >& l )
{
  l.setZero();
  for( unsigned i = 0; i < N; ++i )
  {
    for( unsigned j = 0; j <= i; ++j )
    {
      T sum = a( i, j );
      for( unsigned k = 0; k < j; ++k ) { sum -= l( i, k ) * l( j, k ); }

      if( i == j )
      {
        if( sum <= T( 0 ) ) { return false; }
        l( i, j ) = std::sqrt( sum );
      }
      else
      {
        l( i, j ) = sum / l( j, j );
      }
    }
  }
  return true;
}

// ----------------------------------------------------------------------------
/// The least-squares solution of `A x = b`, by the singular decomposition.
///
/// Small singular values are dropped rather than inverted, at the relative
/// threshold Eigen's `setThreshold` default uses.
template < typename T >
std::vector< T > solve_least_squares( dynamic_matrix< T > const& a,
                                      std::vector< T > const& b )
{
  jacobi_svd< T > svd( a );
  auto const& u = svd.matrixU();
  auto const& v = svd.matrixV();
  auto const& s = svd.singularValues();

  unsigned const n = v.cols();
  T const largest = s.empty() ? T( 0 ) : s[ 0 ];
  T const cutoff = largest * std::numeric_limits< T >::epsilon() *
                   static_cast< T >( a.rows() > a.cols() ? a.rows() : a.cols() );

  std::vector< T > out( n, T( 0 ) );
  for( unsigned j = 0; j < n; ++j )
  {
    if( s[ j ] <= cutoff ) { continue; }

    T dot = T( 0 );
    for( unsigned i = 0; i < a.rows(); ++i ) { dot += u( i, j ) * b[ i ]; }
    dot /= s[ j ];

    for( unsigned i = 0; i < n; ++i ) { out[ i ] += v( i, j ) * dot; }
  }
  return out;
}

} // namespace vital

} // namespace kwiver

#endif
