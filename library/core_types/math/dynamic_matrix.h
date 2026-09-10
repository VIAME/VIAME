/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief A matrix whose size is known at run time
///
/// Only the DLT systems need one: triangulation stacks two rows per view and
/// the association matrix is as wide as there are tracks. Column-major, like
/// its fixed-size sibling, so the two share the decomposition code.

#ifndef VIAME_CORE_TYPES_MATH_DYNAMIC_MATRIX_H_
#define VIAME_CORE_TYPES_MATH_DYNAMIC_MATRIX_H_

#include "matrix.h"

#include <cmath>
#include <iostream>
#include <vector>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
template < typename T >
class dynamic_matrix
{
public:
  using value_type = T;
  using Scalar = T;

  dynamic_matrix() : rows_( 0 ), cols_( 0 ) {}

  dynamic_matrix( unsigned rows, unsigned cols )
    : rows_( rows ), cols_( cols ), d_( std::size_t( rows ) * cols, T( 0 ) ) {}

  template < unsigned R, unsigned C >
  explicit dynamic_matrix( matrix_< R, C, T > const& m )
    : rows_( R ), cols_( C ), d_( std::size_t( R ) * C )
  {
    for( unsigned i = 0; i < R * C; ++i ) { d_[ i ] = m.data()[ i ]; }
  }

  static dynamic_matrix Zero( unsigned rows, unsigned cols )
  { return dynamic_matrix( rows, cols ); }

  static dynamic_matrix Identity( unsigned n )
  {
    dynamic_matrix out( n, n );
    for( unsigned i = 0; i < n; ++i ) { out( i, i ) = T( 1 ); }
    return out;
  }

  void resize( unsigned rows, unsigned cols )
  {
    rows_ = rows;
    cols_ = cols;
    d_.assign( std::size_t( rows ) * cols, T( 0 ) );
  }

  T& operator()( unsigned r, unsigned c )
  { return d_[ std::size_t( c ) * rows_ + r ]; }

  T const& operator()( unsigned r, unsigned c ) const
  { return d_[ std::size_t( c ) * rows_ + r ]; }

  T* data()             { return d_.data(); }
  T const* data() const { return d_.data(); }

  unsigned rows() const { return rows_; }
  unsigned cols() const { return cols_; }
  std::size_t size() const { return d_.size(); }

  void setZero() { d_.assign( d_.size(), T( 0 ) ); }

  dynamic_matrix transpose() const
  {
    dynamic_matrix out( cols_, rows_ );
    for( unsigned c = 0; c < cols_; ++c )
    {
      for( unsigned r = 0; r < rows_; ++r ) { out( c, r ) = ( *this )( r, c ); }
    }
    return out;
  }

  T squaredNorm() const
  {
    T total = T( 0 );
    for( T const& x : d_ ) { total += x * x; }
    return total;
  }

  T norm() const { return static_cast< T >( std::sqrt( squaredNorm() ) ); }

  /// The \p R by \p C block whose top left corner is (\p r0, \p c0).
  template < unsigned R, unsigned C >
  matrix_< R, C, T > block( unsigned r0, unsigned c0 ) const
  {
    matrix_< R, C, T > out;
    for( unsigned c = 0; c < C; ++c )
    {
      for( unsigned r = 0; r < R; ++r )
      {
        out( r, c ) = ( *this )( r0 + r, c0 + c );
      }
    }
    return out;
  }

  template < unsigned R, unsigned C >
  void set_block( unsigned r0, unsigned c0, matrix_< R, C, T > const& b )
  {
    for( unsigned c = 0; c < C; ++c )
    {
      for( unsigned r = 0; r < R; ++r )
      {
        ( *this )( r0 + r, c0 + c ) = b( r, c );
      }
    }
  }

  template < unsigned C >
  void set_row( unsigned r, vector_< C, T > const& v )
  { for( unsigned c = 0; c < C; ++c ) { ( *this )( r, c ) = v[ c ]; } }

private:
  unsigned rows_;
  unsigned cols_;
  std::vector< T > d_;
};

// ----------------------------------------------------------------------------
template < typename T >
dynamic_matrix< T > operator*( dynamic_matrix< T > const& a,
                               dynamic_matrix< T > const& b )
{
  dynamic_matrix< T > out( a.rows(), b.cols() );
  for( unsigned c = 0; c < b.cols(); ++c )
  {
    for( unsigned k = 0; k < a.cols(); ++k )
    {
      T const bkc = b( k, c );
      if( bkc == T( 0 ) ) { continue; }
      for( unsigned r = 0; r < a.rows(); ++r )
      {
        out( r, c ) += a( r, k ) * bkc;
      }
    }
  }
  return out;
}

template < typename T >
std::ostream& operator<<( std::ostream& os, dynamic_matrix< T > const& m )
{
  for( unsigned r = 0; r < m.rows(); ++r )
  {
    for( unsigned c = 0; c < m.cols(); ++c )
    {
      os << m( r, c );
      if( c + 1 < m.cols() ) { os << " "; }
    }
    if( r + 1 < m.rows() ) { os << "\n"; }
  }
  return os;
}

/// \cond DoxygenSuppress
typedef dynamic_matrix< double > matrix_d;
typedef dynamic_matrix< float >  matrix_f;
/// \endcond

} // namespace vital

} // namespace kwiver

#endif
