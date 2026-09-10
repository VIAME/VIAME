/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief A column vector whose length is known at run time
///
/// The only one VIAME has is a camera's distortion coefficients, which is as
/// long as the model has parameters. `Map` is a copy here rather than a view:
/// the two callers hand it a `std::vector` that outlives the result anyway,
/// and a view would carry an aliasing question for no gain at these lengths.

#ifndef VIAME_CORE_TYPES_MATH_DYNAMIC_VECTOR_H_
#define VIAME_CORE_TYPES_MATH_DYNAMIC_VECTOR_H_

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace kwiver {

namespace vital {

// ----------------------------------------------------------------------------
template < typename T > class transposed_vector_view;

// ----------------------------------------------------------------------------
template < typename T >
class dynamic_vector
{
public:
  using value_type = T;
  using Scalar = T;

  dynamic_vector() = default;
  explicit dynamic_vector( unsigned n ) : d_( n, T( 0 ) ) {}
  dynamic_vector( std::initializer_list< T > values ) : d_( values ) {}

  static dynamic_vector Zero( unsigned n ) { return dynamic_vector( n ); }

  /// A copy of \p n values starting at \p first.
  static dynamic_vector Map( T const* first, std::size_t n )
  {
    dynamic_vector out;
    out.d_.assign( first, first + n );
    return out;
  }

  T& operator()( unsigned i )             { return d_[ i ]; }
  T const& operator()( unsigned i ) const { return d_[ i ]; }
  T& operator[]( unsigned i )             { return d_[ i ]; }
  T const& operator[]( unsigned i ) const { return d_[ i ]; }

  T* data()             { return d_.data(); }
  T const* data() const { return d_.data(); }

  std::size_t size() const { return d_.size(); }
  /// A column vector's row count is its length, which is Eigen's spelling.
  std::size_t rows() const { return d_.size(); }
  static constexpr std::size_t cols() { return 1; }
  bool empty() const { return d_.empty(); }
  void resize( unsigned n ) { d_.assign( n, T( 0 ) ); }

  /// Eigen leaves `VectorXd( n )` uninitialised and callers say this; the
  /// constructor here zeroes already, so this only has to keep the spelling.
  void setZero() { d_.assign( d_.size(), T( 0 ) ); }
  void setConstant( T v ) { d_.assign( d_.size(), v ); }

  auto begin()       { return d_.begin(); }
  auto end()         { return d_.end(); }
  auto begin() const { return d_.begin(); }
  auto end() const   { return d_.end(); }

  bool allFinite() const
  {
    for( T const& x : d_ ) { if( !std::isfinite( x ) ) { return false; } }
    return true;
  }

  T squaredNorm() const
  {
    T total = T( 0 );
    for( T const& x : d_ ) { total += x * x; }
    return total;
  }

  T norm() const { return static_cast< T >( std::sqrt( squaredNorm() ) ); }

  /// The same values, printed on one line. See `transposed_vector_view`.
  transposed_vector_view< T > transpose() const;

  bool operator==( dynamic_vector const& o ) const { return d_ == o.d_; }
  bool operator!=( dynamic_vector const& o ) const { return d_ != o.d_; }

private:
  std::vector< T > d_;
};

// ----------------------------------------------------------------------------
/// A vector on its way to a stream, printed on one line
///
/// What Eigen's `transpose()` did for a column vector being written out. A
/// view rather than a copy, and a type of its own rather than a nested one so
/// that the stream operator can deduce the element type.
template < typename T >
class transposed_vector_view
{
public:
  explicit transposed_vector_view( dynamic_vector< T > const& v ) : v_( v ) {}
  dynamic_vector< T > const& vector() const { return v_; }

private:
  dynamic_vector< T > const& v_;
};

template < typename T >
std::ostream& operator<<( std::ostream& os,
                          transposed_vector_view< T > const& t )
{
  for( std::size_t i = 0; i < t.vector().size(); ++i )
  {
    os << t.vector()[ i ];
    if( i + 1 < t.vector().size() ) { os << " "; }
  }
  return os;
}

template < typename T >
std::ostream& operator<<( std::ostream& os, dynamic_vector< T > const& v )
{
  for( std::size_t i = 0; i < v.size(); ++i )
  {
    os << v[ i ];
    if( i + 1 < v.size() ) { os << "\n"; }
  }
  return os;
}

template < typename T >
transposed_vector_view< T > dynamic_vector< T >::transpose() const
{ return transposed_vector_view< T >( *this ); }

/// \cond DoxygenSuppress
typedef dynamic_vector< double > vector_d;
typedef dynamic_vector< float >  vector_f;
/// \endcond

} // namespace vital

} // namespace kwiver

#endif
