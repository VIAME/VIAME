/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief numpy conversions for the math types
///
/// What `pybind11/eigen.h` did for Eigen. A binding that took an
/// `Eigen::Vector3d` took a numpy array of three doubles and returned one;
/// the same is true here, with the same shapes, so no python changes.
///
/// The shapes are Eigen's: a vector is a one-dimensional array of N, a matrix
/// is two-dimensional R by C. A vector will also accept an (N, 1) or (1, N)
/// array on the way in, because numpy code that has been through a matrix
/// product produces those and pybind11's Eigen caster accepted them.
///
/// Conversion is by value in both directions. `pybind11/eigen.h` could return
/// a view onto an Eigen object's storage when the lifetimes allowed it; that
/// is not worth reproducing for four doubles, and a copy cannot dangle.

#ifndef VIAME_CORE_TYPES_CASTERS_H_
#define VIAME_CORE_TYPES_CASTERS_H_

#include <viame/core_types/math/dynamic_matrix.h>
#include <viame/core_types/math/dynamic_vector.h>
#include <viame/core_types/math/matrix.h>
#include <viame/core_types/math/vector.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

namespace pybind11 {

namespace detail {

// ----------------------------------------------------------------------------
template < unsigned N, typename T >
struct type_caster< kwiver::vital::vector_< N, T > >
{
  using type = kwiver::vital::vector_< N, T >;
  using array_type = array_t< T, array::forcecast | array::c_style >;

  PYBIND11_TYPE_CASTER( type, const_name( "numpy.ndarray" ) );

  bool load( handle src, bool )
  {
    if( !src ) { return false; }

    array_type a = array_type::ensure( src );
    if( !a ) { return false; }

    if( a.ndim() == 1 )
    {
      if( a.shape( 0 ) != static_cast< ssize_t >( N ) ) { return false; }
      for( unsigned i = 0; i < N; ++i ) { value[ i ] = a.at( i ); }
      return true;
    }

    // A column or a row, which numpy produces from a matrix product
    if( a.ndim() == 2 )
    {
      if( a.shape( 0 ) == static_cast< ssize_t >( N ) && a.shape( 1 ) == 1 )
      {
        for( unsigned i = 0; i < N; ++i ) { value[ i ] = a.at( i, 0 ); }
        return true;
      }
      if( a.shape( 0 ) == 1 && a.shape( 1 ) == static_cast< ssize_t >( N ) )
      {
        for( unsigned i = 0; i < N; ++i ) { value[ i ] = a.at( 0, i ); }
        return true;
      }
    }

    return false;
  }

  static handle cast( type const& src, return_value_policy, handle )
  {
    array_t< T > out( static_cast< ssize_t >( N ) );
    auto view = out.template mutable_unchecked< 1 >();
    for( unsigned i = 0; i < N; ++i ) { view( i ) = src[ i ]; }
    return out.release();
  }
};

// ----------------------------------------------------------------------------
template < unsigned R, unsigned C, typename T >
struct type_caster< kwiver::vital::matrix_< R, C, T > >
{
  using type = kwiver::vital::matrix_< R, C, T >;
  using array_type = array_t< T, array::forcecast | array::c_style >;

  PYBIND11_TYPE_CASTER( type, const_name( "numpy.ndarray" ) );

  bool load( handle src, bool )
  {
    if( !src ) { return false; }

    array_type a = array_type::ensure( src );
    if( !a || a.ndim() != 2 ) { return false; }
    if( a.shape( 0 ) != static_cast< ssize_t >( R ) ||
        a.shape( 1 ) != static_cast< ssize_t >( C ) )
    {
      return false;
    }

    for( unsigned r = 0; r < R; ++r )
    {
      for( unsigned c = 0; c < C; ++c ) { value( r, c ) = a.at( r, c ); }
    }
    return true;
  }

  static handle cast( type const& src, return_value_policy, handle )
  {
    array_t< T > out( { static_cast< ssize_t >( R ),
                        static_cast< ssize_t >( C ) } );
    auto view = out.template mutable_unchecked< 2 >();
    for( unsigned r = 0; r < R; ++r )
    {
      for( unsigned c = 0; c < C; ++c ) { view( r, c ) = src( r, c ); }
    }
    return out.release();
  }
};

// ----------------------------------------------------------------------------
template < typename T >
struct type_caster< kwiver::vital::dynamic_vector< T > >
{
  using type = kwiver::vital::dynamic_vector< T >;
  using array_type = array_t< T, array::forcecast | array::c_style >;

  PYBIND11_TYPE_CASTER( type, const_name( "numpy.ndarray" ) );

  bool load( handle src, bool )
  {
    if( !src ) { return false; }

    array_type a = array_type::ensure( src );
    if( !a ) { return false; }

    if( a.ndim() == 1 )
    {
      value.resize( static_cast< unsigned >( a.shape( 0 ) ) );
      for( ssize_t i = 0; i < a.shape( 0 ); ++i )
      {
        value[ static_cast< unsigned >( i ) ] = a.at( i );
      }
      return true;
    }

    if( a.ndim() == 2 && a.shape( 1 ) == 1 )
    {
      value.resize( static_cast< unsigned >( a.shape( 0 ) ) );
      for( ssize_t i = 0; i < a.shape( 0 ); ++i )
      {
        value[ static_cast< unsigned >( i ) ] = a.at( i, 0 );
      }
      return true;
    }

    return false;
  }

  static handle cast( type const& src, return_value_policy, handle )
  {
    array_t< T > out( static_cast< ssize_t >( src.size() ) );
    auto view = out.template mutable_unchecked< 1 >();
    for( std::size_t i = 0; i < src.size(); ++i )
    {
      view( static_cast< ssize_t >( i ) ) = src[ static_cast< unsigned >( i ) ];
    }
    return out.release();
  }
};

// ----------------------------------------------------------------------------
template < typename T >
struct type_caster< kwiver::vital::dynamic_matrix< T > >
{
  using type = kwiver::vital::dynamic_matrix< T >;
  using array_type = array_t< T, array::forcecast | array::c_style >;

  PYBIND11_TYPE_CASTER( type, const_name( "numpy.ndarray" ) );

  bool load( handle src, bool )
  {
    if( !src ) { return false; }

    array_type a = array_type::ensure( src );
    if( !a || a.ndim() != 2 ) { return false; }

    value.resize( static_cast< unsigned >( a.shape( 0 ) ),
                  static_cast< unsigned >( a.shape( 1 ) ) );
    for( ssize_t r = 0; r < a.shape( 0 ); ++r )
    {
      for( ssize_t c = 0; c < a.shape( 1 ); ++c )
      {
        value( static_cast< unsigned >( r ), static_cast< unsigned >( c ) ) =
          a.at( r, c );
      }
    }
    return true;
  }

  static handle cast( type const& src, return_value_policy, handle )
  {
    array_t< T > out( { static_cast< ssize_t >( src.rows() ),
                        static_cast< ssize_t >( src.cols() ) } );
    auto view = out.template mutable_unchecked< 2 >();
    for( unsigned r = 0; r < src.rows(); ++r )
    {
      for( unsigned c = 0; c < src.cols(); ++c )
      {
        view( static_cast< ssize_t >( r ), static_cast< ssize_t >( c ) ) =
          src( r, c );
      }
    }
    return out.release();
  }
};

} // namespace detail

} // namespace pybind11

#endif
