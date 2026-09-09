// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief core covariance interface

#ifndef VITAL_COVARIANCE_H_
#define VITAL_COVARIANCE_H_

#include <viame/core_types/vital_types_export.h>

#include <cassert>
#include <iostream>

#include <viame/core_types/matrix.h>

namespace kwiver {

namespace vital {

/// A representation of covariance of a measurement
template < size_t N, typename T >
class covariance_
{
public:
  /// Number of unique values in a NxN symmetric matrix
  static auto const data_size = N * ( N + 1 ) / 2;
  using data_type = T;
  using matrix_type = Eigen::Matrix< T, N, N >;

  /// Default Constructor - Initialize to identity
  covariance_()
  {
    // Setting identity matrix values
    size_t n = 0;
    for( size_t j = 0; j < N; ++j )
    {
      for( size_t i = 0; i < j; ++i )
      {
        data_[ n++ ] = T( 0 );
      }
      data_[ n++ ] = T( 1 );
    }
  }

  /// Copy constructor
  covariance_( const covariance_< N, T >& other )
  {
    memcpy( data_, other.data_, sizeof( data_ ) );
  }

  /// Copy Constructor from another type
  template < typename U > explicit covariance_(
    const covariance_< N, U >& other )
  {
    const U* in = other.data();
    T* out = this->data_;
    for( size_t i = 0; i < data_size; ++i, ++in, ++out )
    {
      *out = static_cast< T >( *in );
    }
  }

  /// Constructor - initialize to identity matrix times a scalar
  explicit covariance_( const T& value )
  {
    size_t n = 0;
    for( size_t j = 0; j < N; ++j )
    {
      for( size_t i = 0; i < j; ++i )
      {
        data_[ n++ ] = T( 0 );
      }
      data_[ n++ ] = value;
    }
  }

  /// Constructor - from a matrix
  ///
  /// Averages off diagonal elements to enforce symmetry
  /// \param mat matrix to construct from.
  explicit covariance_( const matrix_type& mat )
  {
    size_t n = 0;
    for( size_t j = 0; j < N; ++j )
    {
      for( size_t i = 0; i < j; ++i )
      {
        data_[ n++ ] = ( mat( i, j ) + mat( j, i ) ) / 2;
      }
      data_[ n++ ] = mat( j, j );
    }
  }

  /// Assignment operator
  covariance_< N, T >&
  operator=( const covariance_< N, T >& other )
  {
    memcpy( data_, other.data_, sizeof( data_ ) );
    return *this;
  }

  /// Extract a full matrix
  matrix_type
  matrix() const
  {
    Eigen::Matrix< T, N, N > mat;
    size_t n = 0;
    for( size_t j = 0; j < N; ++j )
    {
      for( size_t i = 0; i < j; ++i )
      {
        mat( i, j ) = mat( j, i ) = data_[ n++ ];
      }
      mat( j, j ) = data_[ n++ ];
    }
    return mat;
  }

  /// Return the i-th row, j-th column
  T&
  operator()( size_t i, size_t j )
  {
    assert( i < N );
    assert( j < N );
    return data_[ vector_index( i, j ) ];
  }

  /// Return the i-th row, j-th column (const)
  const T&
  operator()( size_t i, size_t j ) const
  {
    assert( i < N );
    assert( j < N );
    return data_[ vector_index( i, j ) ];
  }

  /// Access the underlying data
  const T*
  data() const { return data_; }
  void set_data( T* in_data ) { memcpy( data_, in_data, sizeof( data_ ) ); }

  /// Equality operator
  bool
  operator==( covariance_< N, T > const& other ) const
  {
    const T* d1 = data_;
    const T* d2 = other.data_;
    for( size_t i = 0; i < data_size; ++i, ++d1, ++d2 )
    {
      if( *d1 != *d2 )
      {
        return false;
      }
    }
    return true;
  }

  /// Inequality operator
  bool
  operator!=( covariance_< N, T > const& other ) const
  {
    return !operator==( other );
  }

  // + This does not belong here!!! Serialization is an operation not a
  // property!
  /// Serialization of the class data
  template < class Archive >
  void
  serialize( Archive& archive )
  {
    T* d = data_;
    for( size_t i = 0; i < data_size; ++i, ++d )
    {
      archive( *d );
    }
  }

protected:
  /// Convert from matrix to vector indices
  size_t
  vector_index( size_t i, size_t j ) const
  {
    return ( j > i ) ? j * ( j + 1 ) / 2 + i
                     : i * ( i + 1 ) / 2 + j;
  }

  /// data of the sparse symmetric covariance matrix; column-major format
  T data_[ data_size ];
};

/// \cond DoxygenSuppress
using covariance_2d = covariance_< 2, double >;
using covariance_2f = covariance_< 2, float >;
using covariance_3d = covariance_< 3, double >;
using covariance_3f = covariance_< 3, float >;
using covariance_4d = covariance_< 4, double >;
using covariance_4f = covariance_< 4, float >;
/// \endcond

} // namespace vital

}   // end namespace vital

#endif // VITAL_COVARIANCE_H_
