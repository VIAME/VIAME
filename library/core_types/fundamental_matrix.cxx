// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief core fundamental matrix template implementations

#include "fundamental_matrix.h"

#include <viame/core_types/math/decomp.h>

#include <cmath>

#include <viame/algorithm_framework/exceptions/math.h>
namespace kwiver {

namespace vital {

/// Construct from a provided matrix
template < typename T >
fundamental_matrix_< T >

::fundamental_matrix_( matrix_< 3, 3, T > const& mat )
{
  jacobi_svd< T > const svd{ dynamic_matrix< T >( mat ) };
  matrix_t const U = svd.matrixU().template block< 3, 3 >( 0, 0 );
  matrix_t const V = svd.matrixV().template block< 3, 3 >( 0, 0 );

  // Clear the last singular value: a fundamental matrix has rank two.
  vector_< 3, T > S( svd.singularValues()[ 0 ], svd.singularValues()[ 1 ],
                     T( 0 ) );
  S = S / S.norm();

  matrix_t diagonal;
  diagonal( 0, 0 ) = S[ 0 ];
  diagonal( 1, 1 ) = S[ 1 ];
  diagonal( 2, 2 ) = S[ 2 ];

  mat_ = U * diagonal * V.transpose();
}

/// Conversion Copy constructor -- float specialization
template <>
template <>
fundamental_matrix_< float >

::fundamental_matrix_( fundamental_matrix_< float > const& other )
  : mat_( other.mat_ )
{}

/// Conversion Copy constructor -- double specialization
template <>
template <>
fundamental_matrix_< double >

::fundamental_matrix_( fundamental_matrix_< double > const& other )
  : mat_( other.mat_ )
{}

/// Construct from a generic fundamental_matrix
template < typename T >
fundamental_matrix_< T >

::fundamental_matrix_( fundamental_matrix const& base )
  : mat_( base.matrix().template cast< T >() )
{}

/// Construct from a generic fundamental_matrix -- double specialization
template <>
fundamental_matrix_< double >

::fundamental_matrix_( fundamental_matrix const& base )
  : mat_( base.matrix() )
{}

/// Create a clone of outself as a shared pointer
template < typename T >
fundamental_matrix_sptr
fundamental_matrix_< T >
::clone() const
{
  return fundamental_matrix_sptr( new fundamental_matrix_< T >( *this ) );
}

/// Get a double-typed copy of the underlying matrix
template < typename T >
matrix_< 3, 3, double >
fundamental_matrix_< T >
::matrix() const
{
  return this->mat_.template cast< double >();
}

/// Specialization for matrices with native double type
template <>
matrix_< 3, 3, double >
fundamental_matrix_< double >
::matrix() const
{
  return this->mat_;
}

// ----------------------------------------------------------------------------
// Other Functions
// ----------------------------------------------------------------------------

/// Output stream operator for \p fundamental_matrix instances
std::ostream&
operator<<( std::ostream& s, fundamental_matrix const& f )
{
  s << f.matrix();
  return s;
}

// ----------------------------------------------------------------------------
// Template class instantiation
// ----------------------------------------------------------------------------
/// \cond DoxygenSuppress
#define INSTANTIATE_FUNDAMENTAL_MATRIX( T ) \
template class fundamental_matrix_< T >;

INSTANTIATE_FUNDAMENTAL_MATRIX( float );
INSTANTIATE_FUNDAMENTAL_MATRIX( double );
#undef INSTANTIATE_FUNDAMENTAL_MATRIX
/// \endcond

} // namespace vital

}   // end vital namespace
