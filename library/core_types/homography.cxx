// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief core homography template implementations

#include "homography.h"

#include <cmath>

#include <viame/algorithm_framework/exceptions/math.h>
namespace kwiver {

namespace vital {

namespace // anonymous
{

/// Private helper method for point transformation via homography matrix
template < typename T >
vector_< 2, T >
h_map_point(
  matrix_< 3, 3, T > const& h,
  vector_< 2, T > const& p )
{
  vector_< 3, T > out_pt = h * vector_< 3, T >(
    p[ 0 ],
    p[ 1 ], 1.0 );

  if( fabs( out_pt[ 2 ] ) <= math_dummy_precision< T >() )
  {
    VITAL_THROW( point_maps_to_infinity );
  }
  return vector_< 2, T >(
    out_pt[ 0 ] / out_pt[ 2 ],
    out_pt[ 1 ] / out_pt[ 2 ] );
}

} // end anonymous namespace

/// Construct an identity homography
template < typename T >
homography_< T >

::homography_()
  : h_( matrix_t::Identity() )
{}

/// Construct from a provided transformation matrix
template < typename T >
homography_< T >

::homography_( matrix_< 3, 3, T > const& mat )
  : h_( mat )
{}

/// Conversion Copy constructor -- float specialization
template <>
template <>
homography_< float >

::homography_( homography_< float > const& other )
  : h_( other.get_matrix() )
{}

/// Conversion Copy constructor -- double specialization
template <>
template <>
homography_< double >

::homography_( homography_< double > const& other )
  : h_( other.get_matrix() )
{}

/// Construct from a generic homography
template < typename T >
homography_< T >

::homography_( homography const& base )
  : h_( base.matrix().template cast< T >() )
{}

/// Construct from a generic homography -- double specialization
template <>
homography_< double >

::homography_( homography const& base )
  : h_( base.matrix() )
{}

/// Create a clone of outself as a shared pointer
template < typename T >
transform_2d_sptr
homography_< T >
::clone() const
{
  return std::make_shared< homography_< T > >( *this );
}

/// Get a double-typed copy of the underlying matrix transformation
template < typename T >
matrix_< 3, 3, double >
homography_< T >
::matrix() const
{
  return this->h_.template cast< double >();
}

/// Specialization for homographies with native double type
template <>
matrix_< 3, 3, double >
homography_< double >
::matrix() const
{
  return this->h_;
}

/// Normalize homography transformation in-place
template < typename T >
homography_sptr
homography_< T >
::normalize() const
{
  matrix_t norm = this->get_matrix();

  if( fabs( norm( 2, 2 ) ) >= math_dummy_precision< T >() )
  {
    norm /= norm( 2, 2 );
  }
  return std::make_shared< homography_< T > >( norm );
}

/// Inverse the homography transformation returning a new transformation
template < typename T >
homography_sptr
homography_< T >
::inverse() const
{
  // Eigen answered this with `computeInverseWithCheck`, which is the
  // determinant against the same threshold.
  T const det = this->h_.determinant();
  if( !( std::abs( det ) > math_dummy_precision< T >() ) )
  {
    VITAL_THROW( non_invertible );
  }

  matrix_t const inv = this->h_.inverse();
  return std::make_shared< homography_< T > >( inv );
}

/// Map a 2D double-type point using this homography
template < typename T >
vector_< 2, double >
homography_< T >
::map( vector_< 2, double > const& p ) const
{
  // Explicitly refer to templated version of method so as to not infinitely
  // recurse.
  matrix_< 3, 3, double > m = h_.template cast< double >();

  return h_map_point( m, p );
}

/// Map a 2D double-type point using this homography -- double specialization
template <>
vector_< 2, double >
homography_< double >
::map( vector_< 2, double > const& p ) const
{
  return h_map_point( h_, p );
}

/// Get the underlying matrix transformation
template < typename T >
typename homography_< T >::matrix_t&

homography_< T >
::get_matrix()
{
  return this->h_;
}

/// Get a const new copy of the underlying matrix transformation.
template < typename T >
typename homography_< T >::matrix_t const&

homography_< T >
::get_matrix() const
{
  return this->h_;
}

/// Map a 2D point using this homography -- generic version
template < typename T >
vector_< 2, T >
homography_< T >
::map_point( vector_< 2, T > const& p ) const
{
  return h_map_point< T >( h_.template cast< T >(), p );
}

/// Map a 2D point using this homography -- float specialization
template <>
vector_< 2, float >
homography_< float >
::map_point( vector_< 2, float > const& p ) const
{
  return h_map_point( h_, p );
}

/// Map a 2D point using this homography -- double specialization
template <>
vector_< 2, double >
homography_< double >
::map_point( vector_< 2, double > const& p ) const
{
  return h_map_point( h_, p );
}

/// Custom f2f_homography multiplication operator.
template < typename T >
homography_< T >
homography_< T >
::operator*( homography_< T > const& rhs ) const
{
  return homography_< T >( h_ * rhs.h_ );
}

// ----------------------------------------------------------------------------
// Other Functions
// ----------------------------------------------------------------------------

/// homography_<T> output stream operator
template < typename T >
std::ostream&
operator<<( std::ostream& s, homography_< T > const& h )
{
  s << h.get_matrix();
  return s;
}

/// Output stream operator for \p homography instances
std::ostream&
operator<<( std::ostream& s, homography const& h )
{
  s << h.matrix();
  return s;
}

// ----------------------------------------------------------------------------
// Template class instantiation
// ----------------------------------------------------------------------------
/// \cond DoxygenSuppress
#define INSTANTIATE_HOMOGRAPHY( T )       \
template class homography_< T >;          \
template VITAL_TYPES_EXPORT std::ostream& \
operator<<(                               \
  std::ostream&,                          \
  homography_< T > const& )

INSTANTIATE_HOMOGRAPHY( float );
INSTANTIATE_HOMOGRAPHY( double );
#undef INSTANTIATE_HOMOGRAPHY
/// \endcond

} // namespace vital

}   // end vital namespace
