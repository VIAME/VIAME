// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of \link kwiver::vital::similarity_ similarity_<T> \endlink
 *        for \c T = { \c float, \c double }
 */

#include "similarity.h"
#include <vital/io/eigen_io.h>
#include <cmath>
#include <Eigen/LU>

namespace kwiver {
namespace vital {

/// Constructor - from a matrix
template < typename T >
similarity_< T >
::similarity_( const Eigen::Matrix< T, 4, 4 >& M )
  : m_logger( kwiver::vital::get_logger( "vital.similarity" ) )
{
  if ( ( M( 3, 0 ) != T( 0 ) )
       || ( M( 3, 1 ) != T( 0 ) )
       || ( M( 3, 2 ) != T( 0 ) )
       || ( M( 3, 3 ) != T( 1 ) ) )
  {
    // not a similarity if bottom row is not [0,0,0,1]
    // TODO throw an exception here
    LOG_WARN( m_logger, "third row of similarity matrix must be [0,0,0,1]" );
    return;
  }
  Eigen::Matrix< T, 3, 3 > sr = M.template block< 3, 3 > ( 0, 0 );
  this->scale_ = sr.determinant();
  if ( this->scale_ <= T( 0 ) )
  {
    // similarity must have positive scale
    // TODO throw an exception
    LOG_WARN( m_logger, "determinant in upper 3x3 of similarity matrix must be positive" );
    return;
  }
  // take the cube root
  this->scale_ = std::pow( this->scale_, static_cast< T > ( 1 / 3.0 ) );
  // factor scale out of sr
  sr /= this->scale_;
  this->rot_ = rotation_< T > ( sr );
  assert( ( this->rot_.matrix() - sr ).norm() < 1e-4 );
  this->trans_ = M.template block< 3, 1 > ( 0, 3 );
}

/// Convert to a 4x4 matrix
template < typename T >
Eigen::Matrix< T, 4, 4 >
similarity_< T >
::matrix() const
{
  Eigen::Matrix< T, 4, 4 > mat = Eigen::Matrix< T, 4, 4 >::Zero();
  mat( 3, 3 ) = 1;
  mat.template block< 3, 3 > ( 0, 0 ) = this->scale_ * this->rot_.matrix();
  mat.template block< 3, 1 > ( 0, 3 ) = this->trans_;
  return mat;
}

/// Compose two transforms
template < typename T >
similarity_< T >
similarity_< T >
::operator*( const similarity_< T >& rhs ) const
{
  return similarity_< T > ( this->scale_ * rhs.scale_,
                            this->rot_ * rhs.rot_,
                            *this * rhs.trans_ );
}

/// Transform a vector
template < typename T >
Eigen::Matrix< T, 3, 1 >
similarity_< T >
::operator*( const Eigen::Matrix< T, 3, 1 >& rhs ) const
{
  return this->scale_ * ( this->rot_ * rhs ) + this->trans_;
}

/// output stream operator for a similarity transformation
template < typename T >
std::ostream&
operator<<( std::ostream& s, const similarity_< T >& t )
{
  s << t.scale() << ' ' << t.rotation() << ' ' << t.translation();
  return s;
}

/// input stream operator for a similarity transformation
template < typename T >
std::istream&
operator>>( std::istream& s, similarity_< T >& t )
{
  T sc;

  rotation_< T > ro;
  Eigen::Matrix< T, 3, 1 > tr;
  s >> sc >> ro >> tr;
  t = similarity_< T > ( sc, ro, tr );
  return s;
}

/// \cond DoxygenSuppress
#define INSTANTIATE_SIMILARITY( T )                         \
  template class VITAL_EXPORT similarity_< T >;         \
  template VITAL_EXPORT std::ostream&                   \
  operator<<( std::ostream& s, const similarity_< T >& t ); \
  template VITAL_EXPORT std::istream&                   \
  operator>>( std::istream& s, similarity_< T >& t )

INSTANTIATE_SIMILARITY( double );
INSTANTIATE_SIMILARITY( float );

#undef INSTANTIATE_SIMILARITY
/// \endcond

} } // end namespace vital
