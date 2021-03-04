// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for \link kwiver::vital::similarity_ similarity_<T> \endlink class
 *        for similarity transformations
 */

#ifndef VITAL_SIMILARITY_H_
#define VITAL_SIMILARITY_H_

#include <iostream>

#include <vital/types/matrix.h>
#include <vital/types/vector.h>
#include <vital/types/rotation.h>
#include <vital/logger/logger.h>

namespace kwiver {
namespace vital {

/// A representation of a 3D similarity transformation.
/**
 * A similarity transformation is one that includes a scaling, rotation,
 * and translation
 */
template < typename T >
class VITAL_EXPORT similarity_
{
public:
  /// Default Constructor
  similarity_< T > ( )
  : scale_( 1 ),
    rot_(),
    trans_( 0, 0, 0 )
  {}

  /// Copy Constructor from another type
  template < typename U >
  explicit similarity_< T > ( const similarity_< U > &other )
  : scale_( static_cast< T > ( other.scale() ) ),
    rot_( static_cast< rotation_< T > > ( other.rotation() ) ),
    trans_( other.translation().template cast< T > () )
  {}

  /// Constructor - from scale, rotatation, and translation
  /**
   * \param s the scale factor
   * \param r the rotation
   * \param t the translation vector
   */
  similarity_< T > ( const T &s, const rotation_< T > &r,
                     const Eigen::Matrix< T, 3, 1 > &t )
  : scale_( s ),
    rot_( r ),
    trans_( t )
  {}

  /// Constructor - from a matrix
  /**
   * requires a matrix which represents a similarity tranformation
   * in homogeneous coordinates
   * \param mat Transform in matrix form to initialize from.
   */
  explicit similarity_< T >( const Eigen::Matrix< T, 4, 4 > &mat );

  /// Convert to a 4x4 matrix
  Eigen::Matrix< T, 4, 4 > matrix() const;

  /// Return scale factor
  const T& scale() const { return scale_; }

  /// Return the rotation
  const rotation_< T >& rotation() const { return rot_; }

  /// Return the translation vector
  const Eigen::Matrix< T, 3, 1 >& translation() const { return trans_; }

  /// Compute the inverse similarity
  similarity_< T > inverse() const
  {
    T inv_scale = T( 1 ) / scale_;

    rotation_< T > inv_rot( rot_.inverse() );
    return similarity_< T > ( inv_scale, inv_rot, -inv_scale * ( inv_rot * trans_ ) );
  }

  /// Compose two similarities
  /**
   * \param rhs other similarity to compose with.
   */
  similarity_< T > operator*( const similarity_< T >& rhs ) const;

  /// Transform a vector
  /**
   * \note for a large number of vectors, it is more efficient to
   *       create a transform matrix and use matrix multiplication
   * \param rhs vector to transform.
   */
  Eigen::Matrix< T, 3, 1 > operator*( const Eigen::Matrix< T, 3, 1 >& rhs ) const;

  /// Equality operator
  inline bool operator==( const similarity_< T >& rhs ) const
  {
    return this->scale_ == rhs.scale_ &&
           this->rot_   == rhs.rot_   &&
           this->trans_ == rhs.trans_;
  }

  /// Inequality operator
  inline bool operator!=( const similarity_< T >& rhs ) const
  {
    return ! ( *this == rhs );
  }

protected:
  /// scale factor
  T scale_;
  /// rotation
  rotation_< T > rot_;
  /// translation
  Eigen::Matrix< T, 3, 1 > trans_;

  kwiver::vital::logger_handle_t m_logger;

};

/// \cond DoxygenSuppress
typedef similarity_< double > similarity_d;
typedef similarity_< float > similarity_f;
/// \endcond

/// output stream operator for a similarity transformation
template < typename T >
VITAL_EXPORT std::ostream&  operator<<( std::ostream& s, const similarity_< T >& t );

/// input stream operator for a similarity transformation
template < typename T >
VITAL_EXPORT std::istream&  operator>>( std::istream& s, similarity_< T >& t );

} } // end namespace vital

#endif // VITAL_SIMILARITY_H_
