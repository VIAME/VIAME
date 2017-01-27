/*ckwg +29
 * Copyright 2014-2015 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Core Homography class definition
 */

#ifndef VITAL_HOMOGRAPHY_H_
#define VITAL_HOMOGRAPHY_H_

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vital/types/matrix.h>
#include <vital/types/vector.h>

#include <iostream>
#include <map>
#include <vector>
#include <memory>

namespace kwiver {
namespace vital {


// Forward declarations of abstract homography class
class homography;
// typedef for a homography shared pointer
typedef std::shared_ptr< homography > homography_sptr;


// ===========================================================================
// Homography Base-class
// ---------------------------------------------------------------------------

/// Abstract base homography transformation representation class
class VITAL_EXPORT homography
{
public:
  /// Destructor
  virtual ~homography() VITAL_DEFAULT_DTOR

  /// Access the type info of the underlying data
  virtual std::type_info const& data_type() const = 0;

  /// Create a clone of this homography object, returning as smart pointer
  /**
   * \return A new deep clone of this homography transformation.
   */
  virtual homography_sptr clone() const = 0;

  /// Get a double-typed copy of the underlying matrix transformation
  /**
   * \return A copy of the transformation matrix represented in the double
   *         type.
   */
  virtual Eigen::Matrix< double, 3, 3 > matrix() const = 0;

  /// Get a new \p homography that has been normalized
  /**
   * Normalized \p homography is one in which the lower-right corner
   * (index (2,2]) is 1.
   *
   * If this index is 0, the nothing is modified.
   *
   * \return New homography transformation instance.
   */
  virtual homography_sptr normalize() const = 0;

  /// Get a new \p homography that has been inverted.
  /**
   * \return New homography transformation instance.
   */
  virtual homography_sptr inverse() const = 0;

  /// Map a 2D double-type point using this homography
  /**
   * \param p Point to map against this homography
   * \return New point in the projected coordinate system.
   */
  virtual Eigen::Matrix< double, 2, 1 >
  map( Eigen::Matrix< double, 2, 1 > const& p ) const = 0;

};


// ===========================================================================
// Typed Homography
// ---------------------------------------------------------------------------

/// Representation of a matrix-based homography transformation
/**
 * This class represents a matrix based homography templated on
 * coordinate point element data type.
 *
 * \tparam T Coordinate point data type
 */
template < typename T >
class VITAL_EXPORT homography_ :
  public homography
{
public:
  typedef T value_type;
  typedef Eigen::Matrix< T, 3, 3 > matrix_t;

  /// Construct an identity homography
  homography_< T > ( );

  /// Construct from a provided transformation matrix
  /**
   * \param mat The 3x3 transformation matrix to use.
   */
  explicit
  homography_< T > ( matrix_t const & mat );

  /// Conversion Copy constructor
  /**
   * \param other The other homography whose transformation should be copied.
   */
  template < typename U >
  explicit
  homography_< T > ( homography_< U > const & other )
  : h_( other.h_.template cast< T > () )
  {
  }

  /// Construct from a generic homography
  explicit
  homography_< T > ( homography const & base );

  // ---- Abstract method definitions ----

  /// Access the type info of the underlying data
  virtual std::type_info const& data_type() const { return typeid( T ); }

  /// Create a clone of ourself as a shared pointer
  /**
   * \return A new clone of this homography transformation.
   */
  virtual homography_sptr clone() const;

  /// Get a double-typed copy of the underlying matrix transformation
  /**
   * \return A copy of the transformation matrix represented in the double
   *         type.
   */
  virtual Eigen::Matrix< double, 3, 3 > matrix() const;

  /// Get a new \p homography that has been normalized
  /**
   * Normalized homography is one in which the lower-right corner (index (2,2])
   * is 1.
   *
   * If this index is 0, the nothing is modified.
   *
   * \return New homography transformation instance.
   */
  virtual homography_sptr normalize() const;

  /// Get a new \p homography that has been inverted.
  /**
   * \throws non_invertible_matrix When the homography matrix is non-invertible.
   * \return New homography transformation instance.
   */
  virtual homography_sptr inverse() const;

  /// Map a 2D double-type point using this homography
  /**
   * \param p Point to map against this homography
   * \return New point in the projected coordinate system.
   */
  virtual Eigen::Matrix< double, 2, 1 >
  map( Eigen::Matrix< double, 2, 1 > const& p ) const;

  // ---- Member Functions ----

  /// Get the underlying matrix transformation
  /**
   * \return The reference to this homography's transformation matrix.
   */
  matrix_t& get_matrix();

  /// Get a const new copy of the underlying matrix transformation.
  matrix_t const& get_matrix() const;

  /// Map a 2D point using this homography
  /**
   * \tparam T Point vector data type
   * \param p Point to map against this homography
   * \return New point in the projected coordinate system.
   */
  Eigen::Matrix< T, 2, 1 > map_point( Eigen::Matrix< T, 2, 1 > const& p ) const;

  /// Custom multiplication operator that multiplies the underlying matrices
  /**
   * \tparam T Homography data type
   * \param rhs Right-hand-side operand homography.
   * \return New homography object whose transform is the result of
   *         \p this * \p rhs.
   */
  virtual homography_< T > operator*( homography_< T > const& rhs ) const;


protected:
  /// homography transformation matrix
  matrix_t h_;
};


// ===========================================================================
// Utility Functions
// ---------------------------------------------------------------------------

/// Output stream operator for \p homography base-class
VITAL_EXPORT std::ostream& operator<<( std::ostream& s, homography const& h );

/// homography_<T> output stream operator
template < typename T >
VITAL_EXPORT std::ostream& operator<<( std::ostream& s, homography_< T > const& h );


} } // end namespace vital

#endif // VITAL_HOMOGRAPHY_H_
