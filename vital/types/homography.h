// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Core Homography class definition
 */

#ifndef VITAL_HOMOGRAPHY_H_
#define VITAL_HOMOGRAPHY_H_

#include <vital/types/matrix.h>
#include <vital/types/transform_2d.h>

#include <iostream>

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
class VITAL_EXPORT homography : public transform_2d
{
public:
  /// Destructor
  virtual ~homography() = default;

  /// Access the type info of the underlying data
  virtual std::type_info const& data_type() const = 0;

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
  homography_sptr inverse() const
  { return std::static_pointer_cast< homography >( this->inverse_() ); }
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
  std::type_info const& data_type() const override { return typeid( T ); }

  /// Create a clone of ourself as a shared pointer
  /**
   * \return A new clone of this homography transformation.
   */
  transform_2d_sptr clone() const override;

  /// Get a double-typed copy of the underlying matrix transformation
  /**
   * \return A copy of the transformation matrix represented in the double
   *         type.
   */
  Eigen::Matrix< double, 3, 3 > matrix() const override;

  /// Get a new \p homography that has been normalized
  /**
   * Normalized homography is one in which the lower-right corner (index (2,2])
   * is 1.
   *
   * If this index is 0, the nothing is modified.
   *
   * \return New homography transformation instance.
   */
  homography_sptr normalize() const override;

  /// Get a new \p homography that has been inverted.
  /**
   * \throws non_invertible
   *   When the homography matrix is non-invertible.
   * \return New homography transformation instance.
   */
  homography_sptr inverse() const;

  /// Map a 2D double-type point using this homography
  /**
   * \param p Point to map against this homography
   * \return New point in the projected coordinate system.
   */
  vector_2d map( vector_2d const& p ) const override;

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
  transform_2d_sptr inverse_() const { return this->inverse(); }

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
