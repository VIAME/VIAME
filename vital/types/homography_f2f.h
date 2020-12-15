// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Frame to Frame Homography definition
 */

#ifndef VITAL_HOMOGRAPHY_F2F_H
#define VITAL_HOMOGRAPHY_F2F_H

#include <vital/types/homography.h>

namespace kwiver {
namespace vital {

class VITAL_EXPORT f2f_homography
{
public:
  /// Construct an identity homography for the given frame
  /**
   * \param frame_id
   */
  explicit f2f_homography( frame_id_t const frame_id );

  /// Construct a frame to frame homography using a matrix
  /**
   * \param h
   * \param from_id
   * \param to_id
   * \tparam T Data type for the underlying homography transformation
   */
  template < typename T >
  explicit f2f_homography( Eigen::Matrix< T, 3, 3 > const& h,
                           frame_id_t const from_id,
                           frame_id_t const to_id )
    : h_( homography_sptr( new homography_< T > ( h ) ) ),
      from_id_( from_id ),
      to_id_( to_id )
  { }

  /// Construct a frame to frame homography given an existing transform
  /**
   * The given homography sptr is cloned into this object so we retain a unique
   * copy.
   *
   * \param h
   * \param from_id
   * \param to_id
   */
  explicit f2f_homography( homography_sptr const& h,
                           frame_id_t const       from_id,
                           frame_id_t const       to_id );

  /// Copy constructor
  f2f_homography( f2f_homography const& h );

  /// Destructor
  virtual ~f2f_homography() = default;

  /// Get the sptr of the contained homography transformation
  virtual homography_sptr homography() const;

  /// Frame identifier that the homography maps from.
  virtual frame_id_t from_id() const;

  /// Frame identifier that the homography maps to.
  virtual frame_id_t to_id() const;

  /// Return a new inverse \p f2f_homography instance
  /**
   * \return New \p f2f_homography instance whose transformation is inverted as
   *         well as has flipped from and to ID references.
   */
  virtual f2f_homography inverse() const;

  /// Custom f2f_homography multiplication operator for \p f2f_homography
  /**
   * \throws invalid_matrix_operation
   *    When \p this.from_id() != \p rhs.to_id() as transformed from and to IDs
   *    are undefined otherwise.
   *
   * \param rhs Right-hand-side operand homography.
   * \return New homography object whose transform is the result of
   *         \p this * \p rhs.
   */
  virtual f2f_homography operator*( f2f_homography const& rhs );

protected:
  /// Homography transformation sptr.
  homography_sptr h_;

  /// From frame identifier.
  frame_id_t from_id_;

  /// To frame identifier.
  frame_id_t to_id_;
};

/// Shared pointer for \p f2f_homography
typedef std::shared_ptr< f2f_homography > f2f_homography_sptr;

/// \p f2f_homography output stream operator
VITAL_EXPORT std::ostream& operator<<( std::ostream& s, f2f_homography const& h );

} } // end vital namespace

#endif // VITAL_HOMOGRAPHY_F2F_H
