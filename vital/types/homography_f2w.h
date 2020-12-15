// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Frame to World Homography definition
 */

#ifndef VITAL_HOMOGRAPHY_F2W_H
#define VITAL_HOMOGRAPHY_F2W_H

#include <vital/types/homography.h>

namespace kwiver {
namespace vital {

class VITAL_EXPORT f2w_homography
{
public:
  /// Construct an identity homography for the given frame
  explicit f2w_homography( frame_id_t const frame_id );

  /// Construct given an existing homography
  /**
   * The given homography sptr is cloned into this object so we retain a unique
   * copy.
   */
  f2w_homography( homography_sptr const &h, frame_id_t const frame_id );

  /// Copy Constructor
  f2w_homography( f2w_homography const &h );

  virtual ~f2w_homography() = default;

  /// Get the homography transformation
  virtual homography_sptr homography() const;

  /// Get the frame identifier
  virtual frame_id_t frame_id() const;

protected:
  /// Homography transformation
  homography_sptr h_;

  /// Frame identifier
  frame_id_t frame_id_;
};

} } // end vital namespace

#endif // VITAL_HOMOGRAPHY_F2W_H
