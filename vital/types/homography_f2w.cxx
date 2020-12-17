// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Frame to World Homography implementation
 */

#include "homography_f2w.h"

namespace kwiver {
namespace vital {

/// Construct an identity homography for the given frame
f2w_homography
::f2w_homography( frame_id_t const frame_id )
  : h_( homography_sptr( new homography_<double>() ) ),
    frame_id_( frame_id )
{
}

/// Construct given an existing homography
f2w_homography
::f2w_homography( homography_sptr const &h, frame_id_t const frame_id )
  : h_( std::static_pointer_cast< vital::homography >( h->clone() ) ),
    frame_id_( frame_id )
{
}

/// Copy Constructor
f2w_homography
::f2w_homography( f2w_homography const &h )
  : h_( std::static_pointer_cast< vital::homography >( h.h_->clone() ) ),
    frame_id_( h.frame_id_ )
{
}

/// Get the homography transformation
homography_sptr
f2w_homography
::homography() const
{
  return this->h_;
}

/// Get the frame identifier
frame_id_t
f2w_homography
::frame_id() const
{
  return this->frame_id_;
}

} } // end vital namespace
