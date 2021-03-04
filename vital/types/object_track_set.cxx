// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of \link kwiver::vital::track_set track_set \endlink
 *        member functions
 */

#include "object_track_set.h"

#include <limits>

namespace kwiver {
namespace vital {

typedef std::unique_ptr< track_set_implementation > tsi_uptr;

/// Default Constructor
object_track_set
::object_track_set()
  : track_set( tsi_uptr( new simple_track_set_implementation ) )
{
}

/// Constructor specifying the implementation
object_track_set
::object_track_set( std::unique_ptr<track_set_implementation> impl )
  : track_set( std::move( impl ) )
{
}

/// Constructor from a vector of tracks
object_track_set
::object_track_set( std::vector< track_sptr > const& tracks )
  : track_set( tsi_uptr( new simple_track_set_implementation( tracks ) ) )
{
}

/// Clone the track state (polymorphic copy constructor)
track_state_sptr
object_track_state::clone ( clone_type ct ) const
{
  if ( ct == clone_type::DEEP )
  {
    auto new_detection =
      ( this->detection_ ? this->detection_->clone() : nullptr );

    auto copy = std::make_shared< object_track_state >(
      this->frame(), this->time(), std::move( new_detection ) );

    copy->set_image_point( this->image_point_ );
    copy->set_track_point( this->track_point_ );

    return std::move( copy );
  }
  else
  {
    return std::make_shared< object_track_state >( *this );
  }
}

} } // end namespace vital
