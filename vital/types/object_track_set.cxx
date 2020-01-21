/*ckwg +29
 * Copyright 2013-2017, 2019 by Kitware, Inc.
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
