/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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

#include "track_set.h"

#include <limits>

#include <vital/vital_foreach.h>

namespace kwiver {
namespace vital {

/// Return the number of tracks in the set
size_t
track_set_implementation
::size() const
{
  return this->tracks().size();
}


/// Return whether or not there are any tracks in the set
bool
track_set_implementation
::empty() const
{
  return this->tracks().empty();
}


/// Return the set of all frame IDs covered by these tracks
std::set<frame_id_t>
track_set_implementation
::all_frame_ids() const
{
  std::set<frame_id_t> ids;
  const std::vector<track_sptr> all_tracks = this->tracks();

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    std::set<frame_id_t> t_ids = t->all_frame_ids();
    ids.insert(t_ids.begin(), t_ids.end());
  }

  return ids;
}


/// Return the set of all track IDs in this track set
std::set<track_id_t>
track_set_implementation
::all_track_ids() const
{
  std::set<track_id_t> ids;
  const std::vector<track_sptr> all_tracks = this->tracks();

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    ids.insert(t->id());
  }

  return ids;
}


/// Return the last (largest) frame number containing tracks
frame_id_t
track_set_implementation
::last_frame() const
{
  frame_id_t last_frame = 0;
  const std::vector<track_sptr> all_tracks = this->tracks();

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->last_frame() > last_frame )
    {
      last_frame = t->last_frame();
    }
  }

  // If there was no frame intersection, then last_frame is not modified, and
  // we are returning the default of 0.
  return last_frame;
}


/// Return the first (smallest) frame number containing tracks
frame_id_t
track_set_implementation
::first_frame() const
{
  frame_id_t first_frame = std::numeric_limits<frame_id_t>::max();
  const std::vector<track_sptr> all_tracks = this->tracks();
  bool intersects_frame = false;

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->first_frame() < first_frame )
    {
      intersects_frame = true;
      first_frame = t->first_frame();
    }
  }

  if( intersects_frame )
  {
    return first_frame;
  }
  else
  {
    return 0;
  }
}


/// Return the track in the set with the specified id.
track_sptr const
track_set_implementation
::get_track(track_id_t tid) const
{
  const std::vector<track_sptr> all_tracks = this->tracks();

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->id() == tid )
    {
      return t;
    }
  }

  return track_sptr();
}

/// Return all tracks active on a frame.
std::vector< track_sptr >
track_set_implementation
::active_tracks(frame_id_t offset) const
{
  frame_id_t frame_number = offset_to_frame(offset);
  const std::vector<track_sptr> all_tracks = this->tracks();
  std::vector<track_sptr> active_tracks;

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->find(frame_number) != t->end() )
    {
      active_tracks.push_back(t);
    }
  }

  return active_tracks;
}


/// Return all tracks active on a frame.
std::vector< track_sptr >
track_set_implementation
::inactive_tracks(frame_id_t offset) const
{
  frame_id_t frame_number = offset_to_frame(offset);
  const std::vector<track_sptr> all_tracks = this->tracks();
  std::vector<track_sptr> inactive_tracks;

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->find(frame_number) == t->end() )
    {
      inactive_tracks.push_back(t);
    }
  }

  return inactive_tracks;
}


/// Return all new tracks on a given frame.
std::vector< track_sptr >
track_set_implementation
::new_tracks(frame_id_t offset) const
{
  frame_id_t frame_number = offset_to_frame(offset);
  const std::vector<track_sptr> all_tracks = this->tracks();
  std::vector<track_sptr> new_tracks;

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->first_frame() == frame_number )
    {
      new_tracks.push_back(t);
    }
  }

  return new_tracks;
}


/// Return all new tracks on a given frame.
std::vector< track_sptr >
track_set_implementation
::terminated_tracks(frame_id_t offset) const
{
  frame_id_t frame_number = offset_to_frame(offset);
  const std::vector<track_sptr> all_tracks = this->tracks();
  std::vector<track_sptr> terminated_tracks;

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    if( t->last_frame() == frame_number )
    {
      terminated_tracks.push_back(t);
    }
  }

  return terminated_tracks;
}


/// Return the percentage of tracks successfully tracked to the next frame.
double
track_set_implementation
::percentage_tracked(frame_id_t offset1, frame_id_t offset2) const
{
  const frame_id_t frame_number1 = offset_to_frame(offset1);
  const frame_id_t frame_number2 = offset_to_frame(offset2);

  const std::vector<track_sptr> all_tracks = this->tracks();
  unsigned total_tracks = 0, tracks_both = 0;

  VITAL_FOREACH( track_sptr t , all_tracks)
  {
    const bool found_on_f1 = t->find(frame_number1) != t->end();
    const bool found_on_f2 = t->find(frame_number2) != t->end();

    total_tracks += (found_on_f1 || found_on_f2 ? 1 : 0);
    tracks_both += (found_on_f1 && found_on_f2 ? 1 : 0);
  }

  if( total_tracks == 0 )
  {
    return 0.0;
  }
  return static_cast<double>(tracks_both) / total_tracks;
}


/// Return a vector of state data corresponding to the tracks on the given frame.
std::vector<track_state_sptr>
track_set_implementation
::frame_states( frame_id_t offset ) const
{
  const frame_id_t frame_number = offset_to_frame(offset);
  const std::vector<track_sptr> all_tracks = this->tracks();
  std::vector<track_state_sptr> vdata;

  VITAL_FOREACH( track_sptr t, all_tracks)
  {
    track::history_const_itr itr = t->find(frame_number);
    if( itr != t->end() )
    {
      vdata.push_back(*itr);
    }
  }

  return vdata;
}


/// Convert an offset number to an absolute frame number
frame_id_t
track_set_implementation
::offset_to_frame(frame_id_t offset) const
{
  if( offset >= 0 )
  {
    return offset;
  }

  frame_id_t frame_number = this->last_frame() + 1;

  if( -offset <= frame_number )
  {
    frame_number += offset;
  }
  return frame_number;
}



/// Default Constructor
track_set
::track_set()
  : impl_(new simple_track_set_implementation)
{
}


/// Constructor specifying the implementation
track_set
::track_set(std::unique_ptr<track_set_implementation> impl)
  : impl_(std::move(impl))
{
}


/// Constructor from a vector of tracks
track_set
::track_set(std::vector< track_sptr > const& tracks)
  : impl_(new simple_track_set_implementation(tracks))
{
}



} } // end namespace vital
