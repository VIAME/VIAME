/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief Implementation of customized track set implementations
 */

#include "track_set_impl.h"

#include <algorithm>
#include <iterator>
#include <limits>

namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;


/// Constructor from a vector of tracks
frame_index_track_set_impl
::frame_index_track_set_impl( const std::vector< track_sptr >& tracks )
  : all_tracks_( tracks )
{
}


/// Populate frame_map_ with data from all_tracks_
void
frame_index_track_set_impl
::populate_frame_map() const
{
  frame_map_.clear();
  for(auto const& track : all_tracks_)
  {
    for(auto const& ts : *track)
    {
      frame_map_[ts->frame()].insert(ts);
    }
  }
}


/// Populate frame_map_ only if it is empty
void
frame_index_track_set_impl
::populate_frame_map_on_demand() const
{
  if(frame_map_.empty() && !all_tracks_.empty())
  {
    populate_frame_map();
  }
}


/// Return the number of tracks in the set
size_t
frame_index_track_set_impl
::size() const
{
  return this->all_tracks_.size();
}


/// Return whether or not there are any tracks in the set
bool
frame_index_track_set_impl
::empty() const
{
  return this->all_tracks_.empty();
}


/// Return true if the set contains a specific track
bool
frame_index_track_set_impl
::contains( vital::track_sptr t ) const
{
  return std::find(all_tracks_.begin(), all_tracks_.end(), t) != all_tracks_.end();
}


/// Assign a vector of track shared pointers to this container
void
frame_index_track_set_impl
::set_tracks( std::vector< vital::track_sptr > const& tracks )
{
  all_tracks_ = tracks;
  frame_map_.clear();
}


/// Insert a track shared pointer into this container
void
frame_index_track_set_impl
::insert( vital::track_sptr t )
{
  all_tracks_.push_back( t );

  // update the frame map with the new track
  for(auto const& ts : *t)
  {
    frame_map_[ts->frame()].insert(ts);
  }
}


/// Notify the container that a new state has been added to an existing track
void
frame_index_track_set_impl
::notify_new_state( vital::track_state_sptr ts )
{
  // update the frame map with the new state
  frame_map_[ts->frame()].insert(ts);
}


/// Remove a track from the set and return true if successful
bool
frame_index_track_set_impl
::remove( vital::track_sptr t )
{
  auto itr = std::find(all_tracks_.begin(), all_tracks_.end(), t);
  if ( itr == all_tracks_.end() )
  {
    return false;
  }
  all_tracks_.erase(itr);

  // remove from the frame map
  for(auto const& ts : *t)
  {
    frame_map_[ts->frame()].erase(ts);
  }

  return true;
}


/// Return a vector of track shared pointers
std::vector< track_sptr >
frame_index_track_set_impl
::tracks() const
{
  return all_tracks_;
}


/// Return the set of all frame IDs covered by these tracks
std::set<frame_id_t>
frame_index_track_set_impl
::all_frame_ids() const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::set<frame_id_t> ids;
  // extract all the keys from frame_map_
  for(auto const& fmi : frame_map_)
  {
    ids.insert(fmi.first);
  }
  return ids;
}


/// Return the set of all track IDs in this track set
std::set<track_id_t>
frame_index_track_set_impl
::all_track_ids() const
{
  std::set<track_id_t> ids;
  for( auto const& t : all_tracks_)
  {
    ids.insert(t->id());
  }
  return ids;
}


/// Return the last (largest) frame number containing tracks
frame_id_t
frame_index_track_set_impl
::last_frame() const
{
  if( frame_map_.empty() )
  {
    return track_set_implementation::last_frame();
  }
  return frame_map_.rbegin()->first;
}


/// Return the first (smallest) frame number containing tracks
frame_id_t
frame_index_track_set_impl
::first_frame() const
{
  if( frame_map_.empty() )
  {
    return track_set_implementation::first_frame();
  }
  return frame_map_.begin()->first;
}


/// Return the track in the set with the specified id.
track_sptr const
frame_index_track_set_impl
::get_track(track_id_t tid) const
{
  const std::vector<track_sptr> all_tracks = this->tracks();

  for( auto const& t : all_tracks_)
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
frame_index_track_set_impl
::active_tracks(frame_id_t offset) const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::vector<track_sptr> active_tracks;
  frame_id_t frame_number = offset_to_frame(offset);
  auto const& map_itr = frame_map_.find(frame_number);
  if( map_itr != frame_map_.end() )
  {
    for( auto const& ts : map_itr->second )
    {
      active_tracks.push_back(ts->track());
    }
  }
  return active_tracks;
}


/// Return all tracks not active on a frame.
std::vector< track_sptr >
frame_index_track_set_impl
::inactive_tracks(frame_id_t offset) const
{
  std::vector<track_sptr> inactive_tracks;
  frame_id_t frame_number = offset_to_frame(offset);
  // TODO consider computing this as a set difference between all_tracks_
  // and active_tracks()
  for( auto const& t : all_tracks_)
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
frame_index_track_set_impl
::new_tracks(frame_id_t offset) const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::vector<track_sptr> new_tracks;
  frame_id_t frame_number = offset_to_frame(offset);
  auto const& map_itr = frame_map_.find(frame_number);
  if( map_itr != frame_map_.end() )
  {
    for( auto const& ts : map_itr->second )
    {
      auto t = ts->track();
      if( t->first_frame() == frame_number )
      {
        new_tracks.push_back(ts->track());
      }
    }
  }
  return new_tracks;
}


/// Return all terminated tracks on a given frame.
std::vector< track_sptr >
frame_index_track_set_impl
::terminated_tracks(frame_id_t offset) const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::vector<track_sptr> terminated_tracks;
  frame_id_t frame_number = offset_to_frame(offset);
  auto const& map_itr = frame_map_.find(frame_number);
  if( map_itr != frame_map_.end() )
  {
    for( auto const& ts : map_itr->second )
    {
      auto t = ts->track();
      if( t->last_frame() == frame_number )
      {
        terminated_tracks.push_back(ts->track());
      }
    }
  }
  return terminated_tracks;
}


/// Return the percentage of tracks successfully tracked to the next frame.
double
frame_index_track_set_impl
::percentage_tracked(frame_id_t offset1, frame_id_t offset2) const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::vector<track_sptr> tracks1 = this->active_tracks(offset1);
  std::sort(tracks1.begin(), tracks1.end());
  std::vector<track_sptr> tracks2 = this->active_tracks(offset2);
  std::sort(tracks2.begin(), tracks2.end());

  std::vector<track_sptr> isect_tracks;
  std::set_intersection(tracks1.begin(), tracks1.end(),
                        tracks2.begin(), tracks2.end(),
                        std::back_inserter(isect_tracks));

  std::vector<track_sptr> union_tracks;
  std::set_union(tracks1.begin(), tracks1.end(),
                 tracks2.begin(), tracks2.end(),
                 std::back_inserter(union_tracks));

  if( union_tracks.empty() )
  {
    return 0.0;
  }
  return static_cast<double>(isect_tracks.size()) / union_tracks.size();
}


/// Return a vector of state data corresponding to the tracks on the given frame.
std::vector<track_state_sptr>
frame_index_track_set_impl
::frame_states( frame_id_t offset ) const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::vector<track_state_sptr> vdata;
  frame_id_t frame_number = offset_to_frame(offset);
  auto const& map_itr = frame_map_.find(frame_number);
  if( map_itr != frame_map_.end() )
  {
    for( auto const& ts : map_itr->second )
    {
      vdata.push_back(ts);
    }
  }
  return vdata;
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
