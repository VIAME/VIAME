/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

frame_index_track_set_impl
::frame_index_track_set_impl()
{
}


/// Constructor from a vector of tracks
frame_index_track_set_impl
::frame_index_track_set_impl( const std::vector< track_sptr >& tracks )
{
  for (auto const &t : tracks)
  {
    if (t)
    {
      all_tracks_.insert(std::make_pair(t->id(), t));
    }
  }
}

/// Populate frame_map_ with data from all_tracks_
void
frame_index_track_set_impl
::populate_frame_map() const
{
  frame_map_.clear();
  for(auto const& track : all_tracks_)
  {
    for(auto const& ts : *track.second)
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
  if (!t)
  {
    return false;
  }
  auto itr = all_tracks_.find(t->id());
  return itr != all_tracks_.end() && itr->second == t;
}


/// Assign a vector of track shared pointers to this container
void
frame_index_track_set_impl
::set_tracks( std::vector< vital::track_sptr > const& tracks )
{
  all_tracks_.clear();

  for (auto const &t : tracks)
  {
    if (t)
    {
      all_tracks_.insert(std::make_pair( t->id(), t));
    }
  }

  frame_map_.clear();
}


/// Insert a track shared pointer into this container
void
frame_index_track_set_impl
::insert( vital::track_sptr t )
{
  if (!t)
  {
    return;
  }
  all_tracks_.insert(std::make_pair(t->id(), t));

  if (!frame_map_.empty())
  {
    // update the frame map with the new track
    for (auto const& ts : *t)
    {
      frame_map_[ts->frame()].insert(ts);
    }
  }
}


/// Notify the container that a new state has been added to an existing track
void
frame_index_track_set_impl
::notify_new_state( vital::track_state_sptr ts )
{
  if (!frame_map_.empty())
  {
    // update the frame map with the new state
    frame_map_[ts->frame()].insert(ts);
  }
}

/// Notify the container that a state has been removed from an existing track
void
frame_index_track_set_impl
::notify_removed_state(vital::track_state_sptr ts)
{
  if (frame_map_.empty())
  {
    return;
  }

  auto fn = ts->frame();
  auto fm_it = frame_map_.find(fn);
  if (fm_it == frame_map_.end())
  {
    return;
  }

  auto &ts_set = fm_it->second;
  auto ts_it = ts_set.find(ts);
  if (ts_it != ts_set.end())
  {
    ts_set.erase(ts_it);
  }

  if (fm_it->second.empty())
  {
    //no track states for this frame so remove the frame from the map
    frame_map_.erase(fm_it);
  }
}


/// Remove a track from the set and return true if successful
bool
frame_index_track_set_impl
::remove( vital::track_sptr t )
{
  if (!t)
  {
    return false;
  }
  auto itr = all_tracks_.find(t->id());
  if ( itr == all_tracks_.end() || itr->second != t )
  {
    return false;
  }
  all_tracks_.erase(itr);

  if (!frame_map_.empty())
  {
    // remove from the frame map
    for (auto const& ts : *t)
    {
      frame_map_[ts->frame()].erase(ts);
      if (frame_map_[ts->frame()].empty())
      { // There are not track states in the frame map.  So remove the frame's
        // entry from the frame map.
        frame_map_.erase(ts->frame());
      }
    }
  }

  return true;
}


/// Return a vector of track shared pointers
std::vector< track_sptr >
frame_index_track_set_impl
::tracks() const
{
  std::vector<track_sptr> tks(all_tracks_.size());
  size_t i = 0;
  for (auto const &t : all_tracks_)
  {
    tks[i++] = t.second;
  }
  return tks;
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
    ids.insert(t.first);
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
  auto t_it = all_tracks_.find(tid);
  if (t_it != all_tracks_.end())
  {
    return t_it->second;
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
    auto &track_set = map_itr->second;
    active_tracks.reserve(track_set.size());
    for( auto const& ts : track_set)
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
    if( t.second->find(frame_number) == t.second->end() )
    {
      inactive_tracks.push_back(t.second);
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

bool track_less(const track_sptr &t1, const track_sptr &t2)
{
  return t1->id() < t2->id();
}

/// Return the percentage of tracks successfully tracked to the next frame.
double
frame_index_track_set_impl
::percentage_tracked(frame_id_t offset1, frame_id_t offset2) const
{
  // populate the frame map if empty
  populate_frame_map_on_demand();

  std::vector<track_sptr> tracks1 = this->active_tracks(offset1);
  std::sort(tracks1.begin(), tracks1.end(), track_less);
  std::vector<track_sptr> tracks2 = this->active_tracks(offset2);
  std::sort(tracks2.begin(), tracks2.end(), track_less);

  std::vector<track_sptr> isect_tracks;
  std::set_intersection(tracks1.begin(), tracks1.end(),
                        tracks2.begin(), tracks2.end(),
                        std::back_inserter(isect_tracks), track_less);

  std::vector<track_sptr> union_tracks;
  std::set_union(tracks1.begin(), tracks1.end(),
                 tracks2.begin(), tracks2.end(),
                 std::back_inserter(union_tracks), track_less);

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

/// Return the additional data associated with all tracks on the given frame
track_set_frame_data_sptr
frame_index_track_set_impl
::frame_data( frame_id_t offset ) const
{
  frame_id_t frame_number = offset_to_frame(offset);
  auto itr = frame_data_.find(frame_number);
  if ( itr != frame_data_.end() )
  {
    return itr->second;
  }
  return nullptr;
}

/// Removes the frame data for the frame offset
bool
frame_index_track_set_impl
::remove_frame_data(frame_id_t offset)
{
  frame_id_t frame_number = offset_to_frame(offset);
  auto itr = frame_data_.find(frame_number);
  if (itr != frame_data_.end())
  {
    frame_data_.erase(itr);
    return true;
  }
  return false;
}


/// Set additional data associated with all tracks on the given frame
bool
frame_index_track_set_impl
::set_frame_data( track_set_frame_data_sptr data,
                  frame_id_t offset )
{
  frame_id_t frame_number = offset_to_frame(offset);
  if ( !data )
  {
    // remove the data on the specified frame
    auto itr = frame_data_.find(frame_number);
    if ( itr == frame_data_.end() )
    {
      return false;
    }
    frame_data_.erase(itr);
  }
  else
  {
    frame_data_[frame_number] = data;
  }
  return true;
}


track_set_implementation_uptr
frame_index_track_set_impl
::clone( vital::clone_type ct ) const
{
  std::unique_ptr<frame_index_track_set_impl> the_clone =
    std::unique_ptr<frame_index_track_set_impl>(new frame_index_track_set_impl());

  // clone the track data
  for ( auto const& trk : all_tracks_ )
  {
    the_clone->all_tracks_.emplace( trk.first, trk.second->clone( ct ) );
  }

  // clone the frame data
  for ( auto const& fd : frame_data_ )
  {
    the_clone->frame_data_.emplace( fd.first, fd.second->clone() );
  }

#if __GNUC__ > 4 || __clang_major__ > 3
  return the_clone;
#else
  return std::move( the_clone );
#endif
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
