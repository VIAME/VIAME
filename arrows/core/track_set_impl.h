/*ckwg +29
 * Copyright 2017, 2019 by Kitware, Inc.
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
 * \brief Header file for customized track set definitions
 */

#ifndef KWIVER_ARROWS_CORE_TRACK_SET_IMPL_H_
#define KWIVER_ARROWS_CORE_TRACK_SET_IMPL_H_


#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/track_set.h>

#include <map>
#include <unordered_map>

namespace kwiver {
namespace arrows {
namespace core {

/// A custom track set implementation that provides fast indexing by frame id
/**
 * This track_set_implementation is designed to make querying tracks by frame
 * id more efficient.  The simple track set must scan every track state of
 * every track to find tracks on a given frame for each request.  This
 * implementation caches the mapping from frames to track states for faster
 * retrieval.
 */
class KWIVER_ALGO_CORE_EXPORT frame_index_track_set_impl
  : public vital::track_set_implementation
{
public:
  /// Default Constructor
  frame_index_track_set_impl();

  /// Constructor from a vector of tracks
  explicit frame_index_track_set_impl( const std::vector< vital::track_sptr >& tracks );

  /// Destructor
  virtual ~frame_index_track_set_impl() = default;

  /// Return the number of tracks in the set
  virtual size_t size() const;

  /// Return whether or not there are any tracks in the set
  virtual bool empty() const;

  /// Return true if the set contains a specific track
  virtual bool contains( vital::track_sptr t ) const;

  /// Assign a vector of track shared pointers to this container
  virtual void set_tracks( std::vector< vital::track_sptr > const& tracks );

  /// Insert a track shared pointer into this container
  virtual void insert( vital::track_sptr t );

  /// Notify the container that a new state has been added to an existing track
  virtual void notify_new_state( vital::track_state_sptr ts );

  /// Notify the container that a state has been removed from an existing track
  virtual void notify_removed_state(vital::track_state_sptr ts);

  /// Remove a track from the set and return true if successful
  virtual bool remove( vital::track_sptr t );

  /// Return a vector of track shared pointers
  virtual std::vector< vital::track_sptr > tracks() const;

  /// Return the set of all frame IDs covered by these tracks
  virtual std::set< vital::frame_id_t > all_frame_ids() const;

  /// Return the set of all track IDs in this track set
  virtual std::set< vital::track_id_t > all_track_ids() const;

  /// Return the first (smallest) frame number containing tracks
  virtual vital::frame_id_t first_frame() const;

  /// Return the last (largest) frame number containing tracks
  virtual vital::frame_id_t last_frame() const;

  /// Return the track in this set with the specified id.
  virtual vital::track_sptr const get_track( vital::track_id_t tid ) const;

  /// Return all tracks active on a frame.
  virtual std::vector< vital::track_sptr>
  active_tracks( vital::frame_id_t offset = -1 ) const;

  /// Return all tracks inactive on a frame.
  virtual std::vector< vital::track_sptr >
  inactive_tracks( vital::frame_id_t offset = -1 ) const;

  /// Return all tracks newly initialized on the given frame.
  virtual std::vector< vital::track_sptr >
  new_tracks( vital::frame_id_t offset = -1 ) const;

  /// Return all tracks terminated on the given frame.
  virtual std::vector< vital::track_sptr >
  terminated_tracks( vital::frame_id_t offset = -1 ) const;

  /// Return the percentage of tracks successfully tracked between two frames.
  virtual double percentage_tracked( vital::frame_id_t offset1 = -2,
                                     vital::frame_id_t offset2 = -1 ) const;

  /// Return a vector of state data corresponding to the tracks on the given frame.
  virtual std::vector< vital::track_state_sptr >
  frame_states( vital::frame_id_t offset = -1 ) const;

  /// Returns all frame data as map of frame index to track_set_frame_data
  virtual vital::track_set_frame_data_map_t all_frame_data() const
  {
    return frame_data_;
  }

  /// Return the additional data associated with all tracks on the given frame
  virtual vital::track_set_frame_data_sptr
  frame_data( vital::frame_id_t offset = -1 ) const;

  /// Removes the frame data for the frame offset
  virtual bool remove_frame_data(vital::frame_id_t offset);

  /// Set additional frame data associated with all tracks for all frames
  virtual bool set_frame_data( vital::track_set_frame_data_map_t const& fmap )
  {
    frame_data_ = fmap;
    return true;
  }

  /// Set additional data associated with all tracks on the given frame
  virtual bool set_frame_data( vital::track_set_frame_data_sptr data,
                               vital::frame_id_t offset = -1 );

  vital::track_set_implementation_uptr clone(
    vital::clone_type = vital::clone_type::DEEP ) const override;

protected:
  /// Populate frame_map_ with data from all_tracks_
  void populate_frame_map() const;

  /// Populate frame_map_ only if it is empty
  void populate_frame_map_on_demand() const;

  /// The frame data map
  vital::track_set_frame_data_map_t frame_data_;

private:
  /// The vector of all tracks
  std::unordered_map<vital::track_id_t, vital::track_sptr > all_tracks_;

  /// The mapping from frames to track states
  mutable std::map<vital::frame_id_t, std::set<vital::track_state_sptr> > frame_map_;
};



} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
