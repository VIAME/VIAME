// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Header file for customized track set definitions

#ifndef KWIVER_ARROWS_CORE_TRACK_SET_IMPL_H_
#define KWIVER_ARROWS_CORE_TRACK_SET_IMPL_H_

#include "viame_image_processing_export.h"

#include <viame/core_types/track_set.h>

#include <map>
#include <unordered_map>

namespace kwiver {

namespace arrows {

namespace core {

/// A custom track set implementation that provides fast indexing by frame id
///
/// This track_set_implementation is designed to make querying tracks by frame
/// id more efficient.  The simple track set must scan every track state of
/// every track to find tracks on a given frame for each request.  This
/// implementation caches the mapping from frames to track states for faster
/// retrieval.
class VIAME_IMAGE_PROCESSING_EXPORT frame_index_track_set_impl
  : public vital::track_set_implementation
{
public:
  /// Default Constructor
  frame_index_track_set_impl();

  /// Constructor from a vector of tracks
  explicit frame_index_track_set_impl(
    const std::vector< vital::track_sptr >& tracks );

  /// Destructor
  virtual ~frame_index_track_set_impl() = default;

  /// Return the number of tracks in the set
  size_t size() const override;

  /// Return whether or not there are any tracks in the set
  bool empty() const override;

  /// Return true if the set contains a specific track
  bool contains( vital::track_sptr t ) const override;

  /// Assign a vector of track shared pointers to this container
  void set_tracks( std::vector< vital::track_sptr > const& tracks ) override;

  /// Insert a track shared pointer into this container
  //@{
  void insert( vital::track_sptr const& t ) override;
  void insert( vital::track_sptr&& t ) override;
  //@}

  /// Notify the container that a new state has been added to an existing track
  void notify_new_state( vital::track_state_sptr ts ) override;

  /// Notify the container that a state has been removed from an existing track
  void notify_removed_state( vital::track_state_sptr ts ) override;

  /// Remove a track from the set and return true if successful
  bool remove( vital::track_sptr t ) override;

  /// Return a vector of track shared pointers
  std::vector< vital::track_sptr > tracks() const override;

  /// Return the set of all frame IDs covered by these tracks
  std::set< vital::frame_id_t > all_frame_ids() const override;

  /// Return the set of all track IDs in this track set
  std::set< vital::track_id_t > all_track_ids() const override;

  /// Return the first (smallest) frame number containing tracks
  vital::frame_id_t first_frame() const override;

  /// Return the last (largest) frame number containing tracks
  vital::frame_id_t last_frame() const override;

  /// Return the track in this set with the specified id.
  vital::track_sptr const get_track( vital::track_id_t tid ) const override;

  /// Return all tracks active on a frame.
  std::vector< vital::track_sptr >
  active_tracks( vital::frame_id_t offset = -1 ) const override;

  /// Return all tracks inactive on a frame.
  std::vector< vital::track_sptr >
  inactive_tracks( vital::frame_id_t offset = -1 ) const override;

  /// Return all tracks newly initialized on the given frame.
  std::vector< vital::track_sptr >
  new_tracks( vital::frame_id_t offset = -1 ) const override;

  /// Return all tracks terminated on the given frame.
  std::vector< vital::track_sptr >
  terminated_tracks( vital::frame_id_t offset = -1 ) const override;

  /// Return the percentage of tracks successfully tracked between two frames.
  double percentage_tracked(
    vital::frame_id_t offset1 = -2,
    vital::frame_id_t offset2 = -1 ) const override;

  /// Return a vector of state data corresponding to the tracks on the given
  /// frame.
  std::vector< vital::track_state_sptr >
  frame_states( vital::frame_id_t offset = -1 ) const override;

  /// Returns all frame data as map of frame index to track_set_frame_data
  vital::track_set_frame_data_map_t
  all_frame_data() const override
  {
    return frame_data_;
  }

  /// Return the additional data associated with all tracks on the given frame
  vital::track_set_frame_data_sptr
  frame_data( vital::frame_id_t offset = -1 ) const override;

  /// Removes the frame data for the frame offset
  bool remove_frame_data( vital::frame_id_t offset ) override;

  /// Set additional frame data associated with all tracks for all frames
  bool
  set_frame_data( vital::track_set_frame_data_map_t const& fmap ) override
  {
    frame_data_ = fmap;
    return true;
  }

  /// Set additional data associated with all tracks on the given frame
  bool set_frame_data(
    vital::track_set_frame_data_sptr data,
    vital::frame_id_t offset = -1 ) override;

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
  std::unordered_map< vital::track_id_t, vital::track_sptr > all_tracks_;

  /// The mapping from frames to track states
  mutable std::map< vital::frame_id_t,
    std::set< vital::track_state_sptr > > frame_map_;
};

} // end namespace core

} // end namespace arrows

} // end namespace kwiver

#endif
