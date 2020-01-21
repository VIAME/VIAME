/*ckwg +29
 * Copyright 2013-2019 by Kitware, Inc.
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
 * \brief Header file for an abstract \link kwiver::vital::track_set track_set
 *        \endlink and a concrete \link kwiver::vital::simple_track_set
 *        simple_track_set \endlink
 */

#ifndef VITAL_TRACK_SET_H_
#define VITAL_TRACK_SET_H_

#include "track.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vector>
#include <set>
#include <map>
#include <memory>

namespace kwiver {
namespace vital {

class track_set;
/// Shared pointer for base track_set type
typedef std::shared_ptr< track_set > track_set_sptr;

class track_set_interface;
typedef std::shared_ptr< track_set_interface > track_set_interface_sptr;
typedef std::unique_ptr< track_set_interface > track_set_interface_uptr;

class track_set_frame_data;
typedef std::shared_ptr<track_set_frame_data> track_set_frame_data_sptr;
typedef std::map<frame_id_t, track_set_frame_data_sptr> track_set_frame_data_map_t;


// ------------------------------------------------------------------
/// Empty base class for data associated a frame in the track_set
class VITAL_EXPORT track_set_frame_data
{
public:
  // Dynamic copy constructor
  virtual track_set_frame_data_sptr clone() const = 0;

protected:
  virtual ~track_set_frame_data() = default;
};


// ------------------------------------------------------------------
/// Abstract interface for a collection of tracks
class VITAL_EXPORT track_set_interface
{
public:
  /// Destructor
  virtual ~track_set_interface() = default;

  /// Return the number of tracks in the set
  virtual size_t size() const = 0;

  /// Return whether or not there are any tracks in the set
  virtual bool empty() const = 0;

  /// Return true if the set contains a specific track
  virtual bool contains( track_sptr t ) const = 0;

  /// Assign a vector of track shared pointers to this container
  virtual void set_tracks( std::vector< track_sptr > const& tracks ) = 0;

  /// Insert a track shared pointer into this container
  virtual void insert( track_sptr t ) = 0;

  /// Notify the container that a new state has been added to an existing track
  /**
   * Some containers need to know if an existing track was extended with new
   * states.  This function should be called after calling \c t->append(ts) or
   * \c t->insert(ts) if \c t is already a member of this set.
   */
  virtual void notify_new_state( track_state_sptr ts ) = 0;

  /// Notify the container that a track state has been removed
  /**
   * Some containers need to know if a track was removed from them so that
   * they can maintain an internal registry of states in them.  This function
   * should be called after t->remove(history_const_itr)
  */
  virtual void notify_removed_state(track_state_sptr ts) = 0;

  /// Remove a track from the set and return true if successful
  virtual bool remove( track_sptr t ) = 0;

  /// Merge the pair of tracks \p t1 and \p t2, if possible
  /**
   * Try to merge \p t1 into \p t2 if both tracks are found in this set.
   * Merging copies the track states from t1 into t2 and is only allowed if
   * the tracks do not overlap temporally.
   *
   * Internally this uses \c t2->append(*t1) However, it also allows the
   * track set to update its internal bookkeeping about which tracks it
   * contains and on which frames.
   *
   * \note if successful \p t1 is emptied and removed from the set.
   * However, the empty \p t1 is left with a track_data_redirect pointing
   * to \p t2 in case other track matches still refer to \p t1.
   *
   * \param t1  The track to merge from
   * \param t2  The track to merge into
   * \returns   True if the merge is sucessful
   */
  virtual bool merge_tracks( track_sptr t1, track_sptr t2 ) = 0;

  /// Return a vector of track shared pointers
  virtual std::vector< track_sptr > tracks() const = 0;

  /// Return the set of all frame IDs covered by these tracks
  virtual std::set< frame_id_t > all_frame_ids() const = 0;

  /// Return the set of all track IDs in this track set
  virtual std::set< track_id_t > all_track_ids() const = 0;

  /// Return the first (smallest) frame number containing tracks
  /**
   * If there are no tracks in this set, or no tracks have states, this returns
   * 0.
   */
  virtual frame_id_t first_frame() const = 0;

  /// Return the last (largest) frame number containing tracks
  /**
   * If there are no tracks in this set, or no tracks have states, this returns
   * 0.
   */
  virtual frame_id_t last_frame() const = 0;

  /// Return the track in this set with the specified id.
  /**
   * An empty pointer will be returned if the track cannot be found.
   *
   * \param [in] tid track identifier for the desired track.
   *
   * \returns a pointer to the track with the given id.
   */
  virtual track_sptr const get_track( track_id_t tid ) const = 0;

  /// Return all tracks active on a frame.
  /**
   * Active tracks are any tracks which contain a state on the target frame.
   *
   * \param [in] offset the frame offset for selecting the active frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a vector of tracks that is the subset of tracks that are active.
   */
  virtual std::vector< track_sptr> active_tracks( frame_id_t offset = -1 ) const = 0;

  /// Return a set of track ids corresponding to the tracks on the given frame.
  /**
  * \param [in] offset the frame offset for selecting the target frame.
  *                    Positive number are absolute frame numbers while
  *                    negative numbers are relative to the last frame.  For
  *                    example, offset of -1 refers to the last frame and is
  *                    the default.
  *
  * \returns a set for all tracks ids on the given frame.
  */
  virtual std::set<track_id_t> active_track_ids(frame_id_t offset = -1) const = 0;

  /// return the number of active tracks on a frame
  /**
  * Active tracks are any tracks which contain a state on the target frame.
  *
  * \param [in] offset the frame offset for selecting the active frame.
  *                    Positive number are absolute frame numbers while
  *                    negative numbers are relative to the last frame.  For
  *                    example, offset of -1 refers to the last frame and is
  *                    the default.
  *
  * \returns the number of tracks that are active for the given offset
  */
  virtual size_t num_active_tracks(frame_id_t offset = -1) const = 0;

  /// Return all tracks inactive on a frame.
  /**
   * Inactive tracks are any tracks which do not contain a state on the target frame.
   *
   * \param [in] offset the frame offset for selecting the active frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a vector of tracks that is the subset of tracks that are inactive.
   */
  virtual std::vector< track_sptr > inactive_tracks( frame_id_t offset = -1 ) const = 0;

  /// Return all tracks newly initialized on the given frame.
  /**
   * New tracks include any tracks with a first track state on the target frame.
   *
   * \param[in] offset the frame offset for selecting the active frame.
   *                   Positive number are absolute frame numbers while
   *                   negative numbers are relative to the last frame.  For
   *                   example, offset of -1 refers to the last frame and is
   *                   the default.
   *
   * \returns a vector of tracks containing all new tracks for the given frame.
   */
  virtual std::vector< track_sptr > new_tracks( frame_id_t offset = -1 ) const = 0;

  /// Return all tracks terminated on the given frame.
  /**
   * Terminated tracks include any tracks with a last track state on the frame.
   *
   * \param[in] offset the frame offset for selecting the active frame.
   *                   Positive number are absolute frame numbers while
   *                   negative numbers are relative to the last frame.  For
   *                   example, offset of -1 refers to the last frame and is
   *                   the default.
   *
   * \returns a vector of tracks containing all terminated tracks for the given frame.
   */
  virtual std::vector< track_sptr > terminated_tracks( frame_id_t offset = -1 ) const = 0;

  /// Return the percentage of tracks successfully tracked between the two frames.
  /**
   * The percentage of tracks successfully tracked between frames is defined as the
   * number of tracks which have a track state on both frames, divided by the total
   * number of unique tracks which appear on both frames.
   *
   * \param[in] offset1 the frame offset for the first frame in the operation.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   * \param[in] offset2 the frame offset for the second frame in the operation.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a floating point percent value (between 0.0 and 1.0).
   */
  virtual double percentage_tracked( frame_id_t offset1 = -2, frame_id_t offset2 = -1 ) const = 0;

  /// Return a vector of state data corresponding to the tracks on the given frame.
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a vector of track_state_sptr corresponding to the tracks
   *          on this frame and in the same order as active_track(offset)
   */
  virtual std::vector<track_state_sptr> frame_states( frame_id_t offset = -1 ) const = 0;

  /// Returns all frame data as map of frame index to track_set_frame_data
  virtual track_set_frame_data_map_t all_frame_data() const = 0;

  /// Return the additional data associated with all tracks on the given frame
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a track_set_frame_data_sptr containing the frame data or nullptr
   *          if no data is available on this frame.
   */
  virtual track_set_frame_data_sptr frame_data( frame_id_t offset = -1 ) const = 0;

  /// Removes the frame data for the frame offset
  /**
  * \param [in] offset the frame offset for selecting the target frame.
  *                    Positive number are absolute frame numbers while
  *                    negative numbers are relative to the last frame.  For
  *                    example, offset of -1 refers to the last frame and is
  *                    the default.
  *
  * \returns true if the frame data was removed.  False otherwise.
  */
  virtual bool remove_frame_data(frame_id_t offset = -1) = 0;

  /// Set additional frame data associated with all tracks for all frames
  /**
   * This method sets the frame data on all frames at once using a map.
   * Any existing frame data is removed and replaced with the contents of
   * this map.  Therefore, providing an empty map removes all frame data.
   *
   * \param [in] fmap  the map of frame number to frame data to replace all
   *                   existing frame data stored with the tracks.
   *
   * \returns true if the data was successfully set.
   */
  virtual bool set_frame_data( track_set_frame_data_map_t const& fmap ) = 0;

  /// Set additional data associated with all tracks on the given frame
  /**
   * \param [in] data   the frame data object to store on this frame.
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns true if the data was successfully set.
   *
   * \note Set the \c data to \c nullptr to remove the data on a frame.
   */
  virtual bool set_frame_data( track_set_frame_data_sptr data,
                               frame_id_t offset = -1 ) = 0;

  /// Convert an offset number to an absolute frame number
  virtual frame_id_t offset_to_frame( frame_id_t offset ) const = 0;
};


// ------------------------------------------------------------------
/// A base class for the implementation of track sets
/**
 * This class provides default implementations of most functions which are
 * written by calling the tracks() function (which is still abstract here)
 * and operating on the vector of tracks.  These implementations might not
 * be the most efficent depending on how tracks are stored, but derived
 * classes can reimplement more efficient overrides as needed.
 */

class track_set_implementation;
typedef std::shared_ptr<track_set_implementation> track_set_implementation_sptr;
typedef std::unique_ptr<track_set_implementation> track_set_implementation_uptr;

class VITAL_EXPORT track_set_implementation
  : public track_set_interface
{
public:
  /// Destructor
  virtual ~track_set_implementation() = default;

  /// Return the number of tracks in the set
  virtual size_t size() const;

  /// Return whether or not there are any tracks in the set
  virtual bool empty() const;

  /// Notify the container that a new state has been added to an existing track
  virtual void notify_new_state( track_state_sptr ts );

  /// Notify the container that a state has been removed from an existing track
  virtual void notify_removed_state(track_state_sptr ts);

  /// Merge the pair of tracks \p t1 and \p t2, if possible
  virtual bool merge_tracks( track_sptr t1, track_sptr t2 );

  /// Return the set of all frame IDs covered by these tracks
  virtual std::set< frame_id_t > all_frame_ids() const;

  /// Return the set of all track IDs in this track set
  virtual std::set< track_id_t > all_track_ids() const;

  /// Return the first (smallest) frame number containing tracks
  virtual frame_id_t first_frame() const;

  /// Return the last (largest) frame number containing tracks
  virtual frame_id_t last_frame() const;

  /// Return the track in this set with the specified id.
  virtual track_sptr const get_track( track_id_t tid ) const;

  /// Return all tracks active on a frame.
  virtual std::vector< track_sptr> active_tracks( frame_id_t offset = -1 ) const;

  /// Returns all the active track ids on a frame
  virtual std::set<track_id_t> active_track_ids( frame_id_t offset = -1) const;

  /// Return the number of active tracks for a frame
  virtual size_t num_active_tracks(frame_id_t offset = -1) const;

  /// Return all tracks inactive on a frame.
  virtual std::vector< track_sptr > inactive_tracks( frame_id_t offset = -1 ) const;

  /// Return all tracks newly initialized on the given frame.
  virtual std::vector< track_sptr > new_tracks( frame_id_t offset = -1 ) const;

  /// Return all tracks terminated on the given frame.
  virtual std::vector< track_sptr > terminated_tracks( frame_id_t offset = -1 ) const;

  /// Return the percentage of tracks successfully tracked between the two frames.
  virtual double percentage_tracked( frame_id_t offset1 = -2, frame_id_t offset2 = -1 ) const;

  /// Return a vector of state data corresponding to the tracks on the given frame.
  virtual std::vector<track_state_sptr> frame_states( frame_id_t offset = -1 ) const;

  /// Convert an offset number to an absolute frame number
  virtual frame_id_t offset_to_frame( frame_id_t offset ) const;

  /// Clone this track set implementation
  virtual track_set_implementation_uptr clone(
    clone_type = clone_type::DEEP ) const = 0;
};


// ------------------------------------------------------------------
/// A collection of tracks
/**
 * This class dispatches everything to an implementation class as in the
 * bridge design pattern.  This pattern allows multiple back end implementations that
 * store and index track data in different ways.  Each back end can be combined with
 * any of the derived track_set types like feature_track_set and object_track_set.
 */
class VITAL_EXPORT track_set
 : public track_set_interface
{
public:
  /// Destructor
  virtual ~track_set() = default;

  /// Default Constructor
  /**
   * \note implementation defaults to simple_track_set_implementation
   */
  track_set();

  /// Constructor specifying the implementation
  track_set(std::unique_ptr<track_set_implementation> impl);

  /// Constructor from a vector of tracks
  /**
   * \param [in] tracks the vector of tracks to add to the set
   * \param [in] frame_data the optional map of frame_data to add
   *
   * \note implementation defaults to simple_track_set_implementation
   */
  track_set(std::vector< track_sptr > const& tracks,
            track_set_frame_data_map_t const& frame_data =
              track_set_frame_data_map_t());

  /// Return the number of tracks in the set
  virtual size_t size() const
  {
    return impl_->size();
  }

  /// Return whether or not there are any tracks in the set
  virtual bool empty() const
  {
    return impl_->empty();
  }

  /// Return true if the set contains a specific track
  virtual bool contains( track_sptr t ) const
  {
    return impl_->contains( t );
  }

  /// Assign a vector of track shared pointers to this container
  /**
   * \note this replaces any track that are already in the set
   */
  virtual void set_tracks( std::vector< track_sptr > const& tracks )
  {
    impl_->set_tracks(tracks);
  }

  /// Insert a track shared pointer into this container
  virtual void insert( track_sptr t )
  {
    impl_->insert( t );
  }

  /// Notify the container that a new state has been added to an existing track
  virtual void notify_new_state( track_state_sptr ts )
  {
    return impl_->notify_new_state( ts );
  }

  /// Notify the container that a state has been removed from an existing track
  virtual void notify_removed_state(track_state_sptr ts)
  {
    return impl_->notify_removed_state(ts);
  }

  /// Remove a track from the set and return true if successful
  virtual bool remove( track_sptr t )
  {
    return impl_->remove( t );
  }

  /// Merge the pair of tracks \p t1 and \p t2, if possible
  virtual bool merge_tracks( track_sptr t1, track_sptr t2 )
  {
    return impl_->merge_tracks( t1, t2 );
  }

  /// Return a vector of track shared pointers
  virtual std::vector< track_sptr > tracks() const
  {
    return impl_->tracks();
  }

  /// Return the set of all frame IDs covered by these tracks
  virtual std::set< frame_id_t > all_frame_ids() const
  {
    return impl_->all_frame_ids();
  }

  /// Return the set of all track IDs in this track set
  virtual std::set< track_id_t > all_track_ids() const
  {
    return impl_->all_track_ids();
  }

  /// Return the first (smallest) frame number containing tracks
  virtual frame_id_t first_frame() const
  {
    return impl_->first_frame();
  }

  /// Return the last (largest) frame number containing tracks
  virtual frame_id_t last_frame() const
  {
    return impl_->last_frame();
  }

  /// Return the track in this set with the specified id.
  virtual track_sptr const get_track( track_id_t tid ) const
  {
    return impl_->get_track(tid);
  }

  /// Return all tracks active on a frame.
  virtual std::vector< track_sptr> active_tracks( frame_id_t offset = -1 ) const
  {
    return impl_->active_tracks(offset);
  };

  virtual std::set< track_id_t > active_track_ids(frame_id_t offset = -1) const
  {
    return impl_->active_track_ids(offset);
  }

  /// Return number of active tracks for a frame.
  virtual size_t num_active_tracks(frame_id_t offset = -1) const
  {
    return impl_->num_active_tracks(offset);
  };

  /// Return all tracks inactive on a frame.
  virtual std::vector< track_sptr > inactive_tracks( frame_id_t offset = -1 ) const
  {
    return impl_->inactive_tracks(offset);
  }

  /// Return all tracks newly initialized on the given frame.
  virtual std::vector< track_sptr > new_tracks( frame_id_t offset = -1 ) const
  {
    return impl_->new_tracks(offset);
  }

  /// Return all tracks terminated on the given frame.
  virtual std::vector< track_sptr > terminated_tracks( frame_id_t offset = -1 ) const
  {
    return impl_->terminated_tracks(offset);
  }

  /// Return the percentage of tracks successfully tracked between the two frames.
  virtual double percentage_tracked( frame_id_t offset1 = -2, frame_id_t offset2 = -1 ) const
  {
    return impl_->percentage_tracked(offset1, offset2);
  }

  /// Return a vector of state data corresponding to the tracks on the given frame.
  virtual std::vector<track_state_sptr> frame_states( frame_id_t offset = -1 ) const
  {
    return impl_->frame_states(offset);
  }

  /// Returns all frame data as map of frame index to track_set_frame_data
  virtual track_set_frame_data_map_t all_frame_data() const
  {
    return impl_->all_frame_data();
  }

  /// Return the additional data associated with all tracks on the given frame
  virtual track_set_frame_data_sptr frame_data( frame_id_t offset = -1 ) const
  {
    return impl_->frame_data(offset);
  }

  /// Removes the frame data for the frame offset
  virtual bool remove_frame_data(frame_id_t offset = -1)
  {
    return impl_->remove_frame_data(offset);
  }

  /// Set additional frame data associated with all tracks for all frames
  virtual bool set_frame_data( track_set_frame_data_map_t const& fmap )
  {
    return impl_->set_frame_data( fmap );
  }

  /// Set additional data associated with all tracks on the given frame
  virtual bool set_frame_data( track_set_frame_data_sptr data,
                               frame_id_t offset = -1 )
  {
    return impl_->set_frame_data( data, offset );
  }

  /// Convert an offset number to an absolute frame number
  virtual frame_id_t offset_to_frame( frame_id_t offset ) const
  {
    return impl_->offset_to_frame(offset);
  }

  virtual track_set_sptr clone( clone_type = clone_type::DEEP ) const;

  /// Merges the other feature track set into this feature track set.
  /**
  * \param other The other feature track set to merge into this one.
  * \param clone_method How to clone track states, if needed.
  * \param do_not_append_tracks
  *   If \c true, the other tracks are cloned and assigned a new track id. If
  *   \c false, if the same track id is found in other and in the current
  *   track set, then track states from other are cloned and appended to this
  *   object's track.
  */
  virtual void merge_in_other_track_set(
    track_set_sptr const& other,
    clone_type clone_method = clone_type::SHALLOW,
    bool do_not_append_tracks = false);

protected:

  /// The implementation of the track set functions
  std::unique_ptr<track_set_implementation> impl_;
};



// ------------------------------------------------------------------
/// A concrete track set that simply wraps a vector of tracks.
class simple_track_set_implementation
  : public track_set_implementation
{
public:
  /// Default Constructor
  simple_track_set_implementation() { }

  /// Constructor from a vector of tracks
  explicit simple_track_set_implementation(
      std::vector< track_sptr > const& tracks,
      track_set_frame_data_map_t const& frame_data =
        track_set_frame_data_map_t());

  /// Return the number of tracks in the set
  virtual size_t size() const { return data_.size(); }

  /// Return whether or not there are any tracks in the set
  virtual bool empty() const { return data_.empty(); }

  /// Return true if the set contains a specific track
  virtual bool contains( track_sptr t ) const;

  /// Assign a vector of track shared pointers to this container
  virtual void set_tracks( std::vector< track_sptr > const& tracks ) { data_ = tracks; }

  /// Insert a track shared pointer into this container
  virtual void insert( track_sptr t ) { data_.push_back( t ); }

  /// Remove a track from the set and return true if successful
  virtual bool remove( track_sptr t );

  /// Return a vector of track shared pointers
  virtual std::vector< track_sptr > tracks() const { return data_; }

  /// Returns all frame data as map of frame index to track_set_frame_data
  virtual track_set_frame_data_map_t all_frame_data() const { return frame_data_; }

  /// Return the additional data associated with all tracks on the given frame
  virtual track_set_frame_data_sptr frame_data( frame_id_t offset = -1 ) const;

  /// Removes the frame data for the frame offset
  virtual bool remove_frame_data(frame_id_t offset = -1);

  /// Set additional frame data associated with all tracks for all frames
  virtual bool set_frame_data( track_set_frame_data_map_t const& fmap )
  {
    frame_data_ = fmap;
    return true;
  }

  /// Set additional data associated with all tracks on the given frame
  virtual bool set_frame_data( track_set_frame_data_sptr data,
                               frame_id_t offset = -1 );

  track_set_implementation_uptr clone(
    clone_type = clone_type::DEEP ) const override;

protected:
  /// The vector of tracks
  std::vector< track_sptr > data_;

  /// The frame data map
  track_set_frame_data_map_t frame_data_;

};

} } // end namespace vital

#endif // VITAL_TRACK_SET_H_
