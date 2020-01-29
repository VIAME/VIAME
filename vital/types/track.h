/*ckwg +29
 * Copyright 2013-2017, 2019-2020 by Kitware, Inc.
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
 * \brief Header for \link kwiver::vital::track track \endlink objects
 */

#ifndef VITAL_TRACK_H_
#define VITAL_TRACK_H_


#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vector>
#include <set>
#include <memory>

namespace kwiver {
namespace vital {

class track;
class track_state;
using track_sptr = std::shared_ptr< track >;
using track_state_sptr = std::shared_ptr< track_state >;

constexpr track_id_t invalid_track_id = -1;

// ----------------------------------------------------------------------------
class VITAL_EXPORT track_ref : public std::weak_ptr< track >
{
public:
  track_ref() = default;
  track_ref( track_ref const& other ) {}
  track_ref( track_ref&& other ) {}

  track_ref& operator= ( track_ref const& rhs ) = delete;
  track_ref& operator= ( track_ref&& rhs ) = delete;

  track_ref& operator= ( track_sptr const& rhs )
  {
    this->std::weak_ptr< track >::operator= ( rhs );
    return *this;
  }
};

// ----------------------------------------------------------------------------
/// Empty base class for data associated with a track state
class VITAL_EXPORT track_state
{
public:
  friend class track;

  track_state() = default;

  /// Constructor
  track_state( frame_id_t frame )
    : frame_id_( frame )
  {}

  /// Copy constructor
  track_state( track_state const& other ) = default;

  /// Move constructor
  track_state( track_state&& other ) = default;

  /// Clone the track state (polymorphic copy constructor)
  virtual track_state_sptr clone( clone_type = clone_type::DEEP ) const
  {
    return std::make_shared<track_state>( *this );
  }

  /// Access the frame identifier
  frame_id_t frame() const { return frame_id_; }

  /// Access the track containing this state
  track_sptr track() const { return track_.lock(); }

  /// Set the frame identifier
  void set_frame( frame_id_t frame_id ) { frame_id_ = frame_id; }

  virtual ~track_state() = default;

  bool operator==( track_state other ) const { return frame_id_ == other.frame(); }

private:
  /// The frame identifier for this state
  frame_id_t frame_id_ = 0;

  /// A weak reference back to the parent track
  track_ref track_;
};


// ----------------------------------------------------------------------------
/// Empty base class for data associated with a whole track.
class VITAL_EXPORT track_data
{
protected:
  virtual ~track_data() = default;
};

typedef std::shared_ptr< track_data > track_data_sptr;


// ----------------------------------------------------------------------------
/// A special type of track data that redirects to another track
/**
 * The primary use case for this class is to aid bookkeeping for track merging.
 * When you merge one track into another and transfer all of its states it
 * leaves behind an invalid track with no states.  A track may appear in
 * multiple matches and merging one match may invalidate another match if one
 * of the tracks is invalidated.  This class allows the merging function
 * to insert a redirect into an invalidated track such that later matches
 * can find the new version of the track.
 *
 * Example: assume a matcher returns matches (A,B) and (B,C).  The merge
 * function merges B into A producing a longer A and an empty B.  B uses this
 * class to redirect to A.  The merge function tries to merge C into B, but B
 * is now invalid.  However, B redirects to A, so the function tries to merge
 * C into A instead.
 */
class VITAL_EXPORT track_data_redirect : public track_data
{
public:
  // Constructor
  track_data_redirect( track_sptr t, track_data_sptr d )
    : redirect_track( t )
    , old_track_data( d )
  {
  }

  // A redirect to another track
  track_sptr redirect_track;
  // The track data that used to be associated with this track
  track_data_sptr old_track_data;
};


// ----------------------------------------------------------------------------
/// A representation of a track.
/**
 * A track is a sequence of corresponding identifiers associated with each
 * other across time (i.e. frames indicies).  Each track consists of a
 * sequence of track states each with a frame id and optional data field.
 * Frame ids are in monotonically increasing order but need not be sequential.
 * The same track structure can be used to represent feature tracks for
 * image registration or moving object tracks.
 */
class VITAL_EXPORT track : public std::enable_shared_from_this<track>
{
public:
  /// convenience type for the const iterator of the track state vector
  typedef std::vector< track_state_sptr >::const_iterator history_const_itr;

  /// Destructor
  ~track() = default;

  /// Factory function
  static track_sptr create( track_data_sptr data = nullptr );

  /// Clone
  track_sptr clone( clone_type = clone_type::DEEP ) const;

  /// Access the track identification number
  track_id_t id() const { return id_; }

  /// Access the track data
  track_data_sptr data() const { return data_; }

  /// Set the track identification number
  void set_id( track_id_t id ) { id_ = id; }

  /// Set the track data
  void set_data( track_data_sptr d ) { data_ = d; }

  /// Access the first frame number covered by this track
  frame_id_t first_frame() const;

  /// Access the last frame number covered by this track
  frame_id_t last_frame() const;

  /// Append a track state.
  /**
   * The added track state must have a frame_id greater than
   * the last frame in the history.
   *
   * \returns true if successful, false not correctly ordered
   * \param state track state to add to this track.
   */
  bool append( track_state_sptr&& state );

  bool append( track_state_sptr const& state )
  { auto copy = state; return this->append( std::move( copy ) ); }

  /// Append the history contents of another track.
  /**
   * The first state of the input track must contain a frame number
   * greater than the last state of this track.  Track states from
   * \p to_append are reassigned to this track and removed from
   * \p to_append.  The track data in \p to_append is modified to redirect
   * to this track using \p track_data_redirect.
   *
   * \returns true if successful, false not correctly ordered
   */
  bool append( track& to_append );

  /// Insert a track state.
  /**
   * The added track state must have a frame_id that is not already
   * present in the track history.
   *
   * \returns true if successful, false if already a state on this frame
   * \param state track state to add to this track.
   */
  bool insert( track_state_sptr&& state );

  bool insert( track_state_sptr const& state )
  { auto copy = state; return this->insert( std::move( copy ) ); }

  /// Remove track state
  /**
   * Removes the track state
   * Returns true if the state was found and removed
  */
  bool remove( track_state_sptr const& state );

  /// Remove all track states.
  void clear();

  /// Access a const iterator to the start of the history
  history_const_itr begin() const { return history_.begin(); }

  /// Access a const iterator to the end of the history
  history_const_itr end() const { return history_.end(); }

  /// Access the first entry of the history
  track_state_sptr front() const { return history_.front(); }

  /// Access the last entry of the history
  track_state_sptr back() const { return history_.back(); }

  /// Find the track state iterator matching \a frame
  /**
   *  \param [in] frame the frame number to access
   *  \return an iterator at the frame if found, or end() if not
   */
  history_const_itr find( frame_id_t frame ) const;

  /// Return the set of all frame IDs covered by this track
  std::set< frame_id_t > all_frame_ids() const;

  /// Return the number of states in the track.
  size_t size() const { return history_.size(); }

  /// Return whether or not this track has any states.
  bool empty() const { return history_.empty(); }


protected:
  /// Default Constructor
  explicit track( track_data_sptr d = nullptr );

  /// Copy Constructor
  track( const track& other );

  /// The ordered array of track states
  std::vector< track_state_sptr > history_;
  /// The unique track identification number
  track_id_t id_;
  /// The optional data structure associated with this track
  track_data_sptr data_;
};

} } // end namespace vital

#endif // VITAL_TRACK_H_
