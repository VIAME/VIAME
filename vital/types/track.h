// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
///
/// Header for \link kwiver::vital::track track \endlink objects.

#ifndef VITAL_TRACK_H_
#define VITAL_TRACK_H_

#include <vital/attribute_set.h>
#include <vital/vital_config.h>
#include <vital/vital_export.h>
#include <vital/vital_types.h>

#include <memory>
#include <set>
#include <vector>

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
  track_ref( VITAL_UNUSED track_ref const& other ) {}
  track_ref( VITAL_UNUSED track_ref&& other ) {}

  track_ref& operator= ( track_ref const& rhs ) = delete;
  track_ref& operator= ( track_ref&& rhs ) = delete;

  track_ref& operator= ( track_sptr const& rhs )
  {
    this->std::weak_ptr< track >::operator= ( rhs );
    return *this;
  }
};

// ----------------------------------------------------------------------------
/// Empty base class for data associated with a track state.
class VITAL_EXPORT track_state
{
public:
  friend class track;

  track_state( frame_id_t frame )
    : frame_id_( frame )
  {}

  track_state() = default;
  track_state( track_state const& other ) = default;
  track_state( track_state&& other ) = default;

  /// Clone the track state (polymorphic copy constructor).
  virtual track_state_sptr clone( clone_type = clone_type::DEEP ) const
  {
    return std::make_shared<track_state>( *this );
  }

  /// Access the frame identifier.
  frame_id_t frame() const { return frame_id_; }

  /// Access the track containing this state.
  track_sptr track() const { return track_.lock(); }

  /// Set the frame identifier.
  void set_frame( frame_id_t frame_id ) { frame_id_ = frame_id; }

  virtual ~track_state() = default;

  bool operator==( track_state other ) const { return frame_id_ == other.frame(); }

private:
  /// Frame identifier for this state.
  frame_id_t frame_id_ = 0;

  /// Weak reference back to the parent track.
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
/// Special type of track data that redirects to another track.
///
/// The primary use case for this class is to aid bookkeeping for track
/// merging. When you merge one track into another and transfer all of its
/// states it leaves behind an invalid track with no states. A track may
/// appear in multiple matches and merging one match may invalidate another
/// match if one of the tracks is invalidated. This class allows the merging
/// function to insert a redirect into an invalidated track such that later
/// matches can find the new version of the track.
///
/// \par Example:
///   Assume a matcher returns matches (A,B) and (B,C). The merge function
///   merges B into A producing a longer A and an empty B. B uses this class to
///   redirect to A. The merge function tries to merge C into B, but B is now
///   invalid. However, B redirects to A, so the function tries to merge C
///   into A instead.
class VITAL_EXPORT track_data_redirect : public track_data
{
public:
  track_data_redirect( track_sptr t, track_data_sptr d )
    : redirect_track( t )
    , old_track_data( d )
  {
  }

  // Redirect to another track.
  track_sptr redirect_track;

  // Track data that used to be associated with this track.
  track_data_sptr old_track_data;
};

// ----------------------------------------------------------------------------
/// Representation of a track.
///
/// A track is a sequence of corresponding identifiers associated with each
/// other across time (i.e. frames indices). Each track consists of a sequence
/// of track states each with a frame id and optional data field. Frame
/// identifiers are in monotonically increasing order but need not be
/// sequential. The same track structure can be used to represent feature
/// tracks for image registration or moving object tracks.
class VITAL_EXPORT track : public std::enable_shared_from_this<track>
{
public:
  /// Convenience type for the \c const iterator of the track state vector.
  typedef std::vector< track_state_sptr >::const_iterator history_const_itr;

  ~track() = default;

  /// Factory function to create a new track state.
  static track_sptr create( track_data_sptr data = nullptr );

  /// Clone the track (polymorphic copy constructor).
  // TODO document difference between shallow and deep clones
  track_sptr clone( clone_type = clone_type::DEEP ) const;

  /// Access the track identification number.
  track_id_t id() const { return id_; }

  /// Access the track data.
  track_data_sptr data() const { return data_; }

  /// Set the track identification number.
  void set_id( track_id_t id ) { id_ = id; }

  /// Set the track data.
  void set_data( track_data_sptr d ) { data_ = d; }

  /// Access the first frame number covered by this track.
  frame_id_t first_frame() const;

  /// Access the last frame number covered by this track.
  frame_id_t last_frame() const;

  /// Append a track state.
  ///
  /// The added track state must have a \c frame_id greater than the last frame
  /// in the history.
  ///
  /// \returns \c true if successful, \c false not correctly ordered.
  /// \param state Track state to add to this track.
  bool append( track_state_sptr&& state );

  bool append( track_state_sptr const& state )
  { auto copy = state; return this->append( std::move( copy ) ); }

  /// Append the history contents of another track.
  ///
  /// The first state of the input track must contain a frame number greater
  /// than the last state of this track. Track states from \p to_append are
  /// reassigned to this track and removed from \p to_append. The track data in
  /// \p to_append is modified to redirect to this track using
  /// \p track_data_redirect.
  ///
  /// \returns \c true if successful, \c false not correctly ordered.
  bool append( track& to_append );

  /// Insert a track state.
  ///
  /// The added track state must have a frame_id that is not already
  /// present in the track history.
  ///
  /// \returns
  ///   \c true if successful, \c false if already a state on this frame.
  /// \param state Track state to add to this track.
  bool insert( track_state_sptr&& state );

  bool insert( track_state_sptr const& state )
  { auto copy = state; return this->insert( std::move( copy ) ); }

  /// Remove track state.
  ///
  /// \param frame The state to remove.
  /// \returns \c true if the state was found and removed.
  bool remove( track_state_sptr const& state );

  /// Remove track state by frame number.
  ///
  /// \param frame The frame number at which to remove a state.
  /// \returns \c true if a state with the specified frame number was found and
  ///          removed.
  bool remove( frame_id_t frame );

  /// Remove all track states.
  void clear();

  /// Access a \c const iterator to the start of the history.
  history_const_itr begin() const { return history_.begin(); }

  /// Access a \c const iterator to the end of the history.
  history_const_itr end() const { return history_.end(); }

  /// Access the first entry of the history.
  track_state_sptr front() const { return history_.front(); }

  /// Access the last entry of the history.
  track_state_sptr back() const { return history_.back(); }

  /// Find the track state iterator matching \p frame.
  ///
  /// \param frame The frame number to access.
  /// \return An iterator at the frame if found, or end() if not.
  history_const_itr find( frame_id_t frame ) const;

  /// Test if the track contains \p frame.
  ///
  /// \param frame The frame number to find.
  /// \return \c true if the track contains the specified \p frame,
  ///         otherwise \c false.
  bool contains( frame_id_t frame ) const;

  /// Erase track state at iterator.
  ///
  /// \return An iterator to the item which followed the erased item.
  history_const_itr erase( history_const_itr );

  /// Return the set of all frame identifiers covered by this track.
  std::set< frame_id_t > all_frame_ids() const;

  /// Return the number of states in the track.
  size_t size() const { return history_.size(); }

  /// Return whether or not this track has any states.
  bool empty() const { return history_.empty(); }

  /// Get attribute set.
  ///
  /// This method returns a pointer to the attribute set that is attached to
  /// this track. It is possible that the pointer is null, so check before
  /// using it.
  ///
  /// \return Pointer to attribute set or \c nullptr
  attribute_set_sptr attributes() const;

  /// Attach attribute set to this track.
  ///
  /// This method attaches the specified attribute set to this track.
  ///
  /// \param attrs Pointer to attribute set to attach.
  ///@{
  void set_attributes( attribute_set_sptr&& attrs );
  void set_attributes( attribute_set_sptr const& attrs );
  ///@}

protected:
  explicit track( track_data_sptr d = nullptr );
  track( const track& other );

  /// Ordered array of track states.
  std::vector< track_state_sptr > history_;

  /// Unique track identification number.
  track_id_t id_;

  /// Optional data structure associated with this track.
  track_data_sptr data_;
  /// The optional data structure associated with this track
  attribute_set_sptr attrs_;
};

} } // end namespace vital

#endif // VITAL_TRACK_H_
