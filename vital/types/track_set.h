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
#include <memory>

namespace kwiver {
namespace vital {

class track_set;
/// Shared pointer for base track_set type
typedef std::shared_ptr< track_set > track_set_sptr;

/// A collection of tracks
class VITAL_EXPORT track_set
{
public:
  /// Destructor
  virtual ~track_set() VITAL_DEFAULT_DTOR

  /// Return the number of tracks in the set
  virtual size_t size() const;

  /// Return whether or not there are any tracks in the set
  virtual bool empty() const;

  /// Return a vector of track shared pointers
  virtual std::vector< track_sptr > tracks() const = 0;

  /// Return the set of all frame IDs covered by these tracks
  virtual std::set< frame_id_t > all_frame_ids() const;

  /// Return the set of all track IDs in this track set
  virtual std::set< track_id_t > all_track_ids() const;

  /// Return the first (smallest) frame number containing tracks
  /**
   * If there are no tracks in this set, or no tracks have states, this returns
   * 0.
   */
  virtual frame_id_t first_frame() const;

  /// Return the last (largest) frame number containing tracks
  /**
   * If there are no tracks in this set, or no tracks have states, this returns
   * 0.
   */
  virtual frame_id_t last_frame() const;

  /// Return the track in this set with the specified id.
  /**
   * An empty pointer will be returned if the track cannot be found.
   *
   * \param [in] tid track identifier for the desired track.
   *
   * \returns a pointer to the track with the given id.
   */
  virtual track_sptr const get_track( track_id_t tid ) const;

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
  virtual std::vector< track_sptr> active_tracks( frame_id_t offset = -1 );

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
  virtual std::vector< track_sptr > inactive_tracks( frame_id_t offset = -1 );

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
  virtual std::vector< track_sptr > new_tracks( frame_id_t offset = -1 );

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
  virtual std::vector< track_sptr > terminated_tracks( frame_id_t offset = -1 );

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
  virtual double percentage_tracked( frame_id_t offset1 = -2, frame_id_t offset2 = -1 );

  /// Return a vector of state data corresponding to the tracks on the given frame.
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a vector of track_state_data_sptr corresponding to the tracks
              on this frame and in the same order as active_track(offset)
   */
  virtual std::vector<track_state_data_sptr> frame_state_data( frame_id_t offset = -1 ) const;

protected:
  /// Convert an offset number to an absolute frame number
  frame_id_t offset_to_frame( frame_id_t offset ) const;
};



/// A concrete track set that simply wraps a vector of tracks.
class simple_track_set :
  public track_set
{
public:
  /// Default Constructor
  simple_track_set() { }

  /// Constructor from a vector of tracks
  explicit simple_track_set( const std::vector< track_sptr >& tracks )
    : data_( tracks ) { }

  /// Return the number of tracks in the set
  virtual size_t size() const { return data_.size(); }

  /// Return a vector of track shared pointers
  virtual std::vector< track_sptr > tracks() const { return data_; }


protected:
  /// The vector of tracks
  std::vector< track_sptr > data_;
};

} } // end namespace vital

#endif // VITAL_TRACK_SET_H_
