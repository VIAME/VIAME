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
 * \brief Header file for \link kwiver::vital::feature_track_set feature_track_set
 *        \endlink
 */

#ifndef VITAL_FEATURE_TRACK_SET_H_
#define VITAL_FEATURE_TRACK_SET_H_

#include "descriptor_set.h"
#include "feature_set.h"
#include "track_set.h"

#include <vital/vital_export.h>
#include <vital/vital_config.h>
#include <vital/vital_types.h>

#include <vital/range/transform.h>

#include <limits>
#include <memory>
#include <vector>

namespace kwiver {
namespace vital {

class feature_track_set_frame_data;
using feature_track_set_frame_data_sptr =
  std::shared_ptr< feature_track_set_frame_data >;

// ============================================================================
/// A derived track_state for feature tracks
class VITAL_EXPORT feature_track_state : public track_state
{
public:
  //@{
  /// Constructor
  explicit feature_track_state( frame_id_t frame,
                                feature_sptr const& feature = nullptr,
                                descriptor_sptr const& descriptor = nullptr,
                                bool inlier = false )
    : track_state{ frame }
    , feature{ feature }
    , descriptor{ descriptor }
    , inlier{ inlier }
  {}

  explicit feature_track_state( frame_id_t frame,
                                feature_sptr&& feature,
                                descriptor_sptr&& descriptor,
                                bool inlier = false )
    : track_state{ frame }
    , feature{ std::move( feature ) }
    , descriptor{ std::move( descriptor ) }
    , inlier{ inlier }
  {}
  //@}

  /// Copy constructor
  feature_track_state( feature_track_state const& other ) = default;

  /// Move constructor
  feature_track_state( feature_track_state&& ) = default;

  /// Clone the track state (polymorphic copy constructor)
  track_state_sptr clone( clone_type ct = clone_type::DEEP ) const override
  {
    if ( ct == clone_type::DEEP )
    {
      auto new_feature =
        ( this->feature ? this->feature->clone() : nullptr );
      auto new_descriptor =
        ( this->descriptor ? this->descriptor->clone() : nullptr );
      return std::make_shared< feature_track_state >(
        this->frame(), std::move( new_feature ),
        std::move( new_descriptor ), this->inlier );
    }
    else
    {
      return std::make_shared< feature_track_state >( *this );
    }
  }

  static std::shared_ptr< feature_track_state > downcast(
    track_state_sptr const& sp )
  {
    return std::dynamic_pointer_cast< feature_track_state >( sp );
  }

  static constexpr auto downcast_transform = range::transform( downcast );

  feature_sptr feature;
  descriptor_sptr descriptor;
  bool inlier;
};

/// Shared pointer for feature_track_state type
using feature_track_state_sptr = std::shared_ptr< feature_track_state >;

// ============================================================================
/// A derived track_state_frame_data for feature tracks
class VITAL_EXPORT feature_track_set_frame_data
 : public track_set_frame_data
{
public:
  // Dynamic copy constructor
  track_set_frame_data_sptr clone() const override
  {
    return std::make_shared<feature_track_set_frame_data>(*this);
  }

  bool is_keyframe;
};


class feature_info {
public:
  feature_set_sptr features;
  descriptor_set_sptr descriptors;
  std::vector<track_sptr> corresponding_tracks;
};

typedef std::shared_ptr< feature_info> feature_info_sptr;

/// A collection of 2D feature point tracks
class VITAL_EXPORT feature_track_set : public track_set
{
public:
  /// Default Constructor
  /**
   * \note implementation defaults to simple_track_set_implementation
   */
  feature_track_set();

  /// Constructor specifying the implementation
  feature_track_set(std::unique_ptr<track_set_implementation> impl);

  /// Constructor from a vector of tracks
  /**
   * \note implementation defaults to simple_track_set_implementation
   */
  feature_track_set(std::vector< track_sptr > const& tracks);

  /// Destructor
  virtual ~feature_track_set() = default;

  /**
  * \note returns a deep copy of the feature_track_set
  */
  track_set_sptr clone( clone_type = clone_type::DEEP ) const override;

  /// Return the set of features in tracks on the last frame
  virtual feature_set_sptr last_frame_features() const;

  /// Return the set of descriptors in tracks on the last frame
  virtual descriptor_set_sptr last_frame_descriptors() const;

  /// Return the set of features in all tracks for the given frame.
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a feature_set_sptr for all features on the give frame.
   */
  virtual feature_set_sptr frame_features( frame_id_t offset = -1 ) const;

  /// Return the set of descriptors in all tracks for the given frame.
  /**
   * \param [in] offset the frame offset for selecting the target frame.
   *                    Positive number are absolute frame numbers while
   *                    negative numbers are relative to the last frame.  For
   *                    example, offset of -1 refers to the last frame and is
   *                    the default.
   *
   * \returns a descriptor_set_sptr for all features on the give frame.
   */
  virtual descriptor_set_sptr frame_descriptors( frame_id_t offset = -1 ) const;


  /// Return a vector of feature track states corresponding to the tracks on the given frame.
  /**
  * \param [in] offset the frame offset for selecting the target frame.
  *                    Positive number are absolute frame numbers while
  *                    negative numbers are relative to the last frame.  For
  *                    example, offset of -1 refers to the last frame and is
  *                    the default.
  *
  * \returns a vector for all feature tracks states on the given frame.
  */
  virtual std::vector<feature_track_state_sptr>
    frame_feature_track_states(frame_id_t offset = -1) const;

  /// Return a map of all feature_track_set_frame_data
  /**
   * This function is similar to \c all_frame_data() except that it checks
   * the type of the frame data and dynamically casts it to the specialized
   * frame data for feature_track_set.  Any frame data of a different type
   * is not included in this ouput.
   */
  virtual std::map<frame_id_t, feature_track_set_frame_data_sptr>
    all_feature_frame_data() const;

  /// Return the set of all keyframes in the track set
  /**
   * Keyframes are designated as frames which have an associated
   * feature_track_set_frame_data marked with is_keyframe == true
   */
  virtual std::set<frame_id_t> keyframes() const;

  virtual feature_info_sptr frame_feature_info(frame_id_t offset = -1,
    bool only_features_with_descriptors = true) const;

  /// Return the additional data associated with all feature tracks on the given frame
  feature_track_set_frame_data_sptr feature_frame_data(frame_id_t offset = -1) const;
};

/// Shared pointer for feature_track_set type
using feature_track_set_sptr = std::shared_ptr< feature_track_set >;

/// Helper to iterate over the states of a track as object track states
/**
 * This object is an instance of a range transform adapter that can be applied
 * to a track_sptr in order to directly iterate over the underlying
 * feature_track_state instances.
 *
 * \par Example:
 * \code
 * namespace kv = kwiver::vital;
 * namespace r = kwiver::vital::range;
 *
 * kv::track_sptr ft = get_the_feature_track();
 * for ( auto s : ft | kv::as_feature_track )
 *   std::cout << s->inlier << std::endl;
 * \endcode
 *
 * \sa kwiver::vital::range::transform_view
 */
static constexpr auto as_feature_track =
  feature_track_state::downcast_transform;


class feature_track_set_changes;
typedef std::shared_ptr<feature_track_set_changes> feature_track_set_changes_sptr;

class VITAL_EXPORT feature_track_set_changes
{
public:
  struct state_data {
    state_data(frame_id_t fid, track_id_t tid, bool inlier) :
      frame_id_(fid), track_id_(tid), inlier_(inlier) { }

    frame_id_t frame_id_;
    track_id_t track_id_;
    bool inlier_;
  };

  feature_track_set_changes() {};

  feature_track_set_changes(std::vector<state_data> const& changes) {
    m_changes = changes;
  };

  void clear() { m_changes.clear(); }

  void add_change(frame_id_t fid, track_id_t tid, bool inlier)
  {
    m_changes.push_back(state_data(fid, tid, inlier));
  }

  feature_track_set_changes_sptr clone()
  {
    return std::make_shared<feature_track_set_changes>(this->m_changes);
  }

  std::vector<state_data> m_changes;
};


} } // end namespace vital

#endif // VITAL_FEATURE_TRACK_SET_H_
