/*ckwg +29
 * Copyright 2013-2017, 2019 by Kitware, Inc.
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

#include "feature_track_set.h"

#include <limits>


namespace kwiver {
namespace vital {

typedef std::unique_ptr<track_set_implementation> tsi_uptr;


/// Default Constructor
feature_track_set
::feature_track_set()
  : track_set(tsi_uptr(new simple_track_set_implementation))
{
}


/// Constructor specifying the implementation
feature_track_set
::feature_track_set(std::unique_ptr<track_set_implementation> impl)
  : track_set(std::move(impl))
{
}


/// Constructor from a vector of tracks
feature_track_set
::feature_track_set(std::vector< track_sptr > const& tracks)
  : track_set(tsi_uptr(new simple_track_set_implementation(tracks)))
{
}

track_set_sptr
feature_track_set
::clone( clone_type ct ) const
{
  track_set_implementation_uptr new_imp =
    this->impl_->clone( ct );
  feature_track_set_sptr new_fts =
    std::make_shared<feature_track_set>(std::move(new_imp));
  return std::dynamic_pointer_cast<track_set>(new_fts);
}


/// Return the set of features in tracks on the last frame
feature_set_sptr
feature_track_set
::last_frame_features() const
{
  return frame_features(this->last_frame());
}


/// Return the set of descriptors in tracks on the last frame
descriptor_set_sptr
feature_track_set
::last_frame_descriptors() const
{
  return frame_descriptors(this->last_frame());
}


/// Return the set of features in all tracks for the given frame.
feature_set_sptr
feature_track_set
::frame_features( frame_id_t offset ) const
{
  std::vector<feature_sptr> features;
  std::vector<track_state_sptr> fsd = this->frame_states(offset);
  for( auto const data : fsd )
  {
    feature_sptr f = nullptr;
    auto fdata = std::dynamic_pointer_cast<feature_track_state>(data);
    if( fdata )
    {
      f = fdata->feature;
    }
    features.push_back(f);
  }
  return feature_set_sptr(new simple_feature_set(features));
}


/// Return the set of descriptors in all tracks for the given frame.
descriptor_set_sptr
feature_track_set
::frame_descriptors(frame_id_t offset) const
{
  std::vector<descriptor_sptr> descriptors;
  std::vector<track_state_sptr> fsd = this->frame_states(offset);
  for( auto const data : fsd )
  {
    descriptor_sptr d = nullptr;
    auto fdata = std::dynamic_pointer_cast<feature_track_state>(data);
    if( fdata )
    {
      d = fdata->descriptor;
    }
    descriptors.push_back(d);
  }

  return descriptor_set_sptr(new simple_descriptor_set(descriptors));
}

/// Return the vector of feature track states for all tracks for the given frame.
std::vector<feature_track_state_sptr>
feature_track_set
::frame_feature_track_states(frame_id_t offset) const
{
  std::vector<feature_track_state_sptr>  feat_states;
  std::vector<track_state_sptr> fsd = this->frame_states(offset);
  for (auto const data : fsd)
  {
    auto fdata = std::dynamic_pointer_cast<feature_track_state>(data);
    if (fdata)
    {
      feat_states.push_back(fdata);
    }
  }
  return feat_states;
}


feature_info_sptr
feature_track_set
::frame_feature_info(frame_id_t offset,
  bool only_features_with_descriptors) const
{
  feature_info_sptr fi = std::make_shared<feature_info>();

  std::vector<feature_sptr> features;
  std::vector<descriptor_sptr> descriptors;
  std::vector<track_state_sptr> fsd = this->frame_states(offset);

  for (auto const data : fsd)
  {
    feature_sptr f = nullptr;
    descriptor_sptr d = nullptr;
    track_sptr t = nullptr;

    auto fdata = std::dynamic_pointer_cast<feature_track_state>(data);

    if (fdata)
    {
      f = fdata->feature;
      d = fdata->descriptor;
      t = fdata->track();
      if (only_features_with_descriptors && !d)
      {
        continue;
      }

      features.push_back(f);
      descriptors.push_back(d);
      fi->corresponding_tracks.push_back(t);
    }
  }

  fi->features = feature_set_sptr(new simple_feature_set(features));
  fi->descriptors = descriptor_set_sptr(new simple_descriptor_set(descriptors));

  return fi;
}


/// Return a map of all feature_track_set_frame_data
std::map<frame_id_t, feature_track_set_frame_data_sptr>
feature_track_set
::all_feature_frame_data() const
{
  std::map<frame_id_t, feature_track_set_frame_data_sptr> feature_fmap;
  track_set_frame_data_map_t fmap = this->all_frame_data();
  for (auto fd : fmap)
  {
    auto ftsfd =
      std::dynamic_pointer_cast<feature_track_set_frame_data>(fd.second);
    if ( ftsfd )
    {
      feature_fmap[fd.first] = ftsfd;
    }
  }
  return feature_fmap;
}


/// Return the set of all keyframes in the track set
std::set<frame_id_t>
feature_track_set
::keyframes() const
{
  std::set<frame_id_t> keyframes;
  track_set_frame_data_map_t fdm = this->all_frame_data();
  for (auto fd : fdm)
  {
    auto ftsfd =
      std::dynamic_pointer_cast<feature_track_set_frame_data>(fd.second);
    if (ftsfd && ftsfd->is_keyframe)
    {
      keyframes.insert(fd.first);
    }
  }
  return keyframes;
}

feature_track_set_frame_data_sptr
feature_track_set
::feature_frame_data(frame_id_t offset) const
{
  return std::dynamic_pointer_cast<feature_track_set_frame_data>(impl_->frame_data(offset));
}

} } // end namespace vital
