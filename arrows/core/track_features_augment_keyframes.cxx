/*ckwg +29
 * Copyright 2017-2018 by Kitware, Inc.
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
 * \brief Implementation of core track_features_augment_keyframes
 */

#include "track_features_augment_keyframes.h"

#include <vector>

#include <vital/exceptions/image.h>
#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


class track_features_augment_keyframes::priv
{
public:

  /// The descriptor extractor algorithm to use
  vital::algo::extract_descriptors_sptr extractor;

  const std::string detector_name;

  const std::string extractor_name;
  priv()
    :detector_name("kf_only_feature_detector")
    ,extractor_name("kf_only_descriptor_extractor")
  {

  }
};


/// Augment existing tracks with additional feature if a keyframe
vital::feature_track_set_sptr
track_features_augment_keyframes
::track(kwiver::vital::feature_track_set_sptr tracks,
        unsigned int frame_number,
        kwiver::vital::image_container_sptr image_data,
        kwiver::vital::image_container_sptr mask) const
{

  // FORCE DETECTION ON EVERY FRAME

  //auto fmap = tracks->all_feature_frame_data();
  //auto ftsfd = fmap.find(frame_number);
  //if (ftsfd == fmap.end() || !ftsfd->second || !ftsfd->second->is_keyframe)
  //{
  //  // this is not a keyframe, so return the orignial tracks
  //  // no changes made so no deep copy necessary
  //  return tracks;
  //}

  auto track_states = tracks->frame_states(frame_number);
  auto new_feat = tracks->frame_features(frame_number);

  //describe the features.  Note this will recalculate the feature angles.
  vital::descriptor_set_sptr new_desc =
    d_->extractor->extract(image_data, new_feat, mask);

  std::vector<feature_sptr> vf = new_feat->features();
  std::vector<descriptor_sptr> df = new_desc->descriptors();
  for (size_t i = 0; i < vf.size(); ++i)
  {
    auto feat = vf[i];
    auto desc = df[i];

    // Go through existing features and find the one that equals feat.
    // The feature pointers may have changed in detect so we can't use them
    // directly with a map.
    for (auto ts : track_states)
    {
      auto fts = std::static_pointer_cast<feature_track_state>(ts);
      if (fts && fts->feature && fts->feature->equal_except_for_angle(*feat))
      {
        //feature must be set because extract will have calculated a new feature angle
        fts->feature = feat;
        fts->descriptor = desc;
        break;
      }
    }
  }

  return tracks;
}

track_features_augment_keyframes
::track_features_augment_keyframes()
  :d_(new priv)
{
}


/// Destructor
track_features_augment_keyframes
::~track_features_augment_keyframes() noexcept
{
}

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
track_features_augment_keyframes
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Descriptor Extractor algorithm
  algo::extract_descriptors::
    get_nested_algo_configuration(d_->extractor_name, config, d_->extractor);

  return config;
}


/// Set this algo's properties via a config block
void
track_features_augment_keyframes
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::extract_descriptors_sptr ed;
  algo::extract_descriptors::set_nested_algo_configuration(d_->extractor_name, config, ed);
  d_->extractor = ed;
}


bool
track_features_augment_keyframes
::check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;

  config_valid = algo::detect_features::check_nested_algo_configuration(
    d_->detector_name, config) && config_valid;

  config_valid = algo::extract_descriptors::check_nested_algo_configuration(
    d_->extractor_name, config) && config_valid;

  return config_valid;
 }

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
