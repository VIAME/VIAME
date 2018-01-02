/*ckwg +29
 * Copyright 2013-2015 by Kitware, Inc.
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
 * \brief OCV detect_features algorithm implementation
 */

#include "detect_features_if_keyframe.h"

#include <vector>

#include <vital/exceptions/image.h>
#include <arrows/ocv/feature_set.h>
#include <arrows/ocv/image_container.h>
#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ocv {


class detect_features_if_keyframe::priv
{
public:

  /// The feature detector algorithm to use
  vital::algo::detect_features_sptr detector;

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

/// Extract a set of image features from the provided image
vital::feature_track_set_sptr
detect_features_if_keyframe
::detect(kwiver::vital::image_container_sptr image_data,
         unsigned int frame_number,
         kwiver::vital::feature_track_set_sptr feat_track_set,
         kwiver::vital::image_container_sptr mask) const
{
  
  vital::keyframe_metadata_sptr kfmd_ptr = 
    feat_track_set->get_keyframe_data()->get_frame_metadata(frame_number);
  if (!kfmd_ptr)
  {
    //no keyframe data for this frame id.  So, return the orignial feat_track_set
    return feat_track_set;  //no changes made so no deep copy necessary
  }

  vital::keyframe_metadata_for_basic_selector_sptr kfmd_basic_ptr =
    std::dynamic_pointer_cast<keyframe_metadata_for_basic_selector>(kfmd_ptr);
  if (!kfmd_basic_ptr)
  {
    //Can't cast it to a keyframe_metadata_for_basic_selector to get the
    //bool is_keyframe flag.  So, just return the original feat_track_set
    return feat_track_set; //no changes made so no deep copy necessary
  }

  //basic keyframe data is present.  Is it a keyframe?
  if (!kfmd_basic_ptr->is_keyframe)
  {
    //nope, so return the original feat_track_set
    return feat_track_set; //no changes made so no deep copy necessary
  }

  //detect the features
  vital::feature_set_sptr new_feat = d_->detector->detect(image_data, mask);

  //describe the features
  vital::descriptor_set_sptr new_desc = d_->extractor->extract(image_data, new_feat, mask);

  std::vector<feature_sptr> vf = new_feat->features();
  std::vector<descriptor_sptr> df = new_desc->descriptors();
  // get the last track id in the existing set of tracks and increment it
  track_id_t next_track_id = (*feat_track_set->all_track_ids().crbegin()) + 1;

  for (size_t i = 0; i < vf.size(); ++i)
  {
    auto fts = std::make_shared<feature_track_state>(frame_number);
    fts->feature = vf[i];
    fts->descriptor = df[i];
    auto t = vital::track::create();
    t->append(fts);
    t->set_id(next_track_id++);
    feat_track_set->insert(t);
  }

  //note that right now are haven't done any matching.  Each newly detected feature is in its own track.

  return feat_track_set;
}

detect_features_if_keyframe
::detect_features_if_keyframe()
  :d_(new priv)
{

}


/// Destructor
detect_features_if_keyframe
::~detect_features_if_keyframe() noexcept
{
}

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
detect_features_if_keyframe
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm
  algo::detect_features::
    get_nested_algo_configuration(d_->detector_name, config, d_->detector);

  // - Descriptor Extractor algorithm
  algo::extract_descriptors::
    get_nested_algo_configuration(d_->extractor_name, config, d_->extractor);

  return config;
}


/// Set this algo's properties via a config block
void
detect_features_if_keyframe
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::detect_features_sptr df;
  algo::detect_features::set_nested_algo_configuration(d_->detector_name, config, df);
  d_->detector = df;

  algo::extract_descriptors_sptr ed;
  algo::extract_descriptors::set_nested_algo_configuration(d_->extractor_name, config, ed);
  d_->extractor = ed;


}


bool
detect_features_if_keyframe
::check_configuration(vital::config_block_sptr config) const
{
  bool config_valid = true;

  config_valid = algo::detect_features::check_nested_algo_configuration(
    d_->detector_name, config) && config_valid;

  config_valid = algo::extract_descriptors::check_nested_algo_configuration(
    d_->extractor_name, config) && config_valid;

  return config_valid;
 }

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
