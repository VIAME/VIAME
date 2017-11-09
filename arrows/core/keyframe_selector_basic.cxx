/*ckwg +29
* Copyright 2017 by Kitware, Inc.
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
* \brief Implementation of keyframe_selector_basic
*/

#include "keyframe_selector_basic.h"

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

class keyframe_selector_basic::priv {
public:
  priv()
    : keyframe_min_feature_count(50)
    , fraction_tracks_lost_to_necessitate_new_keyframe(0.65f)  //setting very high for debugging purposes
    , next_candidate_keyframe_id(-1)
  {
  }  

  virtual ~priv() {}

  /// Set our parameters based on the given config block
  void set_config(const vital::config_block_sptr & config)
  {
    if (config->has_value("fraction_tracks_lost_to_necessitate_new_keyframe"))
    {
      fraction_tracks_lost_to_necessitate_new_keyframe = config->get_value<float>(
        "fraction_tracks_lost_to_necessitate_new_keyframe");
    }
    if (config->has_value("keyframe_min_feature_count"))
    {
      keyframe_min_feature_count = config->get_value<int>(
        "keyframe_min_feature_count");
    }
  }

  /// Set current parameter values to the given config block
  void update_config(vital::config_block_sptr &config) const
  {
    config->set_value("fraction_tracks_lost_to_necessitate_new_keyframe", 
      fraction_tracks_lost_to_necessitate_new_keyframe,
      "if this fraction of more of features is lost then select a new keyframe");
    config->set_value("keyframe_min_feature_count",
      keyframe_min_feature_count,
      "minimum number of features required for a frame to become a keyframe");
  }

  bool check_configuration(vital::config_block_sptr config) const
  {
    bool success(true);

    float test_fraction_tracks_lost_to_necessitate_new_keyframe = config->get_value<float>(
      "fraction_tracks_lost_to_necessitate_new_keyframe");

    if (!(0 < test_fraction_tracks_lost_to_necessitate_new_keyframe && 
          test_fraction_tracks_lost_to_necessitate_new_keyframe <= 1.0))
    {         
      LOG_ERROR(m_logger, "fraction_tracks_lost_to_necessitate_new_keyframe ("
        << test_fraction_tracks_lost_to_necessitate_new_keyframe
        << ") should be greater than zero and <= 1.0");
      success = false;
    }

    int test_keyframe_min_feature_count = config->get_value<int>(
      "keyframe_min_feature_count");

    if (test_keyframe_min_feature_count < 0)
    {
      LOG_ERROR(m_logger, "keyframe_min_feature_count ("
        << test_keyframe_min_feature_count
        << ") should be greater than zero");
      success = false;
    }

    return success;
  }

  void initial_keyframe_selection(
    kwiver::vital::track_set_sptr tracks);

  void continuing_keyframe_selection(
    kwiver::vital::track_set_sptr tracks);

  int keyframe_min_feature_count;
  float fraction_tracks_lost_to_necessitate_new_keyframe;
  frame_id_t next_candidate_keyframe_id;   // this is a member variable to 
                                           // prevent us from checking the 
                                           // same frames repeatedly to see if 
                                           // they are key-frames
  kwiver::vital::logger_handle_t m_logger;
};

void
keyframe_selector_basic::priv
::initial_keyframe_selection(
  kwiver::vital::track_set_sptr tracks)
{
  auto kfd = tracks->get_keyframe_data();
  auto kfd_metadata_map = kfd->get_keyframe_metadata_map();

  // start with first frame, can it be a keyframe?  If so add it, if not keep
  // going until we find a first suitable keyframe.
  auto frame_ids = tracks->all_frame_ids();
  for (auto frame : frame_ids)
  {    
    if (tracks->num_active_tracks() >= keyframe_min_feature_count)
    {
      //this is the first frame that can be a keyframe
      std::shared_ptr<keyframe_metadata> sptr = 
        std::shared_ptr<keyframe_metadata>(
        (keyframe_metadata*)new keyframe_metadata_for_basic_selector(true));
      tracks->set_frame_metadata(frame, sptr);
      break;
    }
  }
}

void
keyframe_selector_basic::priv
::continuing_keyframe_selection(
  kwiver::vital::track_set_sptr tracks)
{
  auto kfd = tracks->get_keyframe_data();
  auto kfd_metadata_map = kfd->get_keyframe_metadata_map();

  // go to the last key-frame, then consider each frame newer than that one in 
  // order and decide if it should be a keyframe
  if (kfd_metadata_map->empty())
  {
    return;
  }

  auto latest_kfmd_it = kfd_metadata_map->crbegin();
  frame_id_t last_keyframe_id = latest_kfmd_it->first;
  keyframe_metadata_sptr latest_kfmd = latest_kfmd_it->second;

  keyframe_metadata_for_basic_selector_sptr latest_kfmd_basic = 
    std::dynamic_pointer_cast<keyframe_metadata_for_basic_selector>(latest_kfmd);
  if (!latest_kfmd_basic)
  {
    //metadata should be keyframe_metadata_for_basic_selector and is not.
    return;
  }

  if (!latest_kfmd_basic->is_keyframe)
  {
    //this should be a keyframe
    return;
  }

  if (next_candidate_keyframe_id < 0)
  {
    next_candidate_keyframe_id = last_keyframe_id + 1;
  }

  frame_id_t last_frame_id = tracks->last_frame();

  for ( ; next_candidate_keyframe_id <= last_frame_id ; ++next_candidate_keyframe_id)
  {
    double percentage_tracked = 
      tracks->percentage_tracked(last_keyframe_id, next_candidate_keyframe_id);
    if ( percentage_tracked > (1.0 - fraction_tracks_lost_to_necessitate_new_keyframe))
    {
      continue;  // no need to make candidate_keyframe_id a keyframe
    }

    //ok we could make this a keyframe.  Does it have enough features?
    if (tracks->num_active_tracks() >= keyframe_min_feature_count)
    {  //yep, it's a keyframe
      //add it's metadata to tracks      
      std::shared_ptr<keyframe_metadata> sptr = 
        std::dynamic_pointer_cast<keyframe_metadata>(
          std::make_shared<keyframe_metadata_for_basic_selector>(true));
      tracks->set_frame_metadata(next_candidate_keyframe_id, sptr);
      last_keyframe_id = next_candidate_keyframe_id;
    }
  }
}



/// Default Constructor
keyframe_selector_basic
::keyframe_selector_basic()
{
  d_ = std::make_shared<keyframe_selector_basic::priv>();

  attach_logger("keyframe_selector_basic");
  d_->m_logger = this->logger();
}

/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
keyframe_selector_basic
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  d_->update_config(config);

  return config;
}

/// Set this algo's properties via a config block
void
keyframe_selector_basic
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are 
  // present.  An alternative is to check for key presence before performing a 
  //get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);
  d_->set_config(in_config);
}

bool
keyframe_selector_basic
::check_configuration(vital::config_block_sptr config) const
{
  return d_->check_configuration(config);
}

kwiver::vital::track_set_sptr
keyframe_selector_basic
::select(kwiver::vital::track_set_sptr tracks) const
{
  // General idea here:
  // Add a key frame if
  // 1) Number of continuous feature tracks to a frame drops below 90%
  //    of features existing in any neighboring key-frame
  // 2) number of features in frame is greater than some minimum.  This prevents 
  //    keyframes from being added in areas with little texture (few features).

  keyframe_data_const_sptr kfd = tracks->get_keyframe_data();

  auto kfd_metadata_map = kfd->get_keyframe_metadata_map();

  if (kfd_metadata_map->empty())
  {
    //we don't have any keyframe data yet for this set of tracks.
    d_->initial_keyframe_selection(tracks);
  }

  if (!kfd_metadata_map->empty())
  { //check again because initial keyframe selection could have added a keyframe
    d_->continuing_keyframe_selection(tracks);
  }

  return tracks; // no op for now
}
/*
//this will eventually go in keyframe_selector_graph
kwiver::vital::track_set_sptr
keyframe_selector_graph
::select(kwiver::vital::track_set_sptr tracks) const
{
  // General idea here:
  // Add a key frame if
  // 1) Number of continuous feature tracks to a frame drops below 90%
  //    of features existing in any neighboring key-frame
  // 2) number of features in frame is greater than some minimum.  This prevents 
  //    keyframes from being added in areas with little texture (few features).

  keyframe_data_const_sptr kfd = tracks->get_keyframe_data();
  //std::shared_ptr<const keyframe_data_graph> kfd_g;
  //kfd_g = std::dynamic_pointer_cast<const keyframe_data_graph>(kfd);
  //if(kfd_g)
  //{
  //  //do the algorithm that uses the graph data structures
  //  return tracks;
  //}

  // we can add if statements for algorithms that correspond to other underlying
  // keyframe_data structures here

  //do the basic algorithm that uses the
  auto kfd_metadata_map = kfd->get_keyframe_metadata_map();


  return tracks; // no op for now
}
*/

} // end namespace core
} // end namespace arrows
} // end namespace kwiver