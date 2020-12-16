// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Implementation of close_loops_bad_frames_only
 */

#include "close_loops_bad_frames_only.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <functional>

#include <vital/algo/algorithm.h>
#include <vital/exceptions/algorithm.h>

namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;

/// Default Constructor
close_loops_bad_frames_only
::close_loops_bad_frames_only()
: enabled_(true),
  percent_match_req_(0.35),
  new_shot_length_(2),
  max_search_length_(5)
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
close_loops_bad_frames_only
::get_configuration() const
{
  // get base config from base class
  kwiver::vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Matcher algorithm
  algo::match_features::get_nested_algo_configuration("feature_matcher", config, matcher_);

  // Bad frame detection parameters
  config->set_value("enabled", enabled_,
                    "Should bad frame detection be enabled? This option will attempt to "
                    "bridge the gap between frames which don't meet certain criteria "
                    "(percentage of feature points tracked) and will instead attempt "
                    "to match features on the current frame against past frames to "
                    "meet this criteria. This is useful when there can be bad frames.");

  config->set_value("percent_match_req", percent_match_req_,
                    "The required percentage of features needed to be matched for a "
                    "stitch to be considered successful (value must be between 0.0 and "
                    "1.0).");

  config->set_value("new_shot_length", new_shot_length_,
                    "Number of frames for a new shot to be considered valid before "
                    "attempting to stitch to prior shots.");

  config->set_value("max_search_length", max_search_length_,
                    "Maximum number of frames to search in the past for matching to "
                    "the end of the last shot.");

  return config;
}

// ----------------------------------------------------------------------------
/// Set this algo's properties via a config block
void
close_loops_bad_frames_only
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::match_features_sptr mf;
  algo::match_features::set_nested_algo_configuration("feature_matcher", config, mf);
  matcher_ = mf;

  // Settings for bad frame detection
  enabled_ = config->get_value<bool>("enabled");
  percent_match_req_ = config->get_value<double>("percent_match_req");
  max_search_length_ = config->get_value<unsigned>("max_search_length");
  new_shot_length_ = config->get_value<unsigned>("new_shot_length");
  new_shot_length_ = ( new_shot_length_ ? new_shot_length_ : 1 );
}

// ----------------------------------------------------------------------------
  bool
close_loops_bad_frames_only
::check_configuration(vital::config_block_sptr config) const
{
  return (
    algo::match_features::check_nested_algo_configuration("feature_matcher", config)
    &&
    std::abs( config->get_value<double>("percent_match_req") ) <= 1.0
  );
}

// ----------------------------------------------------------------------------
/// Handle track bad frame detection if enabled
vital::feature_track_set_sptr
close_loops_bad_frames_only
::stitch( vital::frame_id_t frame_number,
          vital::feature_track_set_sptr input,
          vital::image_container_sptr,
          vital::image_container_sptr ) const
{
  // check if enabled and possible
  if( !enabled_ || frame_number <= new_shot_length_ )
  {
    return input;
  }

  // check if we should attempt to stitch together past frames
  std::vector< vital::track_sptr > all_tracks = input->tracks();
  vital::frame_id_t frame_to_stitch = frame_number - new_shot_length_ + 1;
  double pt = input->percentage_tracked( frame_to_stitch - 1, frame_to_stitch );
  bool stitch_required = ( pt < percent_match_req_ );

  // confirm that the new valid shot criteria length is satisfied
  vital::frame_id_t frame_to_test = frame_to_stitch + 1;
  while( stitch_required && frame_to_test <= frame_number )
  {
    pt = input->percentage_tracked( frame_to_test - 1, frame_to_test );
    stitch_required = ( pt >= percent_match_req_ );
    frame_to_test++;
  }

  // determine if a stitch can be attempted
  if( !stitch_required )
  {
    return input;
  }

  // attempt to stitch start of shot frame against past n frames
  frame_to_test = frame_to_stitch - 2;
  vital::frame_id_t last_frame_to_test = 0;

  if( frame_to_test > max_search_length_ )
  {
    last_frame_to_test = frame_to_test - max_search_length_;
  }

  auto stitch_frame_set = std::make_shared<vital::feature_track_set>(
                              input->active_tracks( frame_to_stitch ) );

  for( ; frame_to_test > last_frame_to_test; frame_to_test-- )
  {
    auto test_frame_set = std::make_shared<vital::feature_track_set>(
                              input->active_tracks( frame_to_test ) );

    // run matcher alg
    vital::match_set_sptr mset = matcher_->match(test_frame_set->frame_features( frame_to_test ),
                                          test_frame_set->frame_descriptors( frame_to_test ),
                                          stitch_frame_set->frame_features( frame_to_stitch ),
                                          stitch_frame_set->frame_descriptors( frame_to_stitch ));

    // test matcher results
    unsigned total_features = static_cast<unsigned>(test_frame_set->size() + stitch_frame_set->size());

    if( 2*mset->size() >= static_cast<unsigned>(percent_match_req_*total_features) )
    {
      // modify track history and exit
      std::vector<vital::track_sptr> test_frame_trks = test_frame_set->tracks();
      std::vector<vital::track_sptr> stitch_frame_trks = stitch_frame_set->tracks();
      std::vector<vital::match> matches = mset->matches();

      for( unsigned i = 0; i < matches.size(); i++ )
      {
        input->merge_tracks( stitch_frame_trks[ matches[i].second ],
                             test_frame_trks[ matches[i].first] );
      }

      return input;
    }
  }

  // bad frame detection has failed
  return input;
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
