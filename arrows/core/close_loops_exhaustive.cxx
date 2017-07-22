/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * \brief Implementation of close_loops_exhaustive
 */

#include "close_loops_exhaustive.h"
#include "merge_tracks.h"

#include <algorithm>
#include <set>
#include <vector>

#include <vital/exceptions/algorithm.h>
#include <vital/algo/match_features.h>
#include <vital/util/thread_pool.h>


namespace kwiver {
namespace arrows {
namespace core {

using namespace kwiver::vital;

/// Private implementation class
class close_loops_exhaustive::priv
{
public:
  /// Constructor
  priv()
    : match_req(100),
      num_look_back(-1),
      m_logger( vital::get_logger( "arrows.core.close_loops_exhaustive" ))
  {
  }

  /// number of feature matches required for acceptance
  size_t match_req;

  /// Max frames to close loops back to (-1 to beginning of sequence)
  int num_look_back;

  /// The feature matching algorithm to use
  vital::algo::match_features_sptr matcher;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
close_loops_exhaustive
::close_loops_exhaustive()
: d_(new priv)
{
}


/// Destructor
close_loops_exhaustive
::~close_loops_exhaustive() VITAL_NOTHROW
{
}


std::string
close_loops_exhaustive
::description() const
{
  return "Exhaustive matching of all frame pairs, "
         "or all frames within a moving window";
}


/// Get this alg's \link vital::config_block configuration block \endlink
  vital::config_block_sptr
close_loops_exhaustive
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Matcher algorithm
  algo::match_features::get_nested_algo_configuration("feature_matcher", config, d_->matcher);

  config->set_value("match_req", d_->match_req,
                    "The required number of features needed to be matched for a success.");

  config->set_value("num_look_back", d_->num_look_back,
                    "Maximum number of frames to search in the past for matching to "
                    "(-1 looks back to the beginning).");

  return config;
}


/// Set this algo's properties via a config block
void
close_loops_exhaustive
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm configuration
  algo::match_features::set_nested_algo_configuration("feature_matcher",
                                                      config, d_->matcher);

  d_->match_req = config->get_value<size_t>("match_req");
  d_->num_look_back = config->get_value<int>("num_look_back");
}


bool
close_loops_exhaustive
::check_configuration(vital::config_block_sptr config) const
{
  return (
    algo::match_features::check_nested_algo_configuration("feature_matcher", config)
  );
}


/// Exaustive loop closure
vital::feature_track_set_sptr
close_loops_exhaustive
::stitch( vital::frame_id_t frame_number,
          vital::feature_track_set_sptr input,
          vital::image_container_sptr,
          vital::image_container_sptr ) const
{
  frame_id_t last_frame = 0;
  if (d_->num_look_back >= 0)
  {
    const int fnum = static_cast<int>(frame_number);
    last_frame = std::max<int>(fnum - d_->num_look_back, 0);
  }

  std::vector< vital::track_sptr > all_tracks = input->tracks();
  auto current_set = std::make_shared<vital::simple_feature_track_set>(
                         input->active_tracks( frame_number ) );

  std::vector<vital::track_sptr> current_tracks = current_set->tracks();
  vital::descriptor_set_sptr current_descriptors =
      current_set->frame_descriptors( frame_number );
  vital::feature_set_sptr current_features =
      current_set->frame_features( frame_number );

  // lambda function to encapsulate the parameters to be shared across all threads
  auto match_func = [=] (frame_id_t f)
  {
    return match_tracks(d_->matcher, input, current_set,
                        current_features, current_descriptors, f);
  };

  // access the thread pool
  vital::thread_pool& pool = vital::thread_pool::instance();

  std::map<vital::frame_id_t, std::future<track_pairs_t> > all_matches;
  // enqueue a task to run matching for each frame within a neighborhood
  for(vital::frame_id_t f = frame_number - 2; f >= last_frame; f-- )
  {
    all_matches[f] = pool.enqueue(match_func, f);
  }

  // retrieve match results and stitch frames together
  track_map_t track_replacement;
  for(vital::frame_id_t f = frame_number - 2; f >= last_frame; f-- )
  {
    auto const& matches = all_matches[f].get();
    size_t num_matched = matches.size();
    int num_linked = 0;
    if( num_matched >= d_->match_req )
    {
      num_linked = merge_tracks(matches, track_replacement);
    }

    LOG_INFO(d_->m_logger, "Matching frame " << frame_number << " to " << f
                           << " has "<< num_matched << " matches and "
                           << num_linked << " joined tracks");
  }

  // remove all tracks from 'input' that have now been replaced by
  // merging with another track
  input = remove_replaced_tracks(input, track_replacement);

  return input;
}


} // end namespace core
} // end namespace arrows
} // end namespace kwiver
