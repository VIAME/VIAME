// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of close_loops_bad_frames_only

#include "close_loops_bad_frames_only.h"

#include <algorithm>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/algorithm_framework/exceptions/algorithm.h>

namespace kwiver {

namespace arrows {

namespace core {

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
void
close_loops_bad_frames_only
::initialize()
{
  attach_logger( "arrows.core.close_loops_bad_frames_only" );
}

// ----------------------------------------------------------------------------
void
close_loops_bad_frames_only
::set_configuration_internal( [[maybe_unused]] vital::config_block_sptr config )
{
  c_new_shot_length = ( c_new_shot_length ? c_new_shot_length : 1 );
}

// ----------------------------------------------------------------------------
bool
close_loops_bad_frames_only
::check_configuration( vital::config_block_sptr config ) const
{
  return (
    check_nested_algo_configuration< algo::match_features >(
      "feature_matcher",
      config )
    &&
    std::abs( config->get_value< double >( "percent_match_req" ) ) <= 1.0
  );
}

// ----------------------------------------------------------------------------
/// Handle track bad frame detection if enabled
vital::feature_track_set_sptr
close_loops_bad_frames_only
::stitch(
  vital::frame_id_t frame_number,
  vital::feature_track_set_sptr input,
  vital::image_container_sptr,
  vital::image_container_sptr ) const
{
  // check if enabled and possible
  if( !c_enabled || frame_number < 0 ||
      static_cast< size_t >( frame_number ) <= c_new_shot_length )
  {
    return input;
  }

  // check if we should attempt to stitch together past frames
  std::vector< vital::track_sptr > all_tracks = input->tracks();
  vital::frame_id_t frame_to_stitch = frame_number - c_new_shot_length + 1;
  double pt = input->percentage_tracked( frame_to_stitch - 1, frame_to_stitch );
  bool stitch_required = ( pt < c_percent_match_req );

  // confirm that the new valid shot criteria length is satisfied
  vital::frame_id_t frame_to_test = frame_to_stitch + 1;
  while( stitch_required && frame_to_test <= frame_number )
  {
    pt = input->percentage_tracked( frame_to_test - 1, frame_to_test );
    stitch_required = ( pt >= c_percent_match_req );
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

  if( frame_to_test > 0 &&
      static_cast< size_t >( frame_to_test ) > c_max_search_length )
  {
    last_frame_to_test = frame_to_test - c_max_search_length;
  }

  auto stitch_frame_set = std::make_shared< vital::feature_track_set >(
    input->active_tracks( frame_to_stitch ) );

  for(; frame_to_test > last_frame_to_test; frame_to_test-- )
  {
    auto test_frame_set = std::make_shared< vital::feature_track_set >(
      input->active_tracks( frame_to_test ) );

    // run matcher alg
    vital::match_set_sptr mset = c_feature_matcher->match(
      test_frame_set->frame_features( frame_to_test ),
      test_frame_set->frame_descriptors( frame_to_test ),
      stitch_frame_set->frame_features( frame_to_stitch ),
      stitch_frame_set->frame_descriptors( frame_to_stitch ) );

    // test matcher results
    auto total_features = test_frame_set->size() + stitch_frame_set->size();

    if( 2 * mset->size() >=
        static_cast< size_t >( c_percent_match_req * total_features ) )
    {
      // modify track history and exit
      std::vector< vital::track_sptr > test_frame_trks =
        test_frame_set->tracks();
      std::vector< vital::track_sptr > stitch_frame_trks =
        stitch_frame_set->tracks();
      std::vector< vital::match > matches = mset->matches();

      for( size_t i = 0; i < matches.size(); i++ )
      {
        input->merge_tracks(
          stitch_frame_trks[ matches[ i ].second ],
          test_frame_trks[ matches[ i ].first ] );
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
