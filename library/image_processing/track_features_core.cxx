// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Implementation of track_features_core

#include "track_features_core.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <viame/image_processing/track_set_impl.h>

#include <viame/algorithm_framework/algo/close_loops.h>
#include <viame/algorithm_framework/algo/detect_features.h>
#include <viame/algorithm_framework/algo/extract_descriptors.h>
#include <viame/algorithm_framework/algo/feature_descriptor_io.h>
#include <viame/algorithm_framework/algo/match_features.h>

#include <viame/algorithm_framework/io/metadata_io.h>

#include <viame/algorithm_framework/exceptions/algorithm.h>
#include <viame/algorithm_framework/exceptions/image.h>

#include <kwiversys/SystemTools.hxx>

using namespace kwiver::vital;
typedef kwiversys::SystemTools ST;

namespace kwiver {

namespace arrows {

namespace core {

/// Private implementation class
class track_features_core::priv
{
public:
  priv( track_features_core& parent )
    : parent( parent )
  {}

  track_features_core& parent;

  // Configuration values
  config_path_t features_dir() { return parent.c_features_dir; }

  // processing classes
  vital::algo::detect_features_sptr
  feature_detector()
  {
    return parent.c_feature_detector;
  }

  vital::algo::extract_descriptors_sptr
  descriptor_extractor()
  {
    return parent.c_descriptor_extractor;
  }

  vital::algo::feature_descriptor_io_sptr
  feature_io()
  {
    return parent.c_feature_io;
  }

  vital::algo::match_features_sptr
  feature_matcher()
  {
    return parent.c_feature_matcher;
  }

  vital::algo::close_loops_sptr loop_closer() { return parent.c_loop_closer; }
};

// ----------------------------------------------------------------------------
void
track_features_core
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d_ );
  attach_logger( "arrows.core.track_features_core" );
}

/// Destructor
track_features_core
::~track_features_core() // noexcept
{}

bool
track_features_core
::check_configuration( vital::config_block_sptr config ) const
{
  bool config_valid = true;
  // this algorithm is optional
  if( config->has_value( "loop_loop_closer" ) &&
      config->get_value< std::string >( "loop_closer" ) != "" &&
      !check_nested_algo_configuration< algo::close_loops >(
        "loop_closer",
        config ) )
  {
    config_valid = false;
  }

  if( config->has_value( "features_dir" ) &&
      config->get_value< std::string >( "features_dir" ) != "" )
  {
    config_path_t fp = config->get_value< config_path_t >( "features_dir" );
    if( ST::FileExists( fp ) && !ST::FileIsDirectory( fp ) )
    {
      LOG_ERROR(
        logger(), "Given features directory is a file "
                  "(Given: " << fp << ")" );
      config_valid = false;
    }
  }

  // this algorithm is optional
  if( config->has_value( "feature_io" ) &&
      config->get_value< std::string >( "feature_io" ) != "" &&
      !check_nested_algo_configuration< algo::feature_descriptor_io >(
        "feature_io", config ) )
  {
    config_valid = false;
  }
  return (
    check_nested_algo_configuration< algo::detect_features >(
      "feature_detector",
      config )
    &&
    check_nested_algo_configuration< algo::extract_descriptors >(
      "descriptor_extractor", config )
    &&
    check_nested_algo_configuration< algo::match_features >(
      "feature_matcher",
      config )
    &&
    config_valid
  );
}

/// Extend a previous set of tracks using the current frame
feature_track_set_sptr
track_features_core
::track(
  feature_track_set_sptr prev_tracks,
  frame_id_t frame_number,
  image_container_sptr image_data,
  image_container_sptr mask ) const
{
  // verify that all dependent algorithms have been initialized
  if( !d_->feature_detector() || !d_->descriptor_extractor() ||
      !d_->feature_matcher() )
  {
    // Something did not initialize
    VITAL_THROW(
      vital::algorithm_configuration_exception, this->interface_name(),
      this->plugin_name(),
      "not all sub-algorithms have been initialized" );
  }

  // Check that the given mask, when non-zero, matches the size of the image
  // data provided
  if( mask && mask->size() > 0 &&
      ( image_data->width() != mask->width() ||
        image_data->height() != mask->height() ) )
  {
    VITAL_THROW(
      image_size_mismatch_exception,
      "Core track feature algorithm given a non-zero mask image that is "
      "not the same shape as the provided image data.",
      image_data->width(), image_data->height(),
      mask->width(), mask->height()
    );
  }

  std::vector< track_sptr > existing_tracks;
  feature_set_sptr curr_feat;
  descriptor_set_sptr curr_desc;

  // see if there are already existing tracks on this frame
  if( prev_tracks )
  {
    existing_tracks = prev_tracks->active_tracks( frame_number );
    if( !existing_tracks.empty() )
    {
      LOG_DEBUG( logger(), "Using existing features on frame " << frame_number);
      // use existing features
      curr_feat = prev_tracks->frame_features( frame_number );

      // use existng descriptors
      curr_desc = prev_tracks->frame_descriptors( frame_number );
    }
  }

  // see if there are existing features cached on disk
  if( ( !curr_feat || curr_feat->size() == 0 ||
        !curr_desc || curr_desc->size() == 0 ) &&
      d_->feature_io() && d_->features_dir() != "" )
  {
    metadata_sptr md = image_data->get_metadata();
    std::string basename = basename_from_metadata( md, frame_number );
    path_t kwfd_file = d_->features_dir() + "/" + basename + ".kwfd";
    if( ST::FileExists( kwfd_file ) )
    {
      feature_set_sptr feat;
      descriptor_set_sptr desc;
      d_->feature_io()->load( kwfd_file, feat, desc );
      if( feat && feat->size() > 0 && desc && desc->size() > 0 )
      {
        LOG_DEBUG(
          logger(), "Loaded features on frame " << frame_number
                                                << " from " << kwfd_file );
        // Handle the special case where feature were loaded from a track
        // file without descriptors. If the number of features from both
        // sources matches, then assign just the descriptors
        if( ( curr_feat && curr_feat->size() > 0 ) &&
            ( !curr_desc || curr_desc->size() == 0 ) &&
            ( curr_feat->size() == feat->size() ) )
        {
          curr_desc = desc;

          // assign these descriptors to the existing track states
          auto track_states = prev_tracks->frame_states( frame_number );
          if( curr_desc->size() == track_states.size() )
          {
            for( size_t i = 0; i < track_states.size(); ++i )
            {
              auto fts =
                std::dynamic_pointer_cast< feature_track_state >(
                  track_states[ i ] );
              if( fts )
              {
                fts->descriptor = curr_desc->at( i );
              }
            }
          }
        }
        else
        {
          curr_feat = feat;
          curr_desc = desc;
        }
      }
    }
  }

  // compute features and descriptors from the image
  bool features_computed = false;
  if( !curr_feat || curr_feat->size() == 0 )
  {
    LOG_DEBUG( logger(), "Computing new features on frame " << frame_number);
    // detect features on the current frame
    curr_feat = d_->feature_detector()->detect( image_data, mask );
    features_computed = true;
  }
  if( !curr_desc || curr_desc->size() == 0 )
  {
    LOG_DEBUG( logger(), "Computing new descriptors on frame " << frame_number);
    // extract descriptors on the current frame
    curr_desc = d_->descriptor_extractor()->extract(
      image_data, curr_feat,
      mask );
    features_computed = true;
  }

  // cache features if they were just computed and feature I/O is enabled
  if( features_computed && d_->feature_io() && d_->features_dir() != "" )
  {
    metadata_sptr md = image_data->get_metadata();
    std::string basename = basename_from_metadata( md, frame_number );
    path_t kwfd_file = d_->features_dir() + "/" + basename + ".kwfd";

    // make the enclosing directory if it does not already exist
    const kwiver::vital::path_t fd_dir = ST::GetFilenamePath( kwfd_file );
    if( !ST::FileIsDirectory( fd_dir ) )
    {
      if( !ST::MakeDirectory( fd_dir ) )
      {
        LOG_ERROR( logger(), "Unable to create directory: " << fd_dir );
      }
    }
    d_->feature_io()->save( kwfd_file, curr_feat, curr_desc );
    LOG_DEBUG(
      logger(), "Saved features on frame " << frame_number
                                           << " to " << kwfd_file );
  }

  std::vector< feature_sptr > vf = curr_feat->features();

  track_id_t next_track_id = 0;

  // special case for the first frame
  if( !prev_tracks )
  {
    typedef std::vector< feature_sptr >::const_iterator feat_itr;
    typedef descriptor_set::const_iterator desc_itr;

    feat_itr fit = vf.begin();
    desc_itr dit = curr_desc->cbegin();
    std::vector< vital::track_sptr > new_tracks;
    for(; fit != vf.end() && dit != curr_desc->cend(); ++fit, ++dit )
    {
      auto fts = std::make_shared< feature_track_state >( frame_number );
      fts->feature = *fit;
      fts->descriptor = *dit;
      new_tracks.push_back( vital::track::create() );
      new_tracks.back()->append( fts );
      new_tracks.back()->set_id( next_track_id++ );
    }

    // create a new track set since one was not provided
    // use the frame-indexed track_set implementation, which is more efficient
    // for querying tracks by frame number
    typedef std::unique_ptr< track_set_implementation > tsi_uptr;

    auto new_track_set = std::make_shared< feature_track_set >(
      tsi_uptr( new frame_index_track_set_impl( new_tracks ) ) );

    if( d_->loop_closer() )
    {
      // call loop closure on the first frame to establish this
      // frame as the first frame for loop closing purposes
      return d_->loop_closer()->stitch(
        frame_number,
        new_track_set,
        image_data, mask );
    }
    return new_track_set;
  }

  if( !prev_tracks->empty() )
  {
    // get the last track id in the existing set of tracks and increment it
    next_track_id = ( *prev_tracks->all_track_ids().crbegin() ) + 1;
  }

  const vital::frame_id_t last_frame = prev_tracks->last_frame();
  vital::frame_id_t prev_frame = last_frame;

  feature_track_set_sptr active_set;
  // if processing out of order, see if there are tracks on the previous frame
  // and prefer those over the last frame (i.e. largest frame number)
  if( prev_frame >= frame_number && frame_number > 0 )
  {
    auto const frame_ids = prev_tracks->all_frame_ids();
    auto it = frame_ids.lower_bound( frame_number );
    if( it == frame_ids.begin() )
    {
      return prev_tracks;
    }
    --it;
    active_set = std::make_shared< feature_track_set >(
      prev_tracks->active_tracks( *it ) );
    if( active_set->size() > 0 )
    {
      prev_frame = *it;
    }
  }
  if( !active_set )
  {
    active_set = std::make_shared< feature_track_set >(
      prev_tracks->active_tracks( prev_frame ) );
  }

  // detect features on the previous frame
  feature_set_sptr prev_feat = active_set->frame_features( prev_frame );
  // extract descriptors on the previous frame
  descriptor_set_sptr prev_desc = active_set->frame_descriptors( prev_frame );

  // match features to from the previous to the current frame
  match_set_sptr mset = d_->feature_matcher()->match(
    prev_feat, prev_desc,
    curr_feat, curr_desc );
  if( !mset )
  {
    LOG_WARN(
      logger(), "Feature matching between frames " << prev_frame <<
        " and " << frame_number << " failed" );
    return prev_tracks;
  }

  std::vector< track_sptr > active_tracks = active_set->tracks();
  std::vector< match > vm = mset->matches();

  feature_track_set_sptr updated_track_set = prev_tracks;
  // if we previously had tracks on this frame, stitch to a previous frame
  if( !existing_tracks.empty() )
  {
    int num_linked = 0;
    for( match m : vm )
    {
      track_sptr tp = active_tracks[ m.first ];
      track_sptr tc = existing_tracks[ m.second ];
      if( updated_track_set->merge_tracks( tc, tp ) )
      {
        ++num_linked;
      }
    }
    LOG_DEBUG(
      logger(), "Stitched " << num_linked <<
        " existing tracks from frame " << frame_number <<
        " to " << prev_frame );
  }
  else
  {
    std::set< size_t > matched;

    for( match m : vm )
    {
      track_sptr t = active_tracks[ m.first ];
      auto fts = std::make_shared< feature_track_state >( frame_number );
      fts->feature = vf[ m.second ];
      fts->descriptor = curr_desc->at( m.second );
      if( t->append( fts ) || t->insert( fts ) )
      {
        matched.insert( m.second );
        // need to notify the track_set of new states appended to
        // tracks that were previously added
        updated_track_set->notify_new_state( fts );
      }
    }

    // find the set of unmatched active track indices
    std::vector< size_t > unmatched;
    std::back_insert_iterator< std::vector< size_t > >
    unmatched_insert_itr( unmatched );

    //
    // Generate a sequence of numbers
    //
    std::vector< size_t > sequence( vf.size() );
    std::iota( sequence.begin(), sequence.end(), 0 );

    std::set_difference(
      sequence.begin(), sequence.end(),
      matched.begin(), matched.end(),
      unmatched_insert_itr );

    for( size_t i : unmatched )
    {
      auto fts = std::make_shared< feature_track_state >( frame_number );
      fts->feature = vf[ i ];
      fts->descriptor = curr_desc->at( i );

      auto t = vital::track::create();
      t->append( fts );
      t->set_id( next_track_id++ );
      updated_track_set->insert( t );
    }
  }

  // run loop closure if enabled
  if( d_->loop_closer() )
  {
    updated_track_set = d_->loop_closer()->stitch(
      frame_number,
      updated_track_set,
      image_data, mask );
  }

  return updated_track_set;
}

} // end namespace core

} // end namespace arrows

} // end namespace kwiver
