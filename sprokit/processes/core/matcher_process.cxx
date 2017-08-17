/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#include "matcher_process.h"

#include <vital/vital_types.h>
#include <vital/vital_foreach.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>
#include <vital/types/feature_track_set.h>

#include <vital/algo/match_features.h>
#include <vital/algo/close_loops.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

#include <boost/algorithm/string/join.hpp>
#include <boost/iterator/counting_iterator.hpp>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

//----------------------------------------------------------------
// Private implementation class
class matcher_process::priv
{
public:
  priv();
  ~priv();

  unsigned long m_next_track_id;

  vital::feature_track_set_sptr m_curr_tracks;

  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  /// The feature matching algorithm to use
  vital::algo::match_features_sptr m_matcher;

  /// The loop closure algorithm to use
  vital::algo::close_loops_sptr m_closer;

}; // end priv class

// ================================================================

matcher_process
::matcher_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new matcher_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach

  make_ports();
  make_config();
}


matcher_process
::~matcher_process()
{
}


// ----------------------------------------------------------------
void matcher_process
::_configure()
{
  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Instantiate the configured algorithm
  algo::match_features::set_nested_algo_configuration( "feature_matcher", algo_config, d->m_matcher );
  if ( ! d->m_matcher )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create feature_matcher" );
  }

  algo::match_features::get_nested_algo_configuration( "feature_matcher", algo_config, d->m_matcher );

  // Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::match_features::check_nested_algo_configuration( "feature_matcher", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  // - Loop closure algorithm
  algo::close_loops::set_nested_algo_configuration( "loop_closer", algo_config, d->m_closer );
  if ( ! d->m_closer )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create loop_closer" );
  }

  algo::close_loops::get_nested_algo_configuration( "loop_closer", algo_config, d->m_closer );

  if ( ! algo::close_loops::check_nested_algo_configuration( "loop_closer", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }
}


// ----------------------------------------------------------------
void
matcher_process
  ::_step()
{
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );
  kwiver::vital::image_container_sptr image_data = grab_from_port_using_trait( image );
  kwiver::vital::feature_set_sptr curr_feat = grab_from_port_using_trait( feature_set );
  kwiver::vital::descriptor_set_sptr curr_desc = grab_from_port_using_trait( descriptor_set );

  // LOG_DEBUG - this is a good thing to have in all processes that handle frames.
  LOG_DEBUG( logger(), "Processing frame " << frame_time );

  auto frame_number = frame_time.get_frame();
  std::vector<vital::feature_sptr> vf = curr_feat->features();
  std::vector<vital::descriptor_sptr> df = curr_desc->descriptors();

  // special case for the first frame
  if ( ! d->m_curr_tracks )
  {
    typedef std::vector< vital::feature_sptr >::const_iterator feat_itr;
    typedef std::vector< vital::descriptor_sptr >::const_iterator desc_itr;

    feat_itr fit = vf.begin();
    desc_itr dit = df.begin();

    std::vector< vital::track_sptr > new_tracks;
    for ( ; fit != vf.end() && dit != df.end(); ++fit, ++dit )
    {
      auto ts = std::make_shared<vital::feature_track_state>( frame_number, *fit, *dit );
      new_tracks.push_back( vital::track::make() );
      new_tracks.back()->append( ts );
      new_tracks.back()->set_id( d->m_next_track_id++ );
    }
    // call loop closure on the first frame to establish this
    // frame as the first frame for loop closing purposes
    d->m_curr_tracks = d->m_closer->stitch( frame_number,
                                            std::make_shared<vital::feature_track_set>( new_tracks ),
                                            image_data );
  }
  else
  {
    // match features to from the previous to the current frame
    vital::match_set_sptr mset = d->m_matcher->match( d->m_curr_tracks->last_frame_features(),
                                                      d->m_curr_tracks->last_frame_descriptors(),
                                                      curr_feat,
                                                      curr_desc );

    std::vector< vital::track_sptr > active_tracks = d->m_curr_tracks->active_tracks();
    std::vector< vital::track_sptr > all_tracks = d->m_curr_tracks->tracks();
    std::vector< vital::match > vm = mset->matches();
    std::set< unsigned > matched;

    VITAL_FOREACH( vital::match m, vm )
    {
      matched.insert( m.second );
      vital::track_sptr t = active_tracks[m.first];
      auto ts = std::make_shared<vital::feature_track_state>( frame_number, vf[m.second], df[m.second] );
      t->append( ts );
    }

    // find the set of unmatched active track indices
    std::vector< unsigned > unmatched;
    std::back_insert_iterator< std::vector< unsigned > > unmatched_insert_itr( unmatched );
    std::set_difference( boost::counting_iterator< unsigned > ( 0 ),
                         boost::counting_iterator< unsigned > ( static_cast< unsigned int > ( vf.size() ) ),
                         matched.begin(), matched.end(),
                         unmatched_insert_itr );

    VITAL_FOREACH( unsigned i, unmatched )
    {
      auto ts = std::make_shared<vital::feature_track_state>( frame_number, vf[i], df[i] );

      all_tracks.push_back( vital::track::make() );
      all_tracks.back()->append( ts );
      all_tracks.back()->set_id( d->m_next_track_id++ );
    }

    d->m_curr_tracks = d->m_closer->stitch( frame_number,
                  std::make_shared<vital::feature_track_set>( all_tracks ),
                  image_data );
  }

  // push outputs
  push_to_port_using_trait( feature_track_set, d->m_curr_tracks );
} // matcher_process::_step


// ----------------------------------------------------------------
void matcher_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( feature_set, required );
  declare_input_port_using_trait( descriptor_set, required );

  // -- output --
  declare_output_port_using_trait( feature_track_set, optional );
}


// ----------------------------------------------------------------
void matcher_process
::make_config()
{
}


// ================================================================
matcher_process::priv
::priv()
  : m_next_track_id( 0 )
{
}


matcher_process::priv
::~priv()
{
}

} // end namespace
