// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "matcher_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>
#include <vital/types/feature_track_set.h>

#include <vital/algo/match_features.h>
#include <vital/algo/close_loops.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

#include <boost/iterator/counting_iterator.hpp>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( feature_matcher );
create_algorithm_name_config_trait( loop_closer );

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
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Instantiate the configured algorithm
  algo::match_features::set_nested_algo_configuration_using_trait(
    feature_matcher,
    algo_config,
    d->m_matcher );
  if ( ! d->m_matcher )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create feature_matcher" );
  }

  algo::match_features::get_nested_algo_configuration_using_trait(
    feature_matcher,
    algo_config,
    d->m_matcher );

  // Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::match_features::check_nested_algo_configuration_using_trait(
         feature_matcher, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  // - Loop closure algorithm
  algo::close_loops::set_nested_algo_configuration_using_trait(
    loop_closer,
    algo_config,
    d->m_closer );
  if ( ! d->m_closer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create loop_closer" );
  }

  algo::close_loops::get_nested_algo_configuration_using_trait(
    loop_closer,
    algo_config,
    d->m_closer );

  if ( ! algo::close_loops::check_nested_algo_configuration_using_trait(
         loop_closer, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
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

  {
    scoped_step_instrumentation();

    // LOG_DEBUG - this is a good thing to have in all processes that handle frames.
    LOG_DEBUG( logger(), "Processing frame " << frame_time );

    auto frame_number = frame_time.get_frame();
    std::vector<vital::feature_sptr> vf = curr_feat->features();
    // special case for the first frame
    if ( ! d->m_curr_tracks )
    {
      typedef std::vector< vital::feature_sptr >::const_iterator feat_itr;

      feat_itr fit = vf.begin();
      auto dit = curr_desc->cbegin();

      std::vector< vital::track_sptr > new_tracks;
      for ( ; fit != vf.end() && dit != curr_desc->cend(); ++fit, ++dit )
      {
        auto ts = std::make_shared<vital::feature_track_state>( frame_number, *fit, *dit );
        new_tracks.push_back( vital::track::create() );
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

      for( vital::match m : vm )
      {
        matched.insert( m.second );
        vital::track_sptr t = active_tracks[m.first];
        auto ts = std::make_shared<vital::feature_track_state>( frame_number,
                                                                vf[m.second],
                                                                curr_desc->at(m.second) );
        t->append( ts );
      }

      // find the set of unmatched active track indices
      std::vector< unsigned > unmatched;
      std::back_insert_iterator< std::vector< unsigned > > unmatched_insert_itr( unmatched );
      std::set_difference( boost::counting_iterator< unsigned > ( 0 ),
                           boost::counting_iterator< unsigned > ( static_cast< unsigned int > ( vf.size() ) ),
                           matched.begin(), matched.end(),
                           unmatched_insert_itr );

      for( unsigned i : unmatched )
      {
        auto ts = std::make_shared<vital::feature_track_state>( frame_number,
                                                                vf[i],
                                                                curr_desc->at(i) );

        all_tracks.push_back( vital::track::create() );
        all_tracks.back()->append( ts );
        all_tracks.back()->set_id( d->m_next_track_id++ );
      }

      d->m_curr_tracks = d->m_closer->stitch( frame_number,
                                              std::make_shared<vital::feature_track_set>( all_tracks ),
                                              image_data );
    }
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
  declare_config_using_trait( feature_matcher );
  declare_config_using_trait( loop_closer );
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
