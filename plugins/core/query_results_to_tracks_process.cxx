/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Convert query results to object track set
 */

#include "query_results_to_tracks_process.h"

#include <sprokit/processes/kwiver_type_traits.h>

#include <vital/vital_types.h>

#include <vital/types/timestamp.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/object_track_set.h>
#include <vital/types/query_result_set.h>
#include <vital/types/track_descriptor.h>

#include <sstream>
#include <iostream>
#include <limits>


namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

create_config_trait( output_label, std::string, "query_result",
  "Label to assign to output detections" );

create_config_trait( use_relevancy_as_confidence, bool, "true",
  "Use the query result relevancy score as the detection confidence" );

//------------------------------------------------------------------------------
// Private implementation class
class query_results_to_tracks_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_output_label;
  bool m_use_relevancy_as_confidence;

  // Track ID counter
  kv::track_id_t m_next_track_id;
};

// =============================================================================

query_results_to_tracks_process
::query_results_to_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new query_results_to_tracks_process::priv() )
{
  make_ports();
  make_config();
}


query_results_to_tracks_process
::~query_results_to_tracks_process()
{
}


// -----------------------------------------------------------------------------
void
query_results_to_tracks_process
::_configure()
{
  d->m_output_label = config_value_using_trait( output_label );
  d->m_use_relevancy_as_confidence = config_value_using_trait( use_relevancy_as_confidence );
}


// -----------------------------------------------------------------------------
void
query_results_to_tracks_process
::_step()
{
  std::cerr << "[query_results_to_tracks] _step() called" << std::endl;

  // Check for completion signal
  auto const& p_info = peek_at_port_using_trait( query_result );

  if( p_info.datum->type() == sprokit::datum::complete )
  {
    std::cerr << "[query_results_to_tracks] Received completion signal" << std::endl;
    grab_edge_datum_using_trait( query_result );
    mark_process_as_complete();
    return;
  }

  kv::query_result_set_sptr query_results;

  query_results = grab_from_port_using_trait( query_result );

  std::vector< kv::track_sptr > output_tracks;

  std::cerr << "[query_results_to_tracks] query_results is "
            << (query_results ? "not null" : "null") << std::endl;

  if( query_results )
  {
    std::cerr << "[query_results_to_tracks] query_results size: "
              << query_results->size() << std::endl;
    for( auto const& result : *query_results )
    {
      if( !result )
      {
        continue;
      }

      double confidence = d->m_use_relevancy_as_confidence ?
        result->relevancy_score() : 1.0;

      // Get stream_id (source video/image identifier) from result
      std::string stream_id = result->stream_id();

      // Get track descriptors from the result
      kv::track_descriptor_set_sptr descriptors = result->descriptors();

      if( descriptors )
      {
        for( auto const& desc : *descriptors )
        {
          if( !desc )
          {
            continue;
          }

          // Create a new track for this descriptor
          kv::track_sptr trk = kv::track::create();
          trk->set_id( d->m_next_track_id++ );

          // Get history entries (bounding boxes at different frames)
          auto const& history = desc->get_history();

          for( auto const& entry : history )
          {
            kv::bounding_box_d bbox = entry.get_image_location();
            kv::timestamp ts = entry.get_timestamp();

            // Create detected object type
            auto dot = std::make_shared< kv::detected_object_type >();
            dot->set_score( d->m_output_label, confidence );

            // Create detected object
            auto det = std::make_shared< kv::detected_object >(
              bbox, confidence, dot );

            // Create track state
            kv::frame_id_t frame = ts.get_frame();
            kv::time_usec_t time = ts.get_time_usec();

            auto state = std::make_shared< kv::object_track_state >(
              frame, time, det );

            trk->append( state );
          }

          // Only add track if it has states
          if( trk->size() > 0 )
          {
            output_tracks.push_back( trk );
          }
        }
      }

      // Also check if the result has tracks directly
      kv::object_track_set_sptr result_tracks = result->tracks();

      if( result_tracks )
      {
        for( auto const& trk : result_tracks->tracks() )
        {
          if( !trk || trk->empty() )
          {
            continue;
          }

          // Clone the track with updated confidence
          kv::track_sptr new_trk = kv::track::create();
          new_trk->set_id( d->m_next_track_id++ );

          for( auto const& state_ptr : *trk )
          {
            auto ots = std::dynamic_pointer_cast< kv::object_track_state >( state_ptr );
            if( !ots )
            {
              continue;
            }

            kv::detected_object_sptr det = ots->detection();
            if( !det )
            {
              continue;
            }

            // Create new detection with updated confidence
            auto dot = std::make_shared< kv::detected_object_type >();
            dot->set_score( d->m_output_label, confidence );

            auto new_det = std::make_shared< kv::detected_object >(
              det->bounding_box(),
              d->m_use_relevancy_as_confidence ? confidence : det->confidence(),
              dot );

            auto new_state = std::make_shared< kv::object_track_state >(
              ots->frame(), ots->time(), new_det );

            new_trk->append( new_state );
          }

          if( new_trk->size() > 0 )
          {
            output_tracks.push_back( new_trk );
          }
        }
      }
    }
  }

  // Create output track set
  auto output_set = std::make_shared< kv::object_track_set >( output_tracks );

  std::cerr << "[query_results_to_tracks] Outputting " << output_tracks.size()
            << " tracks" << std::endl;

  push_to_port_using_trait( object_track_set, output_set );
}


// -----------------------------------------------------------------------------
void
query_results_to_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( query_result, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}


// -----------------------------------------------------------------------------
void
query_results_to_tracks_process
::make_config()
{
  declare_config_using_trait( output_label );
  declare_config_using_trait( use_relevancy_as_confidence );
}


// =============================================================================
query_results_to_tracks_process::priv
::priv()
  : m_output_label( "query_result" )
  , m_use_relevancy_as_confidence( true )
  , m_next_track_id( 0 )
{
}


query_results_to_tracks_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
