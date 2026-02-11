/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Write query results as object track CSV
 */

#include "write_query_results_as_tracks_process.h"

#include <vital/algo/algorithm.txx>

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
#include <vital/algo/write_object_track_set.h>

#include <map>


namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

create_config_trait( output_label, std::string, "query_result",
  "Label to assign to output detections" );

create_config_trait( use_relevancy_as_confidence, bool, "true",
  "Use the query result relevancy score as the detection confidence" );

create_config_trait( file_name, std::string, "",
  "Output file name for writing tracks. If empty, no file is written." );

create_algorithm_name_config_trait( writer );

// Structure to hold tracks grouped by frame
struct frame_track_info
{
  kv::timestamp timestamp;
  std::string stream_id;
  std::vector< kv::track_sptr > tracks;
};

//------------------------------------------------------------------------------
// Private implementation class
class write_query_results_as_tracks_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  std::string m_output_label;
  bool m_use_relevancy_as_confidence;
  std::string m_file_name;

  // Track writer algorithm
  kv::algo::write_object_track_set_sptr m_writer;

  // Track ID counter
  kv::track_id_t m_next_track_id;

  // Collected frames for writing (frame_id -> frame_track_info)
  std::map< kv::frame_id_t, frame_track_info > m_frame_tracks;
};

// =============================================================================

write_query_results_as_tracks_process
::write_query_results_as_tracks_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new write_query_results_as_tracks_process::priv() )
{
  make_ports();
  make_config();

  // Set data checking level to ensure this sink process is stepped
  set_data_checking_level( check_sync );
}


write_query_results_as_tracks_process
::~write_query_results_as_tracks_process()
{
  // Write and close writer if still open (safety net for cases where
  // completion signal wasn't received)
  if( d->m_writer && !d->m_frame_tracks.empty() )
  {
    for( auto& frame_pair : d->m_frame_tracks )
    {
      auto& info = frame_pair.second;
      auto track_set = std::make_shared< kv::object_track_set >( info.tracks );
      d->m_writer->write_set( track_set, info.timestamp, info.stream_id );
    }
    d->m_writer->close();
  }
}


// -----------------------------------------------------------------------------
void
write_query_results_as_tracks_process
::_configure()
{
  d->m_output_label = config_value_using_trait( output_label );
  d->m_use_relevancy_as_confidence = config_value_using_trait( use_relevancy_as_confidence );
  d->m_file_name = config_value_using_trait( file_name );

  // Configure the writer if file name is specified
  if( !d->m_file_name.empty() )
  {
    kv::config_block_sptr algo_config = get_config();

    set_nested_algo_configuration_using_trait( writer, algo_config, d->m_writer );

    if( !d->m_writer )
    {
      LOG_WARN( logger(), "Unable to create track writer, will not write output file" );
    }
  }
}


// -----------------------------------------------------------------------------
void
write_query_results_as_tracks_process
::_init()
{
  // Open the writer file
  if( d->m_writer && !d->m_file_name.empty() )
  {
    d->m_writer->open( d->m_file_name );
  }
}


// -----------------------------------------------------------------------------
void
write_query_results_as_tracks_process
::_step()
{
  // Check for completion signal
  auto const& p_info = peek_at_port_using_trait( query_result );

  if( p_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( query_result );

    // Write all accumulated tracks grouped by frame
    if( d->m_writer )
    {
      for( auto& frame_pair : d->m_frame_tracks )
      {
        auto& info = frame_pair.second;
        auto track_set = std::make_shared< kv::object_track_set >( info.tracks );
        d->m_writer->write_set( track_set, info.timestamp, info.stream_id );
      }
      d->m_writer->close();
      d->m_writer.reset();
      d->m_frame_tracks.clear();
    }

    mark_process_as_complete();
    return;
  }

  kv::query_result_set_sptr query_results = grab_from_port_using_trait( query_result );

  if( query_results )
  {

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

            // Store frame info for writing
            if( d->m_frame_tracks.find( frame ) == d->m_frame_tracks.end() )
            {
              frame_track_info info;
              info.timestamp = ts;
              info.stream_id = stream_id;
              d->m_frame_tracks[ frame ] = info;
            }
          }

          // Only add track if it has states
          if( trk->size() > 0 )
          {
            // Add to frame-grouped tracks for writing
            kv::frame_id_t last_frame = trk->back()->frame();
            d->m_frame_tracks[ last_frame ].tracks.push_back( trk );
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

          kv::frame_id_t last_frame = 0;

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

            // Store frame info for writing
            kv::frame_id_t frame = ots->frame();
            if( d->m_frame_tracks.find( frame ) == d->m_frame_tracks.end() )
            {
              kv::timestamp ts( ots->time(), frame );
              frame_track_info info;
              info.timestamp = ts;
              info.stream_id = stream_id;
              d->m_frame_tracks[ frame ] = info;
            }

            last_frame = frame;
          }

          if( new_trk->size() > 0 )
          {
            d->m_frame_tracks[ last_frame ].tracks.push_back( new_trk );
          }
        }
      }
    }
  }
}


// -----------------------------------------------------------------------------
void
write_query_results_as_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( query_result, required );
}


// -----------------------------------------------------------------------------
void
write_query_results_as_tracks_process
::make_config()
{
  declare_config_using_trait( output_label );
  declare_config_using_trait( use_relevancy_as_confidence );
  declare_config_using_trait( file_name );
  declare_config_using_trait( writer );
}


// =============================================================================
write_query_results_as_tracks_process::priv
::priv()
  : m_output_label( "query_result" )
  , m_use_relevancy_as_confidence( true )
  , m_file_name( "" )
  , m_next_track_id( 0 )
{
}


write_query_results_as_tracks_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
