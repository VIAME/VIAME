// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "convert_tracks_to_detections_process.h"

#include <viame/core_types/vital_types.h>

#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>
#include <viame/pipeline_framework/type_traits.h>

namespace kwiver {

create_config_trait(
  frame_ids_only, bool, "false",
  "Only use frame IDs, not entire timestamps, for identifying the current frame." );

// ------------------------------------------------------------------------------
// Private implementation class
class convert_tracks_to_detections_process::priv
{
public:
  priv() : frame_ids_only( false ) {}
  ~priv() {}


  bool frame_ids_only;
};

// =============================================================================

convert_tracks_to_detections_process
::convert_tracks_to_detections_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new convert_tracks_to_detections_process::priv )
{
  set_data_checking_level( check_none );

  make_ports();
  make_config();
}

convert_tracks_to_detections_process
::~convert_tracks_to_detections_process()
{}

// -----------------------------------------------------------------------------
void
convert_tracks_to_detections_process
::_configure()
{
  scoped_configure_instrumentation();


  vital::config_block_sptr algo_config = get_config();

  d->frame_ids_only = config_value_using_trait( frame_ids_only );

  process::_configure();
}

// -----------------------------------------------------------------------------
void
convert_tracks_to_detections_process
::_step()
{
  // Check for complete messages on either input port
  auto ts_port_info = peek_at_port_using_trait( timestamp );
  auto trk_port_info = peek_at_port_using_trait( object_track_set );

  if( ts_port_info.datum->type() == sprokit::datum::complete ||
      trk_port_info.datum->type() == sprokit::datum::complete )
  {
    grab_edge_datum_using_trait( timestamp );
    grab_edge_datum_using_trait( object_track_set );
    mark_process_as_complete();


    const sprokit::datum_t dat = sprokit::datum::complete_datum();
    push_datum_to_port_using_trait( detected_object_set, dat );
    return;
  }


  // Retrieve inputs from ports
  vital::timestamp ts = grab_from_port_using_trait( timestamp );
  vital::object_track_set_sptr tracks =
    grab_from_port_using_trait( object_track_set );

  // Output frame ID
  LOG_DEBUG( logger(), "Processing frame " << ts );


  // Split track set into detections
  std::vector< vital::detected_object_sptr > output;

  if( tracks )
  {
    for( auto trk_ptr : tracks->tracks() )
    {
      if( trk_ptr && !trk_ptr->empty() )
      {
        kwiver::vital::object_track_state* state =
          dynamic_cast< kwiver::vital::object_track_state* >( trk_ptr->back().
                                                              get() );

        if( state &&
            ( ( d->frame_ids_only && state->frame() == ts.get_frame() ) ||
              ( state->frame() == ts.get_frame() && state->time() == ts.get_time_usec() ) ) )
        {
          output.push_back( state->detection() );
        }
      }
    }
  }

  // Output results
  push_to_port_using_trait(
    detected_object_set,
    std::make_shared< vital::detected_object_set >( output ) );

  process::_step();
}

// -----------------------------------------------------------------------------
void
convert_tracks_to_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( object_track_set, optional );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
}

// -----------------------------------------------------------------------------
void
convert_tracks_to_detections_process
::make_config()
{
  declare_config_using_trait( frame_ids_only );
}

} // end namespace
