// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "refine_detections_process.h"

#include <viame/algorithm_framework/algo/refine_detections.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/timestamp.h>

#include <viame/pipeline_framework/type_traits.h>
#include <viame/pipeline_framework/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( refiner );

//----------------------------------------------------------------
// Private implementation class
class refine_detections_process::priv
{
public:
  priv();
  ~priv();

   vital::frame_id_t m_current_idx;
   vital::algo::refine_detections_sptr m_refiner;

}; // end priv class

// ==================================================================
refine_detections_process::
refine_detections_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new refine_detections_process::priv )
{
  make_ports();
  make_config();
}

refine_detections_process::
~refine_detections_process()
{
}

// ------------------------------------------------------------------
void
refine_detections_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if ( ! check_nested_algo_configuration_using_trait(
         refiner, algo_config, d->m_refiner ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  set_nested_algo_configuration_using_trait(
    refiner,
    algo_config,
    d->m_refiner );

  if ( ! d->m_refiner )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create refiner" );
  }
}

// ------------------------------------------------------------------
void
refine_detections_process::
_finalize()
{
  mark_process_as_complete();

  const sprokit::datum_t dat = sprokit::datum::complete_datum();

  push_datum_to_port_using_trait( detected_object_set, dat );
  push_datum_to_port_using_trait( object_track_set, dat );
}

// ------------------------------------------------------------------
void
refine_detections_process::
_step()
{
  vital::image_container_sptr image;
  vital::timestamp timestamp;
  vital::detected_object_set_sptr dets;
  vital::object_track_set_sptr tracks;
  vital::frame_id_t cur_frame_id = 0;

  // Every input is optional: an nms or add_fixed refiner works on the
  // detections alone, and a track refiner takes the track set instead. Peek
  // before grabbing so the complete datum is recognised and forwarded rather
  // than consumed as data -- without that the downstream processes never learn
  // the stream ended, and a pipeline that fans one image out to several
  // refiners and merges the results deadlocks with every thread waiting.
  if( has_input_port_edge_using_trait( detected_object_set ) )
  {
    auto port_check = peek_at_port_using_trait( detected_object_set );

    if( port_check.datum->type() == sprokit::datum::complete )
    {
      this->_finalize();
      return;
    }

    dets = grab_from_port_using_trait( detected_object_set );
  }

  if( has_input_port_edge_using_trait( object_track_set ) )
  {
    auto port_check = peek_at_port_using_trait( object_track_set );

    if( port_check.datum->type() == sprokit::datum::complete )
    {
      this->_finalize();
      return;
    }

    tracks = grab_from_port_using_trait( object_track_set );
  }

  if( has_input_port_edge_using_trait( image ) )
  {
    image = grab_from_port_using_trait( image );
  }

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    timestamp = grab_from_port_using_trait( timestamp );

    if( timestamp.has_valid_frame() )
    {
      cur_frame_id = timestamp.get_frame();
    }
  }
  else
  {
    cur_frame_id = d->m_current_idx;
  }

  vital::detected_object_set_sptr output_dets;

  {
    scoped_step_instrumentation();

    if( dets )
    {
      output_dets = d->m_refiner->refine( image, dets );
    }

    if( tracks )
    {
      // Refine the detections belonging to this frame, then write them back
      // into the states they came from so the track set stays intact.
      auto frame_dets = std::make_shared< kwiver::vital::detected_object_set >();

      for( auto& trk : tracks->tracks() )
      {
        for( auto& state : *trk )
        {
          auto obj_state =
            std::static_pointer_cast< kwiver::vital::object_track_state >( state );

          if( state->frame() == cur_frame_id )
          {
            frame_dets->add( obj_state->detection() );
          }
        }
      }

      frame_dets = d->m_refiner->refine( image, frame_dets );

      if( !dets )
      {
        output_dets = frame_dets;
      }

      auto dets_itr = frame_dets->begin();

      for( auto& trk : tracks->tracks() )
      {
        for( auto& state : *trk )
        {
          auto obj_state =
            std::static_pointer_cast< kwiver::vital::object_track_state >( state );

          if( state->frame() == cur_frame_id )
          {
            obj_state->set_detection( *dets_itr++ );
          }
        }
      }
    }
  }

  push_to_port_using_trait( detected_object_set, output_dets );
  push_to_port_using_trait( object_track_set, tracks );

  d->m_current_idx++;
}

// ------------------------------------------------------------------
void
refine_detections_process::
make_ports()
{
  sprokit::process::port_flags_t optional;

  // Nothing is required: which inputs are connected depends on whether the
  // refiner works on detections, on tracks, and on whether it needs the image.
  // -- input --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( detected_object_set, optional );
  declare_input_port_using_trait( object_track_set, optional );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
  declare_output_port_using_trait( object_track_set, optional );
}

// ------------------------------------------------------------------
void
refine_detections_process::
make_config()
{
  declare_config_using_trait( refiner );
}

// ================================================================
refine_detections_process::priv
::priv()
  : m_current_idx( 0 )
{
}

refine_detections_process::priv
::~priv()
{
}

} // end namespace kwiver
