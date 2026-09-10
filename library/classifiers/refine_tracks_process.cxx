// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "refine_tracks_process.h"

#include <viame/algorithm_framework/algo/refine_tracks.h>

#include <viame/pipeline_framework/type_traits.h>
#include <viame/pipeline_framework/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( refiner );

//----------------------------------------------------------------
// Private implementation class
class refine_tracks_process::priv
{
public:
  priv();
  ~priv();

   vital::frame_id_t m_current_idx;
   vital::algo::refine_tracks_sptr m_refiner;
}; // end priv class

// ==================================================================
refine_tracks_process::
refine_tracks_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new refine_tracks_process::priv )
{
  make_ports();
  make_config();
}

refine_tracks_process::
~refine_tracks_process()
{
}

// ------------------------------------------------------------------
void
refine_tracks_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if( ! check_nested_algo_configuration_using_trait( refiner, algo_config, d->m_refiner ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  set_nested_algo_configuration_using_trait( refiner, algo_config, d->m_refiner );

  if( ! d->m_refiner )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create refiner" );
  }
}

// ------------------------------------------------------------------
void
refine_tracks_process::
_finalize()
{
  // Give the algorithm a chance to run deferred processing and emit a
  // final result before the pipeline shuts down.
  auto final_tracks = d->m_refiner->finalize();
  if( final_tracks && !final_tracks->tracks().empty() )
  {
    push_to_port_using_trait( object_track_set, final_tracks );
  }

  mark_process_as_complete();

  const sprokit::datum_t dat = sprokit::datum::complete_datum();

  push_datum_to_port_using_trait( object_track_set, dat );
}

// ------------------------------------------------------------------
void
refine_tracks_process::
_step()
{
  vital::image_container_sptr image;
  vital::timestamp timestamp;
  vital::object_track_set_sptr tracks;
  vital::frame_id_t cur_frame_id;

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
    auto port_check = peek_at_port_using_trait( image );

    if( port_check.datum->type() == sprokit::datum::complete )
    {
      this->_finalize();
      return;
    }

    image = grab_from_port_using_trait( image );
  }

  if( has_input_port_edge_using_trait( timestamp ) )
  {
    auto port_check = peek_at_port_using_trait( timestamp );

    if( port_check.datum->type() == sprokit::datum::complete )
    {
      this->_finalize();
      return;
    }

    timestamp = grab_from_port_using_trait( timestamp );

    if( timestamp.has_valid_frame() )
    {
      cur_frame_id = timestamp.get_frame();
    }
  }
  else
  {
    cur_frame_id = d->m_current_idx;
    timestamp.set_frame( cur_frame_id );
  }

  vital::object_track_set_sptr output_tracks;

  {
    scoped_step_instrumentation();

    // Refine tracks for this frame.  When no input tracks are
    // connected, pass an empty set so the refiner can still detect
    // new objects.
    if( !tracks )
    {
      tracks = std::make_shared< kwiver::vital::object_track_set >();
    }
    output_tracks = d->m_refiner->refine( timestamp, image, tracks );
  }

  push_to_port_using_trait( object_track_set, output_tracks );

  d->m_current_idx++;
}

// ------------------------------------------------------------------
void
refine_tracks_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  // -- input --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( object_track_set, optional );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}

// ------------------------------------------------------------------
void
refine_tracks_process::
make_config()
{
  declare_config_using_trait( refiner );
}

// ================================================================
refine_tracks_process::priv
::priv()
  : m_current_idx( 0 )
{
}

refine_tracks_process::priv
::~priv()
{
}

} // end namespace kwiver
