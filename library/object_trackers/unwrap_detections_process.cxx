// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Unwrap the object detections from object tracks.

#include "unwrap_detections_process.h"

#include <viame/core_types/vital_types.h>

#include <viame/pipeline_framework/type_traits.h>
#include <viame/pipeline_framework/process_exception.h>

namespace kwiver {

// ----------------------------------------------------------------
// Private implementation class
class unwrap_detections_process::priv
{
public:
  priv();
  ~priv();


  vital::frame_id_t m_current_idx;
};

// ===============================================================================

unwrap_detections_process
::unwrap_detections_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new unwrap_detections_process::priv )
{
  make_ports();
  make_config();
}

unwrap_detections_process
::~unwrap_detections_process()
{}

// -------------------------------------------------------------------------------
void
unwrap_detections_process
::_configure()
{}

// -------------------------------------------------------------------------------
void
unwrap_detections_process
::_step()
{
  auto object_tracks = grab_from_port_using_trait( object_track_set );
  auto detected_objects =
    std::make_shared< kwiver::vital::detected_object_set >();

  if( object_tracks )
  {
    for( auto& trk : object_tracks->tracks() )
    {
      for( auto& state : *trk )
      {
        auto obj_state =
          std::static_pointer_cast< kwiver::vital::object_track_state >(
            state );

        if( state->frame() == d->m_current_idx )
        {
          detected_objects->add( obj_state->detection() );
        }
      }
    }
  }

  push_to_port_using_trait( detected_object_set, detected_objects );

  d->m_current_idx++;
}

// -------------------------------------------------------------------------------
void
unwrap_detections_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( object_track_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, required );
}

// -------------------------------------------------------------------------------
void
unwrap_detections_process
::make_config()
{}

// ===============================================================================
unwrap_detections_process::priv
::priv()
  : m_current_idx( 0 )
{}

unwrap_detections_process::priv
::~priv()
{}

} // end namespace
