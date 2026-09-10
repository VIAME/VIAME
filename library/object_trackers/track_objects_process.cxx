// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_objects_process.h"

#include <viame/core_types/vital_types.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/timestamp_config.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/object_track_set.h>

#include <viame/algorithm_framework/algo/track_objects.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

create_algorithm_name_config_trait( track_objects );

/**
 * \class track_objects_process
 *
 * \brief Track detected objects across video frames.
 *
 * \process This process tracks detected objects across video frames
 * using a configurable track_objects algorithm implementation. The
 * actual tracking is done by the selected \b track_objects algorithm
 * implementation (e.g., ByteTrack, DeepSORT, SAM3, etc.)
 *
 * \iports
 *
 * \iport{timestamp} Timestamp for incoming images.
 *
 * \iport{image} Input image to be processed.
 *
 * \iport{detected_object_set} Detected objects from current frame.
 *
 * \oports
 *
 * \oport{object_track_set} Set of tracked objects.
 *
 * \configs
 *
 * \config{track_objects} Name of the configuration subblock that selects
 * and configures the track_objects algorithm.
 */

//----------------------------------------------------------------
// Private implementation class
class track_objects_process::priv
{
public:
  priv();
  ~priv();

  algo::track_objects_sptr m_tracker;

}; // end priv class

// ================================================================

 track_objects_process
::track_objects_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new track_objects_process::priv )
{
  make_ports();
  make_config();
}

 track_objects_process
::~track_objects_process()
{
}

// ----------------------------------------------------------------
void track_objects_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic if any config problems
  // are found
  if ( ! check_nested_algo_configuration_using_trait( track_objects, algo_config, d->m_tracker ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
      "Configuration check failed." );
  }

  // Instantiate the configured algorithm
  set_nested_algo_configuration_using_trait( track_objects, algo_config, d->m_tracker );

  if ( ! d->m_tracker )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
      "Unable to create track_objects" );
  }

}

// ----------------------------------------------------------------
void
track_objects_process
::_step()
{
  // Grab inputs
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  kwiver::vital::detected_object_set_sptr detections =
    grab_from_port_using_trait( detected_object_set );

  kwiver::vital::object_track_set_sptr tracks;

  {
    scoped_step_instrumentation();

    LOG_DEBUG( logger(), "Processing frame " << frame_time );

    // Track objects using the configured algorithm
    tracks = d->m_tracker->track( frame_time, img, detections );
  }

  // Push output
  push_to_port_using_trait( object_track_set, tracks );
}

// ----------------------------------------------------------------
void track_objects_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( object_track_set, optional );
}

// ----------------------------------------------------------------
void track_objects_process
::make_config()
{
  declare_config_using_trait( track_objects );
}

// ================================================================
track_objects_process::priv
::priv()
{
}

track_objects_process::priv
::~priv()
{
}

} // end namespace
