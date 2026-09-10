// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "detect_motion_process.h"

#include <viame/algorithm_framework/util/wall_timer.h>
#include <viame/algorithm_framework/algo/detect_motion.h>

#include <viame/pipeline_framework/type_traits.h>
#include <viame/pipeline_framework/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( algo );

//----------------------------------------------------------------
// Private implementation class
class detect_motion_process::priv
{
public:
  priv();
  ~priv();

  vital::algo::detect_motion_sptr m_algo;
  kwiver::vital::wall_timer m_timer;

}; // end priv class

// ==================================================================
detect_motion_process::
detect_motion_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detect_motion_process::priv )
{
  make_ports();
  make_config();
}

detect_motion_process::
~detect_motion_process()
{
}

// ------------------------------------------------------------------
void
detect_motion_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if ( ! check_nested_algo_configuration_using_trait( algo, algo_config, d->m_algo ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  set_nested_algo_configuration_using_trait( algo, algo_config, d->m_algo );

  if ( ! d->m_algo )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(),
                 "Unable to create motion detector algorithm" );
  }
}

// ------------------------------------------------------------------
void
detect_motion_process::
_step()
{
  d->m_timer.start();

  auto input = grab_from_port_using_trait( image );
  kwiver::vital::image_container_sptr result;

  kwiver::vital::timestamp ts;
  if (has_input_port_edge_using_trait( timestamp ) )
  {
    ts = grab_from_port_using_trait( timestamp );
  }

  bool reset(false);
  if (has_input_port_edge_using_trait( coordinate_system_updated ) )
  {
    reset = grab_from_port_using_trait( coordinate_system_updated );
  }

  {
    scoped_step_instrumentation();
    result = d->m_algo->process_image( ts, input, reset );
  }

  push_to_port_using_trait(motion_heat_map , result );

  d->m_timer.stop();
  double elapsed_time = d->m_timer.elapsed();
  LOG_DEBUG( logger(), "Total processing time: " << elapsed_time << " seconds");
}

// ------------------------------------------------------------------
void
detect_motion_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, optional );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( coordinate_system_updated, optional );

  // -- output --
  declare_output_port_using_trait( motion_heat_map, required );
}

// ------------------------------------------------------------------
void
detect_motion_process::
make_config()
{
  declare_config_using_trait( algo );
}

// ================================================================
detect_motion_process::priv
::priv()
{
}

detect_motion_process::priv
::~priv()
{
}

} // end namespace kwiver
