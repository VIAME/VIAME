/*ckwg +29
 * Copyright 2017, 2020 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#include "detect_motion_process.h"

#include <vital/util/wall_timer.h>
#include <vital/algo/detect_motion.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

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
  if ( ! vital::algo::detect_motion::check_nested_algo_configuration_using_trait( algo, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  vital::algo::detect_motion::set_nested_algo_configuration_using_trait( algo, algo_config, d->m_algo );

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
