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

#include "compute_stereo_depth_map_process.h"

#include <vital/algo/compute_stereo_depth_map.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

create_algorithm_name_config_trait( computer );

// ----------------------------------------------------------------
/**
 * \class compute_stereo_depth_map_process
 *
 * \brief Compute stereo depth map image.
 *
 * \process This process generates a depth map image from a pair of
 * stereo images. The actual calculation is done by the selected \b
 * compute_stereo_depth_map algorithm implementation.
 *
 * \iports
 *
 * \iport{left_image} Left image of the stereo image pair.
 *
 * \iport{right_image} Right image if the stereo pair.
 *
 * \oports
 *
 * \oport{depth_map} Resulting depth map.
 *
 * \configs
 *
 * \config{computer} Name of the configuration subblock that selects
 * and configures the algorithm.
 */

//----------------------------------------------------------------
// Private implementation class
class compute_stereo_depth_map_process::priv
{
public:
  priv();
  ~priv();

   vital::algo::compute_stereo_depth_map_sptr m_computer;

}; // end priv class


// ==================================================================
compute_stereo_depth_map_process::
compute_stereo_depth_map_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new compute_stereo_depth_map_process::priv )
{
  make_ports();
  make_config();
}


compute_stereo_depth_map_process::
~compute_stereo_depth_map_process()
{
}


// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
_configure()
{
  scoped_configure_instrumentation();

  vital::config_block_sptr algo_config = get_config();

  vital::algo::compute_stereo_depth_map::set_nested_algo_configuration_using_trait(
    computer,
    algo_config,
    d->m_computer );

  if ( ! d->m_computer )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create computer" );
  }

  vital::algo::compute_stereo_depth_map::get_nested_algo_configuration_using_trait(
    computer,
    algo_config,
    d->m_computer );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::compute_stereo_depth_map::check_nested_algo_configuration_using_trait(
         computer, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }
}


// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
_step()
{
  vital::image_container_sptr left_image =
      grab_from_port_using_trait( left_image );
  vital::image_container_sptr right_image =
      grab_from_port_using_trait( right_image );

  vital::image_container_sptr depth_map;

  {
    scoped_step_instrumentation();

    // Get detections from computer on image
    depth_map = d->m_computer->compute( left_image, right_image );
  }

  push_to_port_using_trait( depth_map, depth_map );
}


// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( left_image, required );
  declare_input_port_using_trait( right_image, required );

  // -- output --
  declare_output_port_using_trait( depth_map, optional );
}


// ------------------------------------------------------------------
void
compute_stereo_depth_map_process::
make_config()
{
  declare_config_using_trait( computer );
}


// ================================================================
compute_stereo_depth_map_process::priv
::priv()
{
}


compute_stereo_depth_map_process::priv
::~priv()
{
}

} // end namespace kwiver
