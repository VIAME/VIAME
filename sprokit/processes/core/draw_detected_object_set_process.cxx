/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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


#include "draw_detected_object_set_process.h"

#include <vital/algo/draw_detected_object_set.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

// (config-key, value-type, default-value, description )
  create_config_trait( draw_algo, std::string, "", "Name of drawing algorithm config block." );


//----------------------------------------------------------------
// Private implementation class
  class  draw_detected_object_set_process::priv
{
public:
  priv();
  ~priv();

  vital::algo::draw_detected_object_set_sptr m_algo;

}; // end priv class


// ================================================================

draw_detected_object_set_process
::draw_detected_object_set_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new draw_detected_object_set_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach

  make_ports();
  make_config();
}


draw_detected_object_set_process
::~draw_detected_object_set_process()
{
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::_configure()
{
  start_configure_processing();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::draw_detected_object_set::check_nested_algo_configuration( "draw_algo", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  vital::algo::draw_detected_object_set::set_nested_algo_configuration( "draw_algo", algo_config, d->m_algo );
  if ( ! d->m_algo )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create algorithm." );
  }

  stop_configure_processing();
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::_step()
{
  auto input_image = grab_from_port_using_trait( image );
  auto obj_set = grab_from_port_using_trait( detected_object_set );

  start_step_processing();

  auto out_image = d->m_algo->draw( obj_set, input_image );

  stop_step_processing();

  push_to_port_using_trait( image, out_image );
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( image, optional );
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::make_config()
{
  declare_config_using_trait( draw_algo );
}


// ================================================================
draw_detected_object_set_process::priv
::priv()
{
}


draw_detected_object_set_process::priv
::~priv()
{
}

} // end namespace
