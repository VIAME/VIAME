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

#include "algo_wrapper_process.h"

#include <vital/algo/refine_detections.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

/*
 * Note that the //+ comments are intended for the person who is
 * adapting this template and should be removed from the final
 * product.
 */

namespace kwiver {

//+ Define a config entry for the main algorithm configuration
//+ subblock. A name other than "algorithm" can be used if it is more
//+ descriptive. If so, change name throughout the rest of this file also.
create_config_trait( algorithm, std::string, "", "Algorithm configuration subblock" );

//----------------------------------------------------------------
// Private implementation class
class algo_wrapper_process::priv
{
public:
  priv();
  ~priv();

  //+ define sptr to specific algorithm type
  vital::algo::algo_type_sptr m_algo;

}; // end priv class


// ==================================================================
algo_wrapper_process::
algo_wrapper_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new algo_wrapper_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


algo_wrapper_process::
~algo_wrapper_process()
{
}


// ------------------------------------------------------------------
void
algo_wrapper_process::
_configure()
{
  start_configure_processing();

  vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems.
  // Call check_nested_algo_configuration() first so that it will display a list of
  // concrete instances of the desired algorithms that are available if the config
  // does not select a valid one.
  if ( ! vital::algo::refine_detections::check_nested_algo_configuration( "algorithm", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  vital::algo::refine_detections::set_nested_algo_configuration( "algorithm", algo_config, d->m_algo );

  if ( ! d->m_algo )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create refiner" );
  }

  stop_configure_processing();
}


// ------------------------------------------------------------------
void
algo_wrapper_process::
_step()
{
  //+ get inputs as needed
  vital::image_container_sptr image = grab_from_port_using_trait( image );
  vital::detected_object_set_sptr dets = grab_from_port_using_trait( detected_object_set );

  start_step_processing();

  // Send inputs to algorithm

  //+ This call must correspond to the wrapped algorithm.
  vital::detected_object_set_sptr results = d->m_algo->process( image, dets );

  stop_step_processing();

  //+ push output as needed
  push_to_port_using_trait( detected_object_set, results );
}


// ------------------------------------------------------------------
void
algo_wrapper_process::
make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, optional );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
}


// ------------------------------------------------------------------
void
algo_wrapper_process::
make_config()
{
  declare_config_using_trait( refiner );
}


// ================================================================
algo_wrapper_process::priv
::priv()
{
}


algo_wrapper_process::priv
::~priv()
{
}

} // end namespace kwiver
