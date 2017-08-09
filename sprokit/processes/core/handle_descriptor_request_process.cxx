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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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

#include "handle_descriptor_request_process.h"

#include <vital/vital_types.h>
#include <vital/vital_foreach.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/object_track_set.h>
#include <vital/types/matrix.h>

#include <vital/algo/handle_descriptor_request.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

#include <boost/filesystem/path.hpp>

namespace kwiver
{

namespace algo = vital::algo;

create_port_trait( filename, file_name,
  "KWA input filename" );
create_port_trait( stream_id, string,
  "Stream ID to place in file" );

//------------------------------------------------------------------------------
// Private implementation class
class handle_descriptor_request_process::priv
{
public:
  priv();
  ~priv();

  unsigned track_read_delay;

  algo::handle_descriptor_request_sptr m_handler;
}; // end priv class


// =============================================================================

handle_descriptor_request_process
::handle_descriptor_request_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new handle_descriptor_request_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( vital::get_logger( name() ) );

  make_ports();
  make_config();
}


handle_descriptor_request_process
::~handle_descriptor_request_process()
{
}


// -----------------------------------------------------------------------------
void handle_descriptor_request_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  algo::handle_descriptor_request::set_nested_algo_configuration(
    "handler", algo_config, d->m_handler );

  if( !d->m_handler )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create handle_descriptor_request" );
  }

  algo::handle_descriptor_request::get_nested_algo_configuration(
    "handler", algo_config, d->m_handler );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::handle_descriptor_request::check_nested_algo_configuration(
    "handler", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Configuration check failed." );
  }
}


// -----------------------------------------------------------------------------
void
handle_descriptor_request_process
::_step()
{
  // Retrieve inputs from ports
  vital::descriptor_request_sptr request;

  request = grab_from_port_using_trait( descriptor_request );

  // Special case, output empty results
  if( !request )
  {
    push_to_port_using_trait( track_descriptor_set, vital::track_descriptor_set_sptr() );

    push_to_port_using_trait( image, vital::image_container_sptr() );
    push_to_port_using_trait( timestamp, vital::timestamp() );
    push_to_port_using_trait( filename, "" );
    push_to_port_using_trait( stream_id, "" );
    return;
  }

  // Get output matrix and detections
  vital::track_descriptor_set_sptr descriptors;
  std::vector< vital::image_container_sptr > images;

  vital::string_t filename;
  vital::string_t stream_id;

  if( request && !d->m_handler->handle( request, descriptors, images ) )
  {
    //LOG_ERROR( name(), "Could not handle descriptor request" );
  }

  // Return all outputs
  push_to_port_using_trait( track_descriptor_set, descriptors );

  if( request )
  {
    boost::filesystem::path p( request->data_location() );
    filename = p.stem().string();
    stream_id = filename;
  }

  // Step image output pipeline if connected
  VITAL_FOREACH( auto image, images )
  {
    vital::timestamp ts;

    push_to_port_using_trait( image, image );
    push_to_port_using_trait( timestamp, ts );
    push_to_port_using_trait( filename, filename );
    push_to_port_using_trait( stream_id, stream_id );
  }
}


// -----------------------------------------------------------------------------
void handle_descriptor_request_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( descriptor_request, required );

  // -- output --
  declare_output_port_using_trait( track_descriptor_set, optional );

  declare_output_port_using_trait( image, optional );
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( filename, optional );
  declare_output_port_using_trait( stream_id, optional );
}


// -----------------------------------------------------------------------------
void handle_descriptor_request_process
::make_config()
{
}


// =============================================================================
handle_descriptor_request_process::priv
::priv()
{
}


handle_descriptor_request_process::priv
::~priv()
{
}

} // end namespace
