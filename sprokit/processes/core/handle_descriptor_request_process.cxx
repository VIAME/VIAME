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
#include <vital/algo/handle_descriptor_request.h>
#include <vital/types/image_container_set_simple.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>
#include <sprokit/processes/adapters/embedded_pipeline.h>

#include <boost/filesystem/path.hpp>

#include <memory>
#include <fstream>
#include <chrono>

namespace kwiver
{

namespace algo = vital::algo;

create_config_trait( image_pipeline_file, std::string, "",
  "Filename for the image processing pipeline. This pipeline should take, "
  "as input, a filename and produce descriptors as output." );

create_config_trait( assign_uids, bool, "true",
  "Whether or not this process should assign unique UIDs to each output "
  "descriptor produced by this process" );

//------------------------------------------------------------------------------
// Private implementation class
class handle_descriptor_request_process::priv
{
public:
  priv();
  ~priv();

  std::string image_pipeline_file;
  bool assign_uids;

  std::unique_ptr< embedded_pipeline > image_pipeline;

  std::string generate_uid();
}; // end priv class


// =============================================================================

handle_descriptor_request_process
::handle_descriptor_request_process( vital::config_block_sptr const& config )
  : process( config ),
    d( new handle_descriptor_request_process::priv )
{
  make_ports();
  make_config();
}


handle_descriptor_request_process
::~handle_descriptor_request_process()
{
  if( d->image_pipeline )
  {
    d->image_pipeline->send_end_of_input();
    d->image_pipeline->receive();
    d->image_pipeline->wait();
    d->image_pipeline.reset();
  }
}


// -----------------------------------------------------------------------------
void
handle_descriptor_request_process
::_configure()
{
  vital::config_block_sptr algo_config = get_config();

  d->image_pipeline_file = config_value_using_trait( image_pipeline_file );
  d->assign_uids = config_value_using_trait( assign_uids );
}


// -----------------------------------------------------------------------------
void
handle_descriptor_request_process
::_init()
{
  auto dir = boost::filesystem::path( d->image_pipeline_file ).parent_path();

  if( !d->image_pipeline_file.empty() )
  {
    std::unique_ptr< embedded_pipeline > new_pipeline =
      std::unique_ptr< embedded_pipeline >( new embedded_pipeline() );

    std::ifstream pipe_stream;
    pipe_stream.open( d->image_pipeline_file, std::ifstream::in );

    if( !pipe_stream )
    {
      throw sprokit::invalid_configuration_exception(
        name(), "Unable to open pipeline file: " + d->image_pipeline_file );
    }

    try
    {
      new_pipeline->build_pipeline( pipe_stream, dir.string() );
      new_pipeline->start();
    }
    catch( const std::exception& e )
    {
      throw sprokit::invalid_configuration_exception( name(), e.what() );
    }

    d->image_pipeline = std::move( new_pipeline );
    pipe_stream.close();
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

  // Special case, output empty results and pass thru if not specified
  if( !request )
  {
    push_to_port_using_trait( track_descriptor_set, vital::track_descriptor_set_sptr() );
    push_to_port_using_trait( image_set, vital::image_container_set_sptr() );
    return; // Normal return, no failure
  }

  // Get output descriptors from internal pipeline
  vital::track_descriptor_set_sptr descriptors;
  std::vector< vital::image_container_sptr > images;

  // Get filepaths
  boost::filesystem::path p( request->data_location() );

  vital::string_t filename = p.string();
  vital::string_t stream_id = p.stem().string();

  if( d->image_pipeline )
  {
    // Set request on pipeline inputs
    auto ids = adapter::adapter_data_set::create();

    ids->add_value( "filename", filename );
    ids->add_value( "stream_id", stream_id );

    // Send the request through the pipeline and wait for a result
    d->image_pipeline->send( ids );

    auto const& ods = d->image_pipeline->receive();

    if( ods->is_end_of_data() )
    {
      throw std::runtime_error( "Pipeline terminated unexpectingly" );
    }

    // Grab result from pipeline output data set
    auto const& iter = ods->find( "track_descriptor_set" );

    if( iter == ods->end() )
    {
      throw std::runtime_error( "Empty pipeline output" );
    }

    descriptors = iter->second->get_datum< vital::track_descriptor_set_sptr >();

    auto const& iter2 = ods->find( "image" );

    if( iter2 == ods->end() )
    {
      throw std::runtime_error( "Empty pipeline output" );
    }

    images.push_back( iter2->second->get_datum< vital::image_container_sptr >() );

    // Assign optional UID to descriptors
    if( d->assign_uids )
    {
      for( auto track_desc : *descriptors )
      {
        track_desc->set_uid( d->generate_uid() );
      }
    }
  }

  vital::image_container_set_sptr image_set(
    new vital::simple_image_container_set( images ) );

  // Return all outputs
  push_to_port_using_trait( track_descriptor_set, descriptors );
  push_to_port_using_trait( image_set, image_set );
}


// -----------------------------------------------------------------------------
void handle_descriptor_request_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;

  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  sprokit::process::port_flags_t shared;
  shared.insert( flag_output_shared );

  // -- input --
  declare_input_port_using_trait( descriptor_request, required );

  // -- output --
  declare_output_port_using_trait( track_descriptor_set, optional );
  declare_output_port_using_trait( image_set, optional );
}


// -----------------------------------------------------------------------------
void handle_descriptor_request_process
::make_config()
{
  declare_config_using_trait( image_pipeline_file );
  declare_config_using_trait( assign_uids );
}


// =============================================================================
handle_descriptor_request_process::priv
::priv()
  : image_pipeline_file("")
  , assign_uids( true )
  , image_pipeline()
{
}


handle_descriptor_request_process::priv
::~priv()
{
}


std::string
handle_descriptor_request_process::priv
::generate_uid()
{
  static unsigned query_id = 0;

  auto current_time =
    std::chrono::system_clock::to_time_t( std::chrono::system_clock::now() );

  std::string uid =
    "query_" + std::to_string( query_id ) + "_" +
    "time_" + std::to_string( current_time );

  query_id++;

  return uid;
}

} // end namespace
