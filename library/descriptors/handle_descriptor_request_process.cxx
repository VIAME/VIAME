// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "handle_descriptor_request_process.h"

#include <viame/core_types/vital_types.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/timestamp_config.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/matrix.h>

#include <viame/algorithm_framework/algo/handle_descriptor_request.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>

#include <filesystem>

namespace viame {

namespace algo = viame::algo;

create_port_trait( filename, file_name, "KWA input filename" );
create_port_trait( stream_id, string, "Stream ID to place in file" );

create_algorithm_name_config_trait( handler );

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
::handle_descriptor_request_process( viame::config_block_sptr const& config )
  : process( config ),
    d( new handle_descriptor_request_process::priv )
{
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
  viame::config_block_sptr algo_config = get_config();

  set_nested_algo_configuration_using_trait(
    handler,
    algo_config,
    d->m_handler );

  if( !d->m_handler )
  {
    VITAL_THROW( viame::pipeline::invalid_configuration_exception,
                 name(), "Unable to create handle_descriptor_request" );
  }

  get_nested_algo_configuration_using_trait(
    handler,
    algo_config,
    d->m_handler );

  // Check config so it will give run-time diagnostic of config problems
  if( !check_nested_algo_configuration_using_trait(
    handler, algo_config, d->m_handler ) )
  {
    VITAL_THROW( viame::pipeline::invalid_configuration_exception,
                 name(), "Configuration check failed." );
  }
}

// -----------------------------------------------------------------------------
void
handle_descriptor_request_process
::_step()
{
  // Retrieve inputs from ports
  viame::descriptor_request_sptr request;

  request = grab_from_port_using_trait( descriptor_request );

  // Special case, output empty results
  if( !request )
  {
    push_to_port_using_trait( track_descriptor_set, viame::track_descriptor_set_sptr() );

    push_to_port_using_trait( image, viame::image_container_sptr() );
    push_to_port_using_trait( timestamp, viame::timestamp() );
    push_to_port_using_trait( filename, "" );
    push_to_port_using_trait( stream_id, "" );
    return;
  }

  // Get output matrix and detections
  viame::track_descriptor_set_sptr descriptors;
  std::vector< viame::image_container_sptr > images;

  viame::string_t filename;
  viame::string_t stream_id;

  if( request && !d->m_handler->handle( request, descriptors, images ) )
  {
    LOG_ERROR( logger(), "Could not handle descriptor request" );
  }

  // Return all outputs
  push_to_port_using_trait( track_descriptor_set, descriptors );

  if( request )
  {
    std::filesystem::path p( request->data_location() );
    filename = p.stem().string();
    stream_id = filename;
  }

  // Step image output pipeline if connected
  for( auto image : images )
  {
    viame::timestamp ts;

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
  viame::pipeline::process::port_flags_t optional;

  viame::pipeline::process::port_flags_t required;
  required.insert( flag_required );

  viame::pipeline::process::port_flags_t shared;
  shared.insert( flag_output_shared );

  // -- input --
  declare_input_port_using_trait( descriptor_request, required );

  // -- output --
  declare_output_port_using_trait( track_descriptor_set, optional );

  declare_output_port_using_trait( image, shared );
  declare_output_port_using_trait( timestamp, optional );
  declare_output_port_using_trait( filename, optional );
  declare_output_port_using_trait( stream_id, optional );
}

// -----------------------------------------------------------------------------
void handle_descriptor_request_process
::make_config()
{
  declare_config_using_trait( handler );
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

} // namespace viame
