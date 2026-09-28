// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "handle_descriptor_request_process.h"

#include <viame/core_types/viame_core_types.h>
#include <viame/algorithm_framework/algo/handle_descriptor_request.h>
#include <viame/core_types/image_container_set_simple.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/detected_object.h>
#include <viame/core_types/detected_object_type.h>
#include <viame/algorithm_framework/logger/logger.h>

#include <viame/pipeline_framework/type_traits.h>

#include <viame/pipeline_framework/process_exception.h>
#include <viame/pipeline_framework/adapters/embedded_pipeline.h>

#include <algorithm>
#include <filesystem>
#include <memory>
#include <fstream>
#include <chrono>

namespace viame
{

namespace algo = viame::algo;

create_config_trait( image_pipeline_file, std::string, "",
  "Filename for the image processing pipeline. This pipeline should take, "
  "as input, a filename and produce descriptors as output." );

create_config_trait( assign_uids, bool, "true",
  "Whether or not this process should assign unique UIDs to each output "
  "descriptor produced by this process" );

create_port_trait( boxes_provided, bool,
  "Flag indicating if bounding boxes were provided in the descriptor request" );

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
  viame::config_block_sptr algo_config = get_config();

  d->image_pipeline_file = config_value_using_trait( image_pipeline_file );
  d->assign_uids = config_value_using_trait( assign_uids );
}


// -----------------------------------------------------------------------------
void
handle_descriptor_request_process
::_init()
{
  auto dir = std::filesystem::path( d->image_pipeline_file ).parent_path();

  if( !d->image_pipeline_file.empty() )
  {
    std::unique_ptr< embedded_pipeline > new_pipeline =
      std::unique_ptr< embedded_pipeline >( new embedded_pipeline() );

    std::ifstream pipe_stream;
    pipe_stream.open( d->image_pipeline_file, std::ifstream::in );

    if( !pipe_stream )
    {
      throw viame::pipeline::invalid_configuration_exception(
        name(), "Unable to open pipeline file: " + d->image_pipeline_file );
    }

    try
    {
      new_pipeline->build_pipeline( pipe_stream, dir.string() );
      new_pipeline->start();
    }
    catch( const std::exception& e )
    {
      throw viame::pipeline::invalid_configuration_exception( name(), e.what() );
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
  viame::descriptor_request_sptr request;

  request = grab_from_port_using_trait( descriptor_request );

  // Special case, output empty results and pass thru if not specified
  if( !request )
  {
    push_to_port_using_trait( track_descriptor_set, viame::track_descriptor_set_sptr() );
    push_to_port_using_trait( image_set, viame::image_container_set_sptr() );
    push_to_port_using_trait( boxes_provided, false );
    return; // Normal return, no failure
  }

  // Get output descriptors from internal pipeline
  viame::track_descriptor_set_sptr descriptors;
  std::vector< viame::image_container_sptr > images;

  // Get filepaths
  std::filesystem::path p( request->data_location() );

  viame::string_t filename = p.string();
  viame::string_t stream_id = p.stem().string();

  if( d->image_pipeline )
  {
    // Set request on pipeline inputs. Only populate ports the pipeline
    // actually exposes; the input adapter rejects data packets containing
    // entries for unconnected ports (e.g. pipelines with no stream_id
    // consumer).
    auto ids = adapter::adapter_data_set::create();

    auto const& input_ports = d->image_pipeline->input_port_names();
    auto const has_port = [&input_ports]( std::string const& name )
    {
      return std::find( input_ports.begin(), input_ports.end(), name )
             != input_ports.end();
    };

    if( has_port( "filename" ) )
    {
      ids->add_value( "filename", filename );
    }
    if( has_port( "stream_id" ) )
    {
      ids->add_value( "stream_id", stream_id );
    }

    // Extract spatial regions (bounding boxes) from the request and convert
    // to a detected_object_set for descriptor computation. Always send this
    // value even if empty, so the pipeline can merge with detector output.
    auto const& spatial_regions = request->spatial_regions();
    auto dos = std::make_shared< viame::detected_object_set >();
    bool boxes_provided = !spatial_regions.empty();

    for( auto const& box : spatial_regions )
    {
      viame::bounding_box_d bbox(
        static_cast< double >( box.min_x() ),
        static_cast< double >( box.min_y() ),
        static_cast< double >( box.max_x() ),
        static_cast< double >( box.max_y() ) );

      auto det = std::make_shared< viame::detected_object >( bbox );

      // Set a type on the detection so it passes through class filters
      // that require a detected_object_type to be present
      auto dot = std::make_shared< viame::detected_object_type >();
      dot->set_score( "query_region", 1.0 );
      det->set_type( dot );

      dos->add( det );
    }

    if( has_port( "detected_object_set" ) )
    {
      ids->add_value( "detected_object_set", dos );
    }

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

    descriptors = iter->second->get_datum< viame::track_descriptor_set_sptr >();

    auto const& iter2 = ods->find( "image" );

    if( iter2 == ods->end() )
    {
      throw std::runtime_error( "Empty pipeline output" );
    }

    images.push_back( iter2->second->get_datum< viame::image_container_sptr >() );

    // Assign optional UID to descriptors
    if( d->assign_uids )
    {
      for( auto track_desc : *descriptors )
      {
        track_desc->set_uid( d->generate_uid() );
      }
    }
  }

  viame::image_container_set_sptr image_set(
    new viame::simple_image_container_set( images ) );

  // Track if boxes were provided (set in the pipeline section above)
  bool boxes_provided_flag = false;
  if( d->image_pipeline )
  {
    auto const& spatial_regions = request->spatial_regions();
    boxes_provided_flag = !spatial_regions.empty();
  }

  // Return all outputs
  push_to_port_using_trait( track_descriptor_set, descriptors );
  push_to_port_using_trait( image_set, image_set );
  push_to_port_using_trait( boxes_provided, boxes_provided_flag );
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
  declare_output_port_using_trait( image_set, optional );
  declare_output_port_using_trait( boxes_provided, optional );
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
