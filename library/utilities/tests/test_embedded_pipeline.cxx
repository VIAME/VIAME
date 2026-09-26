/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include <gtest/gtest.h>
#include "embedded_pipeline.h"
#include <viame/pipeline_framework/adapters/embedded_pipeline.h>
#include <viame/algorithm_framework/plugin/plugin_manager.h>
#include <viame/core_types/image_container.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

class embedded_pipeline_api : public ::testing::Test
{
protected:
  fs::path dir = fs::temp_directory_path() /
    ( "viame_embedded_cpp_" + std::to_string(
      std::chrono::steady_clock::now().time_since_epoch().count() ) );

  void SetUp() override { fs::create_directories( dir ); }
  void TearDown() override { fs::remove_all( dir ); }
  std::string write( std::string const& name, std::string const& text )
  {
    auto path = dir / name;
    fs::create_directories( path.parent_path() );
    std::ofstream stream( path );
    stream << text;
    if( !stream ) { throw std::runtime_error( "Cannot write test pipeline" ); }
    return path.string();
  }
  std::string graph( int cameras = 1 )
  {
    std::ostringstream text;
    text << "process result\n :: image_writer\n :file_name never-written.png\n";
    for( int camera = 0; camera < cameras; ++camera )
    {
      text << "process input" << camera << "\n :: video_input\n"
           << " :video_filename /missing/images.txt\n"
           << "connect from input" << camera << ".image to result.image" << camera << "\n";
    }
    return text.str();
  }
};

TEST_F( embedded_pipeline_api, native_round_trip_one_two_three_cameras )
{
  // Only native process/scheduler plugins: no Python registration module.
  using manager = viame::plugin_manager;
  manager::instance().load_all_plugins( manager::plugin_type::PROCESSES );
  for( int cameras : { 1, 2, 3 } )
  {
    auto const path = write( "normal.pipe", graph( cameras ) );
    auto const description = viame::prepare_embedded_pipeline( path );
    ASSERT_EQ( description.input_names.size(), cameras );
    ASSERT_EQ( description.input_ports.size(), cameras );
    ASSERT_EQ( description.output_ports.size(), cameras );
    EXPECT_EQ( description.pipeline_text.find( "video_input" ), std::string::npos );
    EXPECT_EQ( description.pipeline_text.find( "image_writer" ), std::string::npos );
    viame::embedded_pipeline pipeline;
    description.build( pipeline ); // No generated pipeline file needed.
    pipeline.start();
    for( int frame = 0; frame < 3; ++frame )
    {
      auto data = viame::adapter::adapter_data_set::create();
      std::vector< viame::image_container_sptr > images;
      for( int camera = 0; camera < cameras; ++camera )
      {
        auto image = std::make_shared< viame::simple_image_container >(
          viame::image( 8, 6, 3 ) );
        images.push_back( image );
        // The datum's static type must be the base image_container pointer.
        data->add_value< viame::image_container_sptr >(
          description.input_ports.at( "input" + std::to_string( camera ) + ".image" ), image );
      }
      pipeline.send( data );
      auto output = pipeline.receive();
      ASSERT_FALSE( output->is_end_of_data() );
      for( int camera = 0; camera < cameras; ++camera )
      {
        auto const result = output->value< viame::image_container_sptr >(
          description.output_ports.at( "result.image" + std::to_string( camera ) ) );
        EXPECT_EQ( result, images[camera] ); // Native containers pass through without copying.
      }
    }
    pipeline.send_end_of_input();
    EXPECT_TRUE( pipeline.receive()->is_end_of_data() );
    pipeline.wait();
    EXPECT_FALSE( fs::exists( dir / "never-written.png" ) );
  }
}

TEST_F( embedded_pipeline_api, includes_and_relative_model_paths )
{
  write( "includes/readers.pipe", "process input\n :: video_input\n" );
  write( "sub/detector.pipe", "process detector\n :: image_filter\n"
         " relativepath filter:model = weights.bin\n" );
  auto path = write( "main.pipe", "include readers.pipe\ninclude sub/detector.pipe\n"
    "process writer\n :: image_writer\n"
    "connect from input.image to detector.image\n"
    "connect from detector.image to writer.image\n" );
  viame::embedded_pipeline_options options;
  options.search_paths = { ( dir / "includes" ).string() };
  auto description = viame::prepare_embedded_pipeline( path, options );
  EXPECT_NE( description.pipeline_text.find( ( dir / "sub/weights.bin" ).string() ), std::string::npos );
  EXPECT_NE( description.pipeline_text.find( ":: image_filter" ), std::string::npos );
  EXPECT_EQ( description.source_directory, dir.string() );
}

TEST_F( embedded_pipeline_api, explicit_selection_and_name_collisions )
{
  auto path = write( "custom.pipe", "process viame_memory_input\n :: custom_source\n"
    "process viame_memory_output\n :: custom_sink\n"
    "connect from viame_memory_input.image to viame_memory_output.image\n" );
  viame::embedded_pipeline_options options;
  options.inputs = { { "viame_memory_input" } };
  options.outputs = { { "viame_memory_output" } };
  auto description = viame::prepare_embedded_pipeline( path, options );
  EXPECT_NE( description.pipeline_text.find( "process viame_memory_input_\n" ), std::string::npos );
  EXPECT_NE( description.pipeline_text.find( "process viame_memory_output_\n" ), std::string::npos );
  EXPECT_EQ( description.input_ports.count( "viame_memory_input.image" ), 1 );
  EXPECT_EQ( description.output_ports.count( "viame_memory_output.image" ), 1 );
}

TEST_F( embedded_pipeline_api, source_fanout_uses_one_adapter_port )
{
  auto path = write( "fanout.pipe", graph() +
    "connect from input0.image to result.image_again\n" );
  auto description = viame::prepare_embedded_pipeline( path );
  EXPECT_EQ( description.input_ports.size(), 1 );
  EXPECT_EQ( description.output_ports.size(), 2 );
}

TEST_F( embedded_pipeline_api, invalid_selections )
{
  auto const path = write( "normal.pipe", graph() );
  viame::embedded_pipeline_options options;
  options.inputs = std::vector< std::string >{};
  EXPECT_THROW( viame::prepare_embedded_pipeline( path, options ), std::invalid_argument );
  options.inputs = { { "missing" } };
  EXPECT_THROW( viame::prepare_embedded_pipeline( path, options ), std::invalid_argument );
  options.inputs = { { "input0", "input0" } };
  EXPECT_THROW( viame::prepare_embedded_pipeline( path, options ), std::invalid_argument );
  options.inputs = { { "input0" } };
  options.outputs = options.inputs;
  EXPECT_THROW( viame::prepare_embedded_pipeline( path, options ), std::invalid_argument );
}

TEST_F( embedded_pipeline_api, rejects_invalid_topology_and_duplicate_processes )
{
  auto path = write( "incoming.pipe", graph() +
    "process extra\n :: input_adapter\nconnect from extra.image to input0.image\n" );
  EXPECT_THROW( viame::prepare_embedded_pipeline( path ), std::invalid_argument );
  path = write( "outgoing.pipe", graph() +
    "process extra\n :: image_filter\nconnect from result.image to extra.image\n" );
  EXPECT_THROW( viame::prepare_embedded_pipeline( path ), std::invalid_argument );
  path = write( "duplicate.pipe", graph() + "process input0\n :: video_input\n" );
  EXPECT_THROW( viame::prepare_embedded_pipeline( path ), std::invalid_argument );
  path = write( "disconnected.pipe", "process input\n :: video_input\nprocess writer\n :: image_writer\n" );
  EXPECT_THROW( viame::prepare_embedded_pipeline( path ), std::invalid_argument );
}
