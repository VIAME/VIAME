/*ckwg +29
 * Copyright 2017-2019 by Kitware, Inc.
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

/**
 * \file
 * \brief test reading images and metadata with video_input_splice
 */

#include <test_gtest.h>

#include <arrows/core/video_input_splice.h>
#include <arrows/tests/test_video_input.h>
#include <vital/algo/algorithm_factory.h>
#include <vital/io/metadata_io.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <memory>
#include <string>
#include <fstream>
#include <iostream>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::core;
static std::string list_file_name = "source_list.txt";

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );
  TEST_LOAD_PLUGINS();

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class video_input_splice : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, create)
{
  EXPECT_NE( nullptr, algo::video_input::create("splice") );
}

// ----------------------------------------------------------------------------
static
bool
set_config(kwiver::vital::config_block_sptr config, std::string const& data_dir)
{
  for ( int n = 1; n < 4; ++n )
  {
    std::string block_name = "video_source_" + std::to_string( n ) + ":";

    config->set_value( block_name + "type", "image_list" );
    if ( kwiver::vital::has_algorithm_impl_name( "image_io", "ocv" ) )
    {
      config->set_value( block_name + "image_list:image_reader:type", "ocv" );
    }
    else if ( kwiver::vital::has_algorithm_impl_name( "image_io", "vxl" ) )
    {
      config->set_value( block_name + "image_list:image_reader:type", "vxl" );
    }
    else
    {
      std::cout << "Skipping tests since there is no image reader." << std::endl;
      return false;
    }
  }

  return true;
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, is_good)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;
  vis.open( list_file );

  kwiver::vital::timestamp ts;

  EXPECT_FALSE( vis.good() );

  // open the video
  vis.open( list_file );
  EXPECT_FALSE( vis.good() );

  // step one frame
  vis.next_frame( ts );
  EXPECT_TRUE( vis.good() );

  // close the video
  vis.close();
  EXPECT_FALSE( vis.good() );

  // Reopen the video
  vis.open( list_file );

  int num_frames = 0;
  while ( vis.next_frame( ts ) )
  {
    ++num_frames;
    EXPECT_TRUE( vis.good() )
      << "Video state on frame " << ts.get_frame();
  }
  EXPECT_EQ( num_expected_frames, num_frames );

  EXPECT_FALSE( vis.good() );
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, next_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;
  vis.open( list_file );

  kwiver::vital::timestamp ts;

  int num_frames = 0;
  while ( vis.next_frame( ts ) )
  {
    auto img = vis.frame_image();
    auto md = vis.frame_metadata();

    if (md.size() > 0)
    {
      std::cout << "-----------------------------------\n" << std::endl;
      kwiver::vital::print_metadata( std::cout, *md[0] );
    }

    ++num_frames;
    EXPECT_EQ( num_frames, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }

  EXPECT_FALSE (vis.next_frame(ts) );
  EXPECT_TRUE( vis.end_of_video() );
  EXPECT_EQ( num_expected_frames, num_frames );
  EXPECT_EQ( num_expected_frames, vis.num_frames() );
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, seek_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_seek_frame( vis );

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, seek_then_next_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_seek_then_next( vis );

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, next_then_seek_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_next_then_seek( vis );

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, next_then_seek_then_next)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_next_then_seek_then_next( vis );

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, metadata_map)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  // Get metadata map
  auto md_map = vis.metadata_map()->metadata();

  EXPECT_EQ( md_map.size(), num_expected_frames )
    << "There should be metadata for every frame";

  // Open the frame list file directly and compare name to metadata
  std::ifstream list_file_stream( data_dir + "/frame_list.txt" );
  int frame_number = 1;
  std::string file_name;
  while ( std::getline( list_file_stream, file_name ) )
  {
    auto md_file_name = md_map[frame_number][0]->find(
        kwiver::vital::VITAL_META_IMAGE_URI).as_string();
    EXPECT_TRUE( md_file_name.find( file_name ) != std::string::npos )
      << "File path in metadata should contain " << file_name;
    frame_number++;
  }
  list_file_stream.close();

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, next_frame_nth_frame_output)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "output_nth_frame", nth_frame_output );

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;
  vis.open( list_file );

  test_read_video_nth_frame( vis );

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, seek_frame_nth_frame_output)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "output_nth_frame", nth_frame_output );

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;
  vis.open( list_file );

  test_seek_nth_frame( vis );

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_splice, test_capabilities)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_splice vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;
  vis.open( list_file );

  auto cap = vis.get_implementation_capabilities();
  auto cap_list = cap.capability_list();

  for( auto one : cap_list )
  {
    std::cout << one << " -- "
              << ( cap.capability( one ) ? "true" : "false" )
              << std::endl;
  }
}
