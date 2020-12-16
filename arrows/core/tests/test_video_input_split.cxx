// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test reading images and metadata with video_input_split
 */

#include <test_gtest.h>

#include <arrows/core/video_input_split.h>
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
static std::string list_file_name = "frame_list.txt";

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
class video_input_split : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(video_input_split, create)
{
  EXPECT_NE( nullptr, algo::video_input::create("split") );
}

// ----------------------------------------------------------------------------
static
bool
set_config(kwiver::vital::config_block_sptr config, std::string const& data_dir)
{
  config->set_value( "image_source:type", "image_list" );
  if ( kwiver::vital::has_algorithm_impl_name( "image_io", "ocv" ) )
  {
    config->set_value( "image_source:image_list:image_reader:type", "ocv" );
  }
  else if ( kwiver::vital::has_algorithm_impl_name( "image_io", "vxl" ) )
  {
    config->set_value( "image_source:image_list:image_reader:type", "vxl" );
  }
  else
  {
    std::cout << "Skipping tests since there is no image reader." << std::endl;
    return false;
  }

  config->set_value( "metadata_source:type", "pos" );
  config->set_value( "metadata_source:pos:metadata_directory", data_dir + "/pos");

  return true;
}

// ----------------------------------------------------------------------------
TEST_F(video_input_split, read_list)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_split vis;

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
    EXPECT_EQ( ts.get_time_usec(), vis.frame_timestamp().get_time_usec() );
    EXPECT_EQ( ts.get_frame(), vis.frame_timestamp().get_frame() );
  }
  EXPECT_EQ( num_expected_frames, num_frames );
  EXPECT_EQ( num_expected_frames, vis.num_frames() );
}

TEST_F(video_input_split, seek_frame)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_split vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_seek_frame( vis );

  vis.close();
}

TEST_F(video_input_split, seek_then_next_frame)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_split vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_seek_then_next( vis );

  vis.close();
}

TEST_F(video_input_split, next_then_seek_frame)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_split vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  test_next_then_seek( vis );

  vis.close();
}

TEST_F(video_input_split, metadata_map)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_split vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + list_file_name;

  // Open the video
  vis.open( list_file );

  // Get metadata map
  auto md_map = vis.metadata_map()->metadata();

  EXPECT_EQ( md_map.size(), num_expected_frames )
    << "There should be metadata for every frame";

  // Open the list file directly and construct metadata file names and compare
  std::ifstream list_file_stream( list_file );
  int frame_number = 1;
  std::string file_name;
  while ( std::getline( list_file_stream, file_name ) )
  {
    file_name.replace(0, 6, "pos");
    file_name.replace(file_name.length() - 3, 3, "pos");

    auto md_test = kwiver::vital::read_pos_file( data_dir + "/" + file_name );
    auto md_vec = md_map[frame_number];

    // Loop over metadata items and compare
    for (auto iter = md_test->begin(); iter != md_test->end(); ++iter)
    {
      bool found_item = false;
      for (auto md : md_vec)
      {
        found_item = found_item || md->has( iter->first );
      }
      EXPECT_TRUE( found_item )
        << "Metadata should have item " << iter->second->name();
    }

    frame_number++;
  }
  list_file_stream.close();

  vis.close();
}

// ----------------------------------------------------------------------------
TEST_F(video_input_split, test_capabilities)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  if( !set_config(config, data_dir) )
  {
    return;
  }

  kwiver::arrows::core::video_input_split vis;

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
