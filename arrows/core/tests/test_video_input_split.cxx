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

/**
 * \file
 * \brief test reading images and metadata with video_input_split
 */

#include <test_gtest.h>

#include "dummy_image_io.h"

#include <arrows/core/video_input_split.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <memory>
#include <string>
#include <iostream>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::core;

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
kwiver::vital::config_block_sptr
make_config(std::string const& data_dir)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();
  config->set_value( "image_source:type", "image_list" );
  config->set_value( "image_source:image_list:image_reader:type", "dummy" );

  config->set_value( "metadata_source:type", "pos" );
  config->set_value( "metadata_source:pos:metadata_directory", data_dir + "/pos");

  return config;
}

// ----------------------------------------------------------------------------
TEST_F(video_input_split, read_list)
{
  // register the dummy_image_io so we can use it in this test
  register_dummy_image_io();

  // make config block
  auto config = make_config(data_dir);

  kwiver::arrows::core::video_input_split vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/frame_list.txt";
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
  }
  EXPECT_EQ( 5, num_frames );
}

// ----------------------------------------------------------------------------
TEST_F(video_input_split, test_capabilities)
{
  // register the dummy_image_io so we can use it in this test
  register_dummy_image_io();

  // make config block
  auto config = make_config(data_dir);

  kwiver::arrows::core::video_input_split vis;

  EXPECT_TRUE( vis.check_configuration( config ) );
  vis.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/frame_list.txt";
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
