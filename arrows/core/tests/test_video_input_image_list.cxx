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
 * \brief test reading video from a list of images.
 */

#include <test_common.h>

#include "dummy_image_io.h"

#include <arrows/core/video_input_image_list.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <memory>
#include <string>
#include <iostream>

#define TEST_ARGS ( kwiver::vital::path_t data_dir )

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(2);

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  testname_t const testname = argv[1];
  kwiver::vital::path_t data_dir( argv[2] );

  RUN_TEST(testname, data_dir);
}

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::core;


// ------------------------------------------------------------------
IMPLEMENT_TEST(create)
{
  algo::video_input_sptr vi = algo::video_input::create("image_list");
  if (!vi)
  {
    TEST_ERROR("Unable to create core::video_input_image_list by name");
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(read_list)
{
  // register the dummy_image_io so we can use it in this test
  register_dummy_image_io();

  // make config block
  auto config = kwiver::vital::config_block::empty_config();
  config->set_value( "image_reader:type", "dummy" );

  kwiver::arrows::core::video_input_image_list viil;

  viil.check_configuration( config );
  viil.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/frame_list.txt";
  viil.open( list_file );

  kwiver::vital::timestamp ts;

  int num_frames = 0;
  while ( viil.next_frame( ts ) )
  {
    auto img = viil.frame_image();
    auto md = viil.frame_metadata();

    if (md.size() > 0)
    {
      std::cout << "-----------------------------------\n" << std::endl;
      kwiver::vital::print_metadata( std::cout, *md[0] );
    }

    ++num_frames;
    TEST_EQUAL( "Sequential frame numbers", ts.get_frame(), num_frames );
  }
  TEST_EQUAL( "Number of frames read", num_frames, 5 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(is_good)
{
  // register the dummy_image_io so we can use it in this test
  register_dummy_image_io();

  // make config block
  auto config = kwiver::vital::config_block::empty_config();
  config->set_value( "image_reader:type", "dummy" );

  kwiver::arrows::core::video_input_image_list viil;

  viil.check_configuration( config );
  viil.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/frame_list.txt";
  kwiver::vital::timestamp ts;

  TEST_EQUAL( "Video state is not \"good\" before open", viil.good(), false );

  // open the video
  viil.open( list_file );
  TEST_EQUAL( "Video state is not \"good\" after open but before first frame",
              viil.good(), false );

  // step one frame
  viil.next_frame( ts );
  TEST_EQUAL( "Video state is \"good\" on first frame", viil.good(), true );

  // close the video
  viil.close();
  TEST_EQUAL( "Video state is not \"good\" after close", viil.good(), false );

  // Reopen the video
  viil.open( list_file );

  int num_frames = 0;
  while ( viil.next_frame( ts ) )
  {
    ++num_frames;
    TEST_EQUAL( "Video state is \"good\" on frame" << ts.get_frame(),
                viil.good(), true );
  }
  TEST_EQUAL( "Number of frames read", num_frames, 5 );
}
