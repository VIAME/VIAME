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

#include <arrows/core/video_input_image_list.h>
#include <vital/vital_foreach.h>
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
IMPLEMENT_TEST(read_list)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();
  config->set_value( "image_reader:type", "vxl" );

  kwiver::arrows::core::video_input_image_list viil;

  viil.check_configuration( config );
  viil.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/image_list.txt";
  viil.open( list_file );

  kwiver::vital::timestamp ts;
  int expected_frame(1);

  while ( viil.next_frame( ts ) )
  {
    TEST_EQUAL( "Returned expected frame", ts.get_frame(), expected_frame );
    ++expected_frame;
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(test_capabilities)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();
  config->set_value( "image_reader:type", "vxl" );

  kwiver::arrows::core::video_input_image_list viil;

  viil.check_configuration( config );
  viil.set_configuration( config );

  auto cap = viil.get_implementation_capabilities();
  auto cap_list = cap.capability_list();

  VITAL_FOREACH( auto one, cap_list )
  {
    std::cout << one << " -- "
              << ( cap.capability( one ) ? "true" : "false" )
              << std::endl;
  }
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(read_list_subset)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();
  // Select the actual image reader
  config->set_value( "image_reader:type", "vxl" );

  // Specify frame range to video_input_image_list
  config->set_value( "start_at_frame", "2" );
  config->set_value( "stop_after_frame", "2" );

  kwiver::arrows::core::video_input_image_list viil;

  viil.check_configuration( config );
  viil.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/image_list.txt";
  viil.open( list_file );

  kwiver::vital::timestamp ts;
  int expected_frame(2);

  while ( viil.next_frame( ts ) )
  {
    TEST_EQUAL( "Returned expected frame", ts.get_frame(), expected_frame );
    ++expected_frame;
  }
}
