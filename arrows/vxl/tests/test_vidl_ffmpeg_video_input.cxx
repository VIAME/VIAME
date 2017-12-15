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

#include <test_gtest.h>

#include <arrows/core/tests/barcode_decode.h>
#include <arrows/vxl/vidl_ffmpeg_video_input.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <memory>
#include <string>
#include <iostream>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
static int num_expected_frames = 50;
static std::string video_file_name = "video.mp4";

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );

  kwiver::vital::plugin_manager::instance().load_all_plugins();

  GET_ARG(1, g_data_dir);

  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class vidl_ffmpeg_video_input : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, create)
{
  EXPECT_NE( nullptr, algo::video_input::create("vidl_ffmpeg") );
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, read_video)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  kwiver::vital::timestamp ts;

  int num_frames = 0;
  while ( vfvi.next_frame( ts ) )
  {
    auto img = vfvi.frame_image();
    auto md = vfvi.frame_metadata();

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
  EXPECT_EQ( num_expected_frames, num_frames );
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, is_good)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  kwiver::vital::timestamp ts;

  EXPECT_FALSE( vfvi.good() )
    << "Video state before open";

  // open the video
  vfvi.open( video_file );
  EXPECT_FALSE( vfvi.good() )
    << "Video state after open but before first frame";

  // step one frame
  vfvi.next_frame( ts );
  EXPECT_TRUE( vfvi.good() )
    << "Video state on first frame";

  // close the video
  vfvi.close();
  EXPECT_FALSE( vfvi.good() )
    << "Video state after close";

  // Reopen the video
  vfvi.open( video_file );

  int num_frames = 0;
  while ( vfvi.next_frame( ts ) )
  {
      ++num_frames;
    EXPECT_TRUE( vfvi.good() )
      << "Video state on frame " << ts.get_frame();
  }
  EXPECT_EQ( num_expected_frames, num_frames );
}

TEST_F(vidl_ffmpeg_video_input, seek_frame)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  kwiver::vital::timestamp ts;

  // Open the video
  vfvi.open( video_file );

  // Video should be seekable
  EXPECT_TRUE( vfvi.seekable() );

  // Test various valid seeks
  int num_seeks = 6;
  kwiver::vital::timestamp::frame_t valid_seeks[num_seeks] =
    {3, 23, 46, 34, 50, 1};
  for (int i=0; i<num_seeks; ++i)
  {
    EXPECT_TRUE( vfvi.seek_frame( ts, valid_seeks[i]) );

    auto img = vfvi.frame_image();

    EXPECT_EQ( valid_seeks[i], ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of video
  num_seeks = 4;
  kwiver::vital::timestamp::frame_t in_valid_seeks[num_seeks] =
    {-3, -1, 51, 55};
  for (int i=0; i<num_seeks; ++i)
  {
    EXPECT_FALSE( vfvi.seek_frame( ts, in_valid_seeks[i]) );
    EXPECT_NE( in_valid_seeks[i], ts.get_frame() );
  }

  vfvi.close();
}

TEST_F(vidl_ffmpeg_video_input, seek_frame_sublist)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "start_at_frame", "11" );
  config->set_value( "stop_after_frame", "30" );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  kwiver::vital::timestamp ts;

  // Open the video
  vfvi.open( video_file );

  // Video should be seekable
  EXPECT_TRUE( vfvi.seekable() );

  // Test various valid seeks
  int num_seeks = 5;
  kwiver::vital::timestamp::frame_t valid_seeks[num_seeks] =
    {11, 17, 28, 21, 30};
  for (int i=0; i<num_seeks; ++i)
  {
    EXPECT_TRUE( vfvi.seek_frame( ts, valid_seeks[i]) );

    auto img = vfvi.frame_image();

    EXPECT_EQ( valid_seeks[i], ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of video
  num_seeks = 8;
  kwiver::vital::timestamp::frame_t in_valid_seeks[num_seeks] =
    {-3, -1, 5, 10, 31, 42, 51, 55};
  for (int i=0; i<num_seeks; ++i)
  {
    EXPECT_FALSE( vfvi.seek_frame( ts, in_valid_seeks[i]) ) << " for " << in_valid_seeks[i];
    EXPECT_NE( in_valid_seeks[i], ts.get_frame() );
  }

  vfvi.close();
}
