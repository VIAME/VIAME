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
#include <arrows/core/tests/seek_frame_common.h>
#include <arrows/vxl/vidl_ffmpeg_video_input.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <memory>
#include <string>
#include <iostream>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
static int num_expected_frames = 50;
static int num_expected_frames_subset = 20;
static int nth_frame_output = 3;
static int num_expected_frames_nth_output = 17;
static std::string video_file_name = "video.mp4";

// ----------------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  ::testing::InitGoogleTest( &argc, argv );

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
  kwiver::vital::plugin_manager::instance().load_all_plugins();
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

  EXPECT_EQ( num_expected_frames, vfvi.num_frames() )
    << "Number of frames before extracting frames should be "
    << num_expected_frames;

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
  EXPECT_EQ( num_expected_frames, num_frames )
    << "Number of frames found should be "
    << num_expected_frames;
  EXPECT_EQ( num_expected_frames, vfvi.num_frames() )
    << "Number of frames after extracting frames should be "
    << num_expected_frames;
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, read_video_subset)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "start_at_frame", "11" );
  config->set_value( "stop_after_frame", "30" );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  kwiver::vital::timestamp ts;

  int num_frames = 0;
  int frame_idx = 10;
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
    ++frame_idx;
    EXPECT_EQ( frame_idx, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }
  EXPECT_EQ( num_expected_frames_subset, num_frames );
  EXPECT_EQ( num_expected_frames_subset, vfvi.num_frames() );
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
  EXPECT_EQ( num_expected_frames, vfvi.num_frames() )
    << "Number of frames after checking frames should be "
    << num_expected_frames;
}

TEST_F(vidl_ffmpeg_video_input, seek_frame)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( video_file );

  test_seek_frame( vfvi );

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

  // Open the video
  vfvi.open( video_file );

  test_seek_frame_sublist( vfvi );

  vfvi.close();
}

TEST_F(vidl_ffmpeg_video_input, metadata_map)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( video_file );

  // Get metadata map
  auto md_map = vfvi.metadata_map()->metadata();

  // Each frame of video should have some metadata
  // at a minimum this is just the video name and timestamp
  EXPECT_EQ( md_map.size(), vfvi.num_frames() );

  if ( md_map.size() != vfvi.num_frames() )
  {
    std::cout << "Found metadata on these frames: ";
    for (auto md : md_map)
    {
      std::cout << md.first << ", ";
    }
    std::cout << std::endl;
  }
}

TEST_F(vidl_ffmpeg_video_input, metadata_map_subset)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  config->set_value("start_at_frame", "11");
  config->set_value("stop_after_frame", "30");

  vfvi.check_configuration(config);
  vfvi.set_configuration(config);

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open(video_file);

  // Advance into the video to make sure these methods work
  // even when not called on the first frame
  kwiver::vital::timestamp ts;
  for (int j = 0; j < 10; ++j)
    vfvi.next_frame(ts);

  // Get metadata map
  auto md_map = vfvi.metadata_map()->metadata();

  // Each frame of video should have some metadata
  // at a minimum this is just the video name and timestamp
  EXPECT_EQ(md_map.size(), vfvi.num_frames());

  if (md_map.size() != vfvi.num_frames())
  {
    std::cout << "Found metadata on these frames: ";
    for (auto md : md_map)
    {
      std::cout << md.first << ", ";
    }
    std::cout << std::endl;
  }
}

TEST_F(vidl_ffmpeg_video_input, metadata_map_nth_frame)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  config->set_value("start_at_frame", "11");
  config->set_value("stop_after_frame", "30");
  config->set_value("output_nth_frame", nth_frame_output);

  vfvi.check_configuration(config);
  vfvi.set_configuration(config);

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open(video_file);

  // Advance into the video to make sure these methods work
  // even when not called on the first frame
  kwiver::vital::timestamp ts;
  for (int j = 0; j < 3; ++j)
    vfvi.next_frame(ts);

  // Get metadata map
  auto md_map = vfvi.metadata_map()->metadata();

  // Each frame of video should have some metadata
  // at a minimum this is just the video name and timestamp
  EXPECT_EQ(md_map.size(), 6);

  if (md_map.size() != 6)
  {
    std::cout << "Found metadata on these frames: ";
    for (auto md : md_map)
    {
      std::cout << md.first << ", ";
    }
    std::cout << std::endl;
  }
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, read_video_nth_frame_output)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "output_nth_frame", nth_frame_output );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  kwiver::vital::timestamp ts;

  EXPECT_EQ( num_expected_frames, vfvi.num_frames() )
    << "Number of frames before extracting frames should be "
    << num_expected_frames;

  int num_frames = 0;
  int expected_frame_num = 1;
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
    EXPECT_EQ( expected_frame_num, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
    expected_frame_num += 3;
  }
  EXPECT_EQ( num_expected_frames_nth_output, num_frames )
    << "Number of frames found should be "
    << num_expected_frames;
  EXPECT_EQ( num_expected_frames, vfvi.num_frames() )
    << "Number of frames after extracting frames should be "
    << num_expected_frames;
}

TEST_F(vidl_ffmpeg_video_input, seek_nth_frame_output)
{
  // make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "output_nth_frame", nth_frame_output );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( video_file );

  kwiver::vital::timestamp ts;

  // Video should be seekable
  EXPECT_TRUE( vfvi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {4, 10, 13, 22, 49};
  for (auto requested_frame : valid_seeks)
  {
    EXPECT_TRUE( vfvi.seek_frame( ts, requested_frame) );

    auto img = vfvi.frame_image();

    EXPECT_EQ( requested_frame, ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of vfvideo
  std::vector<kwiver::vital::timestamp::frame_t> in_valid_seeks =
    {-3, -1, 0, 2, 12, 11, 21, 24, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vfvi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }

  vfvi.close();
}
