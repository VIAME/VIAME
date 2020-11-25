// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test reading video from a list of images.
 */

#include <test_gtest.h>

#include <arrows/tests/test_video_input.h>
#include <arrows/vxl/vidl_ffmpeg_video_input.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <memory>
#include <string>
#include <iostream>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;
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
  // Make config block
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
TEST_F(vidl_ffmpeg_video_input, read_video_sublist)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "start_at_frame", start_at_frame );
  config->set_value( "stop_after_frame", stop_after_frame );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  test_read_video_sublist( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, read_video_sublist_nth_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "start_at_frame", start_at_frame );
  config->set_value( "stop_after_frame", stop_after_frame );
  config->set_value( "output_nth_frame", nth_frame_output );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  EXPECT_TRUE( vfvi.check_configuration( config ) );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  test_read_video_sublist_nth_frame( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, seek_frame_sublist_nth_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "start_at_frame", start_at_frame );
  config->set_value( "stop_after_frame", stop_after_frame );
  config->set_value( "output_nth_frame", nth_frame_output );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  EXPECT_TRUE( vfvi.check_configuration( config ) );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  test_seek_sublist_nth_frame( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, is_good)
{
  // Make config block
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

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, seek_frame)
{
  // Make config block
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

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, seek_frame_sublist)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "start_at_frame", start_at_frame );
  config->set_value( "stop_after_frame", stop_after_frame );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( video_file );

  test_seek_frame_sublist( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, seek_then_next_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  EXPECT_TRUE( vfvi.check_configuration( config ) );
  vfvi.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( list_file );

  test_seek_then_next( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, next_then_seek_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  EXPECT_TRUE( vfvi.check_configuration( config ) );
  vfvi.set_configuration( config );

  kwiver::vital::path_t list_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( list_file );

  test_next_then_seek( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, next_then_seek_then_next)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  EXPECT_TRUE( vfvi.check_configuration( config ) );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( video_file );

  test_next_then_seek_then_next( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, metadata_map)
{
  // Make config block
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

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, metadata_map_subset)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  config->set_value("start_at_frame", start_at_frame);
  config->set_value("stop_after_frame", stop_after_frame);

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

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, metadata_map_nth_frame)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  config->set_value("start_at_frame", start_at_frame);
  config->set_value("stop_after_frame", stop_after_frame);
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
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "output_nth_frame", nth_frame_output );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;
  vfvi.open( video_file );

  test_read_video_nth_frame( vfvi );

  vfvi.close();
}

// ----------------------------------------------------------------------------
TEST_F(vidl_ffmpeg_video_input, seek_nth_frame_output)
{
  // Make config block
  auto config = kwiver::vital::config_block::empty_config();

  config->set_value( "output_nth_frame", nth_frame_output );

  kwiver::arrows::vxl::vidl_ffmpeg_video_input vfvi;

  vfvi.check_configuration( config );
  vfvi.set_configuration( config );

  kwiver::vital::path_t video_file = data_dir + "/" + video_file_name;

  // Open the video
  vfvi.open( video_file );

  test_seek_nth_frame( vfvi );

  vfvi.close();
}
