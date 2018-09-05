/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief test opening/closing a video file
 */

#include <test_gtest.h>

#include <arrows/ffmpeg/ffmpeg_video_input.h>
#include <arrows/core/tests/barcode_decode.h>
#include <vital/exceptions/io.h>
#include <vital/plugin_loader/plugin_manager.h>

#include <arrows/vxl/vidl_ffmpeg_video_input.h>

#include <memory>
#include <string>
#include <iostream>

kwiver::vital::path_t g_data_dir;

namespace algo = kwiver::vital::algo;

static int TOTAL_NUMBER_OF_FRAMES = 50;

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
class ffmpeg_video_input : public ::testing::Test
{
  TEST_ARG(data_dir);
};

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, create)
{
  EXPECT_NE( nullptr, algo::video_input::create("ffmpeg") );
}

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, is_good_correct_file_path)
{
  kwiver::arrows::ffmpeg::ffmpeg_video_input input;
  kwiver::vital::path_t correct_file = data_dir + "/video.mp4";

  EXPECT_FALSE(input.good())
    << "Video state before open";

  // open the video
  input.open(correct_file);
  EXPECT_FALSE(input.good())
    << "Video state after open but before first frame";

  // Get the next frame
  kwiver::vital::timestamp ts;
  EXPECT_TRUE(input.next_frame(ts))
    << "Video state after open but before first frame";
  EXPECT_EQ(ts.get_frame(), 1) << "Initial frame value mismastch";
  EXPECT_TRUE(input.good())
    << "Video state after open but before first frame";

  // close the video
  input.close();
  EXPECT_FALSE(input.good())
    << "Video state after close";
}

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, is_good_invalid_file_path)
{
  kwiver::arrows::ffmpeg::ffmpeg_video_input input;
  kwiver::vital::path_t incorrect_file = data_dir + "/DoesNOTExists.mp4";

  EXPECT_FALSE(input.good())
    << "Video state before open";

  // open the video
  EXPECT_THROW(
    input.open(incorrect_file),
    kwiver::vital::file_not_found_exception);
  EXPECT_FALSE(input.good())
    << "Video state after open but before first frame";

  // Get the next frame
  kwiver::vital::timestamp ts;
  EXPECT_THROW(input.next_frame(ts),
    kwiver::vital::file_not_read_exception);
  EXPECT_EQ(ts.get_frame(), 0) << "Initial frame value mismastch";
  EXPECT_FALSE(input.good())
    << "Video state after open but before first frame";

  // close the video
  input.close();
  EXPECT_FALSE(input.good())
    << "Video state after close";
}

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, frame_image)
{
  kwiver::arrows::ffmpeg::ffmpeg_video_input input;
  kwiver::vital::path_t correct_file = data_dir + "/video.mp4";

  EXPECT_FALSE(input.good())
    << "Video state before open";

  // open the video
  input.open(correct_file);
  EXPECT_FALSE(input.good())
    << "Video state after open but before first frame";
  EXPECT_EQ(input.frame_image(), nullptr) << "Video should not have an image yet";

  // Get the next frame
  kwiver::vital::timestamp ts;
  input.next_frame(ts);
  EXPECT_EQ(ts.get_frame(), 1);

  kwiver::vital::image_container_sptr frame = input.frame_image();
  EXPECT_EQ(frame->depth(), 3);
  EXPECT_EQ(frame->get_image().width(), 80);
  EXPECT_EQ(frame->get_image().height(), 54);
  EXPECT_EQ(frame->get_image().d_step(), 4320*3);
  EXPECT_EQ(frame->get_image().h_step(), 80*3);
  EXPECT_EQ(frame->get_image().w_step(), 3);
  EXPECT_EQ(frame->get_image().is_contiguous(), false);

  EXPECT_EQ(decode_barcode(*frame), 1);
}

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, seek)
{
  kwiver::arrows::ffmpeg::ffmpeg_video_input input;
  EXPECT_TRUE(input.seekable()) << "Seekable";

  kwiver::vital::path_t correct_file = data_dir + "/video.mp4";

  EXPECT_FALSE(input.good())
    << "Video state before open";

  // open the video
  input.open(correct_file);
  EXPECT_FALSE(input.good())
    << "Video state after open but before first frame";
  EXPECT_EQ(input.frame_image(), nullptr) << "Video should not have an image yet";

  kwiver::vital::timestamp ts;

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
  { 3, 23, 46, 34, 50, 1 };
  for (auto requested_frame : valid_seeks)
  {
    EXPECT_TRUE(input.seek_frame(ts, requested_frame));
    EXPECT_EQ(requested_frame, ts.get_frame())
      << "Frame number should match seek request";

    auto img = input.frame_image();
    EXPECT_EQ(ts.get_frame(), decode_barcode(*img))
      << "Frame number should match barcode in frame image";
  }
  // Test various invalid seeks past end of video
  std::vector<kwiver::vital::timestamp::frame_t> in_valid_seeks =
  { -3, -1, 0, 51, 55 };
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE(input.seek_frame(ts, requested_frame));
    EXPECT_NE(requested_frame, ts.get_frame());
  }
}

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, end_of_video)
{
  kwiver::arrows::ffmpeg::ffmpeg_video_input input;

  kwiver::vital::path_t correct_file = data_dir + "/video.mp4";

  EXPECT_TRUE(input.end_of_video())
    << "End of video before open";

  // open the video
  input.open(correct_file);
  EXPECT_FALSE(input.end_of_video())
    << "End of video after open";

  kwiver::vital::timestamp ts;
  while (input.next_frame(ts))
  {
    EXPECT_FALSE(input.end_of_video()) << "End of video while reading";
  }

  EXPECT_EQ(ts.get_frame(), TOTAL_NUMBER_OF_FRAMES) << "Last frame";
  EXPECT_TRUE(input.end_of_video()) << "End of video after last frame";
}

// ----------------------------------------------------------------------------
TEST_F(ffmpeg_video_input, read_video)
{
  // make config block
  kwiver::arrows::ffmpeg::ffmpeg_video_input input;

  kwiver::vital::path_t correct_file = data_dir + "/video.mp4";

  input.open(correct_file);

  kwiver::vital::timestamp ts;

  EXPECT_EQ(TOTAL_NUMBER_OF_FRAMES, input.num_frames())
    << "Number of frames before extracting frames should be "
    << TOTAL_NUMBER_OF_FRAMES;

  int num_frames = 0;
  while (input.next_frame(ts))
  {
    auto img = input.frame_image();

    ++num_frames;
    EXPECT_EQ(num_frames, ts.get_frame())
      << "Frame numbers should be sequential";
    EXPECT_EQ(ts.get_frame(), decode_barcode(*img))
      << "Frame number should match barcode in frame image";
  }
  EXPECT_EQ(TOTAL_NUMBER_OF_FRAMES, num_frames)
    << "Number of frames found should be "
    << TOTAL_NUMBER_OF_FRAMES;
  EXPECT_EQ(TOTAL_NUMBER_OF_FRAMES, input.num_frames())
    << "Number of frames after extracting frames should be "
    << TOTAL_NUMBER_OF_FRAMES;
}
