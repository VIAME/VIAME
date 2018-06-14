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
  EXPECT_EQ(ts.get_frame(), 0) << "Initial frame value mismastch";
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
  EXPECT_EQ(ts.get_frame(), 0);
  EXPECT_EQ(ts.get_time_domain_index(), 0);

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
