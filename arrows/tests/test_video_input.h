/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

// Common test cases for seek_frame in the video_input interface

#ifndef ARROWS_TESTS_TEST_VIDEO_INPUT_H
#define ARROWS_TESTS_TEST_VIDEO_INPUT_H

#include <vital/algo/video_input.h>
#include <vital/types/image_container.h>

constexpr int num_expected_frames = 50;
constexpr int start_at_frame = 11;
constexpr int stop_after_frame = 30;
constexpr int num_expected_frames_subset = 20;
constexpr int nth_frame_output = 3;
constexpr int num_expected_frames_nth_output = 17;

// Ignore 8 pixels on either side of the barcode
constexpr int bc_buffer = 8;

// Barcode lines two pixels wide and 4 pixels high
constexpr int bc_width = 2;
constexpr int bc_height = 4;
constexpr int bit_depth = 256;
constexpr int bc_area = bc_width*bc_height;

// Color test pixel location
constexpr int color_test_pos = 17;

// Test colors
static kwiver::vital::rgb_color red(255, 0, 0);
static kwiver::vital::rgb_color green(0, 255, 0);
static kwiver::vital::rgb_color blue(0, 0, 255);

// Decode barcodes from test frame images
uint32_t decode_barcode(kwiver::vital::image_container& img_ct)
{
  auto img = img_ct.get_image();
  kwiver::vital::image_of<uint8_t> frame_img(img);

  uint32_t retVal = 0;
  uint32_t bit_shift = 0;
  int width = static_cast<int>(img.width());
  // Start at the back
  for (int i=width-bc_buffer-1; i > bc_buffer; i-=bc_width)
  {
    uint16_t bc_sum = 0;
    for (int j=0; j < bc_width; ++j)
    {
      for (int k=0; k < bc_height; ++k)
      {
        bc_sum += frame_img(i-j, k);
      }
    }

    if (bc_sum/bc_area < bit_depth/2)
    {
      retVal += (1 << bit_shift);
    }
    bit_shift++;
  }

  return retVal;
}

kwiver::vital::rgb_color test_color_pixel(
  int color, kwiver::vital::image_container& img_ct)
{
  auto img = img_ct.get_image();
  kwiver::vital::image_of<uint8_t> frame_img(img);

  return frame_img.at(2*color + 1, color_test_pos);
}

// ----------------------------------------------------------------------------
void test_read_video_sublist( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  int num_frames = 0;
  int frame_idx = 10;
  while ( vi.next_frame( ts ) )
  {
    auto img = vi.frame_image();
    auto md = vi.frame_metadata();

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
  EXPECT_EQ( num_expected_frames_subset, vi.num_frames() );
}

// ----------------------------------------------------------------------------
void test_read_video_sublist_nth_frame( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  int num_frames = 2;
  int frame_idx = 10;
  while ( vi.next_frame( ts ) )
  {
    auto img = vi.frame_image();
    auto md = vi.frame_metadata();

    if (md.size() > 0)
    {
      std::cout << "-----------------------------------\n" << std::endl;
      kwiver::vital::print_metadata( std::cout, *md[0] );
    }

    num_frames += nth_frame_output;
    frame_idx += nth_frame_output;
    EXPECT_EQ( frame_idx, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( frame_idx, decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }
  EXPECT_EQ( num_expected_frames_subset, num_frames );
  EXPECT_EQ( num_expected_frames_subset, vi.num_frames() );
}

// ----------------------------------------------------------------------------
void test_seek_frame( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {3, 15, 23, 30, 46, 34, 25, 50, 1};
  for (auto requested_frame : valid_seeks)
  {
    EXPECT_TRUE( vi.seek_frame( ts, requested_frame ) )
      << "seek_frame should return true for requested_frame " << requested_frame;

    auto img = vi.frame_image();

    ASSERT_TRUE( !! img );

    EXPECT_EQ( requested_frame, ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( requested_frame, decode_barcode( *img ) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of video
  std::vector<kwiver::vital::timestamp::frame_t> in_valid_seeks =
    {-3, -1, 0, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }

  // Test valid seeks after invalid seeks
  valid_seeks = {11, 32, 21, 43};

  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }

  EXPECT_EQ( 50, vi.num_frames() );
}

// ----------------------------------------------------------------------------
void test_seek_frame_sublist( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {11, 17, 28, 21, 30};
  for (auto requested_frame : valid_seeks)
  {
    EXPECT_TRUE( vi.seek_frame( ts, requested_frame) );

    auto img = vi.frame_image();

    EXPECT_EQ( requested_frame, ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( requested_frame, decode_barcode( *img ) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of video
  std::vector<kwiver::vital::timestamp::frame_t> in_valid_seeks =
    {-3, -1, 5, 10, 31, 42, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }
}

// ----------------------------------------------------------------------------
void test_seek_then_next( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Seek to frame 17, then run over the rest of the video
  const kwiver::vital::timestamp::frame_t requested_frame = 17;

  ASSERT_TRUE( vi.seek_frame( ts, requested_frame ) );

  auto img = vi.frame_image();

  EXPECT_EQ( requested_frame, ts.get_frame() )
    << "Frame number should match seek request";
  EXPECT_EQ( requested_frame, decode_barcode( *img ) )
    << "Frame number should match barcode in frame image";

  int num_frames = requested_frame;
  while ( vi.next_frame( ts ) )
  {
    img = vi.frame_image();

    ++num_frames;
    EXPECT_EQ( num_frames, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( num_frames, decode_barcode( *img ) )
      << "Frame number should match barcode in frame image";
  }
}

// ----------------------------------------------------------------------------
void test_read_video_nth_frame( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  EXPECT_EQ( num_expected_frames, vi.num_frames() )
    << "Number of frames before extracting frames should be "
    << num_expected_frames;

  int num_frames = 0;
  int expected_frame_num = 1;
  while ( vi.next_frame( ts ) )
  {
    auto img = vi.frame_image();
    auto md = vi.frame_metadata();

    if (md.size() > 0)
    {
      std::cout << "-----------------------------------\n" << std::endl;
      kwiver::vital::print_metadata( std::cout, *md[0] );
    }

    ++num_frames;
    EXPECT_EQ( expected_frame_num, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( expected_frame_num, decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
    expected_frame_num += nth_frame_output;
  }
  EXPECT_EQ( num_expected_frames_nth_output, num_frames )
    << "Number of frames found should be "
    << num_expected_frames;
  EXPECT_EQ( num_expected_frames, vi.num_frames() )
    << "Number of frames after extracting frames should be "
    << num_expected_frames;
}

// ----------------------------------------------------------------------------
void test_seek_nth_frame( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {4, 10, 13, 22, 49};
  for (auto requested_frame : valid_seeks)
  {
    EXPECT_TRUE( vi.seek_frame( ts, requested_frame) );

    auto img = vi.frame_image();

    ASSERT_TRUE( !! img );

    EXPECT_EQ( requested_frame, ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of video
  std::vector<kwiver::vital::timestamp::frame_t> in_valid_seeks =
    {-3, -1, 0, 2, 12, 11, 21, 24, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }
}

// ----------------------------------------------------------------------------
void test_next_then_seek( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Frame by frame until frame 45, then seek back to frame 12

  kwiver::vital::timestamp::frame_t stop_frame = 45;
  kwiver::vital::timestamp::frame_t requested_frame = 12;

  int num_frames = 0;
  while ( vi.next_frame( ts ) && ++num_frames < stop_frame )
  {
    auto img = vi.frame_image();

    EXPECT_EQ( num_frames, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( ts.get_frame(), decode_barcode( *img ) )
      << "Frame number should match barcode in frame image";
  }

  ASSERT_TRUE( vi.seek_frame( ts, requested_frame ) );

  auto img = vi.frame_image();

  EXPECT_EQ( requested_frame, ts.get_frame() )
    << "Frame number should match seek request";
  EXPECT_EQ( ts.get_frame(), decode_barcode( *img ) )
    << "Frame number should match barcode in frame image";
}

// ----------------------------------------------------------------------------
void test_next_then_seek_then_next( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Frame by frame until frame 25, then seek back to frame 12, then
  // frame by frame until frame 20

  kwiver::vital::timestamp::frame_t stop_frame = 25;
  kwiver::vital::timestamp::frame_t requested_frame = 12;
  kwiver::vital::timestamp::frame_t end_frame = 20;

  kwiver::vital::frame_id_t num_frames = 0;
  while ( vi.next_frame( ts ) && ++num_frames < stop_frame )
  {
    auto img = vi.frame_image();

    EXPECT_EQ( num_frames, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( num_frames, decode_barcode( *img ) )
      << "Frame number should match barcode in frame image";
  }

  ASSERT_TRUE( vi.seek_frame( ts, requested_frame ) );
  auto seek_img = vi.frame_image();

  EXPECT_EQ( requested_frame, ts.get_frame() )
    << "Frame number should match seek request";
  EXPECT_EQ( requested_frame, decode_barcode( *seek_img ) )
    << "Frame number should match barcode in frame image";

  num_frames = requested_frame;
  while ( vi.next_frame( ts ) && ++num_frames < end_frame )
  {
    auto img = vi.frame_image();

    EXPECT_EQ( num_frames, ts.get_frame() )
      << "Frame numbers should be sequential";
    EXPECT_EQ( num_frames, decode_barcode( *img ) )
      << "Frame number should match barcode in frame image";
  }

  auto img = vi.frame_image();

  EXPECT_EQ( end_frame, ts.get_frame() )
    << "Frame number should match end frame";
  EXPECT_EQ( end_frame, decode_barcode( *img ) )
    << "Frame number should match barcode in frame image";
}

// ----------------------------------------------------------------------------
void test_seek_sublist_nth_frame( kwiver::vital::algo::video_input& vi )
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  ASSERT_TRUE( vi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {13, 16, 19, 22, 25, 28};
  for (auto requested_frame : valid_seeks)
  {
    EXPECT_TRUE( vi.seek_frame( ts, requested_frame) );

    auto img = vi.frame_image();

    EXPECT_EQ( requested_frame, ts.get_frame() )
      << "Frame number should match seek request";
    EXPECT_EQ( ts.get_frame(), decode_barcode(*img) )
      << "Frame number should match barcode in frame image";
  }

  // Test various invalid seeks past end of video
  std::vector<kwiver::vital::timestamp::frame_t> in_valid_seeks =
    {-1, 0, 2, 7, 10, 11, 12, 21, 24, 31, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }
}

#endif // ARROWS_TESTS_TEST_VIDEO_INPUT_H
