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

// Common test cases for seek_frame in the video_input interface

#ifndef ARROWS_CORE_TEST_SEEK_FRAME_COMMON_H
#define ARROWS_CORE_TEST_SEEK_FRAME_COMMON_H

#include <vital/algo/video_input.h>

void test_seek_frame(kwiver::vital::algo::video_input& vi)
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  EXPECT_TRUE( vi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {3, 23, 46, 34, 50, 1};
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
    {-3, -1, 0, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }
}

void test_seek_frame_sublist(kwiver::vital::algo::video_input& vi)
{
  kwiver::vital::timestamp ts;

  // Video should be seekable
  EXPECT_TRUE( vi.seekable() );

  // Test various valid seeks
  std::vector<kwiver::vital::timestamp::frame_t> valid_seeks =
    {11, 17, 28, 21, 30};
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
    {-3, -1, 5, 10, 31, 42, 51, 55};
  for (auto requested_frame : in_valid_seeks)
  {
    EXPECT_FALSE( vi.seek_frame( ts, requested_frame) );
    EXPECT_NE( requested_frame, ts.get_frame() );
  }
}

#endif /* ARROWS_CORE_TEST_SEEK_FRAME_COMMON_H */
