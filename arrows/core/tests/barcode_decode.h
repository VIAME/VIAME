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

#ifndef ARROWS_CORE_TEST_BARCODE_DECODE_H
#define ARROWS_CORE_TEST_BARCODE_DECODE_H

#include <vital/types/image_container.h>

// Ignore 8 pixels on either side of the barcode
static int bc_buffer = 8;

// Barcode lines two pixels wide and 4 pixels high
static int bc_width = 2;
static int bc_height = 4;
static int bit_depth = 256;
static int bc_area = bc_width*bc_height;

// Color test pixel location
static int color_test_pos = 17;

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

#endif /* ARROWS_CORE_TEST_BARCODE_DECODE_H */
