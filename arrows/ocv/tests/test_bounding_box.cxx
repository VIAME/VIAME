/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
 * \brief test VXL bundle adjustment functionality
 */

#include <arrows/ocv/bounding_box.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(bounding_box, convert_bb2ocv)
{
  kwiver::vital::bounding_box<double> bbox( 1, 3, 10, 34 );
  auto const& vbox = kwiver::arrows::ocv::convert( bbox );

  EXPECT_EQ( bbox.min_x(), vbox.x );
  EXPECT_EQ( bbox.min_y(), vbox.y );
  EXPECT_EQ( bbox.width(), vbox.width );
  EXPECT_EQ( bbox.height(), vbox.height );
}

// ----------------------------------------------------------------------------
TEST(bounding_box, convert_ocv2bb)
{
  CvRect vbox= cvRect( 1, 3, 10, 34 );
  auto const& bbox = kwiver::arrows::ocv::convert<double>( vbox );

  EXPECT_EQ( vbox.x, bbox.min_x() );
  EXPECT_EQ( vbox.y, bbox.min_y() );
  EXPECT_EQ( vbox.width, bbox.width() );
  EXPECT_EQ( vbox.height, bbox.height() );
}
