/*ckwg +29
 * Copyright 2013-2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief test core vector functionality
 */

#include <vital/types/vector.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(vector, construct_2d)
{
  kwiver::vital::vector_2d v2d{ 10.0, 33.3 };
  EXPECT_EQ( 10.0, v2d.x() );
  EXPECT_EQ( 33.3, v2d.y() );

  kwiver::vital::vector_2f v2f{ 5.0f, 4.5f };
  EXPECT_EQ( 5.0f, v2f.x() );
  EXPECT_EQ( 4.5f, v2f.y() );
}

// ----------------------------------------------------------------------------
TEST(vector, construct_3d)
{
  kwiver::vital::vector_3d v3d{ 10.0, 33.3, 12.1 };
  EXPECT_EQ( 10.0, v3d.x() );
  EXPECT_EQ( 33.3, v3d.y() );
  EXPECT_EQ( 12.1, v3d.z() );

  kwiver::vital::vector_3f v3f{ 5.0f, 4.5f, -6.3f };
  EXPECT_EQ( 5.0f, v3f.x() );
  EXPECT_EQ( 4.5f, v3f.y() );
  EXPECT_EQ( -6.3f, v3f.z() );
}

// ----------------------------------------------------------------------------
TEST(vector, construct_4d)
{
  kwiver::vital::vector_4d v4d{ 10.0, 33.3, 12.1, 0.0 };
  EXPECT_EQ( 10.0, v4d.x() );
  EXPECT_EQ( 33.3, v4d.y() );
  EXPECT_EQ( 12.1, v4d.z() );
  EXPECT_EQ( 0.0, v4d.w() );

  kwiver::vital::vector_4f v4f{ 5.0f, 4.5f, -6.3f, 100.0f };
  EXPECT_EQ( 5.0f, v4f.x() );
  EXPECT_EQ( 4.5f, v4f.y() );
  EXPECT_EQ( -6.3f, v4f.z() );
  EXPECT_EQ( 100.0f, v4f.w() );
}
