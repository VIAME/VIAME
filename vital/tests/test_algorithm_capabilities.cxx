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
 * \brief test capabilities class
 */

#include <vital/algorithm_capabilities.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
class algorithm_capabilities : public ::testing::Test
{
public:
  void SetUp()
  {
    cap.set_capability( "cap1", true );
    cap.set_capability( "cap2", false );
  }

  kwiver::vital::algorithm_capabilities cap;
};

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, empty)
{
  kwiver::vital::algorithm_capabilities cap_empty;

  EXPECT_EQ( false, cap_empty.has_capability( "test" ) );

  auto cap_list = cap_empty.capability_list();
  EXPECT_TRUE( cap_list.empty() );
  EXPECT_EQ( 0, cap_list.size() );
}

// ----------------------------------------------------------------------------
static void test_capabilities(
  kwiver::vital::algorithm_capabilities const& cap )
{
  auto cap_list = cap.capability_list();
  EXPECT_EQ( 2, cap_list.size() );

  EXPECT_TRUE( cap.has_capability( "cap1" ) );
  EXPECT_TRUE( cap.has_capability( "cap2" ) );

  EXPECT_EQ( true, cap.capability( "cap1" ) );
  EXPECT_EQ( false, cap.capability( "cap2" ) );
}

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, api)
{
  test_capabilities( cap );
}

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, copy)
{
  kwiver::vital::algorithm_capabilities cap_copied( cap );
  test_capabilities( cap_copied );
}

// ----------------------------------------------------------------------------
TEST_F(algorithm_capabilities, assign)
{
  kwiver::vital::algorithm_capabilities cap_assigned;
  cap_assigned = cap;
  test_capabilities( cap_assigned );
}
