/*ckwg +29
 * Copyright 2014-2017, 2019 by Kitware, Inc.
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

#include <vital/tests/test_track_set.h>

// ----------------------------------------------------------------------------
int main( int argc, char** argv )
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(track_set, accessor_functions)
{
  using namespace kwiver::vital::testing;

  auto test_set = make_simple_track_set(1);
  test_track_set_accessors( test_set );
}

// ----------------------------------------------------------------------------
TEST(track_set, modifier_functions)
{
  using namespace kwiver::vital::testing;

  auto test_set = make_simple_track_set(1);
  test_track_set_modifiers( test_set );
}

// ----------------------------------------------------------------------------
TEST(track_set, merge_functions)
{
  using namespace kwiver::vital::testing;

  auto test_set_1 = make_simple_track_set(1);
  auto test_set_2 = make_simple_track_set(2);
  test_track_set_merge(test_set_1, test_set_2);

  auto test_set_3 = std::make_shared< kwiver::vital::track_set >();
  ASSERT_TRUE( test_set_3->empty() );

  test_set_3->merge_in_other_track_set( test_set_2 );

  EXPECT_FALSE( test_set_3->empty() );
  ASSERT_EQ( 4, test_set_3->size() );
}

