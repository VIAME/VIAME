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
 * \brief test detected_object class
 */

#include <vital/types/detected_object_set.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

namespace {

std::vector<std::string> const names =
  { "person", "vehicle", "other", "clam", "barnacle" };

std::vector<double> const scores  = { 0.65, 0.6, 0.005, 0.07, 0.005 };
std::vector<double> const scores1 = { 0.0065, 0.006, 0.005, 0.775, 0.605 };
std::vector<double> const scores2 = { 0.0065, 0.006, 0.005, 0.605, 0.775 };
std::vector<double> const scores3 = { 0.5065, 0.006, 0.005, 0.775, 0.605 };

}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {

// ----------------------------------------------------------------------------
detected_object_set_sptr make_do_set()
{
  auto do_set = std::make_shared<detected_object_set>();

  bounding_box_d bb{ 10, 20, 30, 40 };

  auto dot = std::make_shared<detected_object_type>( names, scores );

  do_set->add( std::make_shared<detected_object>( bb ) ); // using defaults

  EXPECT_EQ( 1, do_set->size() );

  do_set->add( std::make_shared<detected_object>( bb, 0.65, dot ) );

  auto dot1 = std::make_shared<detected_object_type>( names, scores1 );
  do_set->add( std::make_shared<detected_object>( bb, 0.75, dot1 ) );

  auto dot2 = std::make_shared<detected_object_type>( names, scores2 );
  do_set->add( std::make_shared<detected_object>( bb, 0.78, dot2 ) );

  auto dot3 = std::make_shared<detected_object_type>( names, scores3 );
  do_set->add( std::make_shared<detected_object>( bb, 0.70, dot3 ) );

  EXPECT_EQ( 5, do_set->size() );

  return do_set;
}

// ----------------------------------------------------------------------------
void test_do_set( detected_object_set_sptr const& do_set )
{
  [&]{
    // get whole list sorted by confidence
    auto const& list = do_set->select();

    ASSERT_EQ( 5, list->size() );

    auto it = list->cbegin();

    EXPECT_EQ( 1.00, ( *it++ )->confidence() );
    EXPECT_EQ( 0.78, ( *it++ )->confidence() );
    EXPECT_EQ( 0.75, ( *it++ )->confidence() );
    EXPECT_EQ( 0.70, ( *it++ )->confidence() );
    EXPECT_EQ( 0.65, ( *it++ )->confidence() );
  }();

  [&]{
    // get list with confidence threshold
    auto const& list = do_set->select( 0.75 );

    ASSERT_EQ( 3, list->size() );

    auto it = list->cbegin();

    EXPECT_EQ( 1.00, ( *it++ )->confidence() );
    EXPECT_EQ( 0.78, ( *it++ )->confidence() );
    EXPECT_EQ( 0.75, ( *it++ )->confidence() );
  }();

  [&]{
    // get list by object type
    auto const& list = do_set->select( "clam" );

    ASSERT_EQ( 4, list->size() );

    EXPECT_EQ( 0.775, list->at( 0 )->type()->score( "clam" ) );
    EXPECT_EQ( 0.775, list->at( 1 )->type()->score( "clam" ) );
    EXPECT_EQ( 0.605, list->at( 2 )->type()->score( "clam" ) );
    EXPECT_EQ( 0.07,  list->at( 3 )->type()->score( "clam" ) );

    EXPECT_THROW( list->at( 4 ), std::out_of_range );

    auto it = list->cbegin();

    EXPECT_EQ( 0.775, ( *it++ )->type()->score( "clam" ) );
    EXPECT_EQ( 0.775, ( *it++ )->type()->score( "clam" ) );
    EXPECT_EQ( 0.605, ( *it++ )->type()->score( "clam" ) );
    EXPECT_EQ( 0.07,  ( *it++ )->type()->score( "clam" ) );
  }();
}

} // end namespace

// ----------------------------------------------------------------------------
TEST(detected_object_set, api)
{
  test_do_set( make_do_set() );
}

// ----------------------------------------------------------------------------
TEST(detected_object_set, clone)
{
  test_do_set( make_do_set()->clone() );
}

// ----------------------------------------------------------------------------
TEST(detected_object_set, clone_2)
{
  detected_object_set do_set;

  bounding_box_d bb{ 10, 20, 30, 40 };

  auto dot = std::make_shared<detected_object_type>( names, scores );

  auto detection = std::make_shared<detected_object>( bb ); // using defaults
  do_set.add( detection );

  EXPECT_EQ( 1, do_set.clone()->size() );

  auto attr_set = std::make_shared<attribute_set>();
  do_set.set_attributes( attr_set );

  EXPECT_EQ( 1, do_set.clone()->size() );
}
