/*ckwg +29
 * Copyright 2016-2020 by Kitware, Inc.
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
 * \brief test class_map class
 */

#include <vital/types/class_map.txx>

#include <vital/types/activity_type.h>
#include <vital/types/detected_object_type.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

namespace {

std::vector<std::string> const names =
  { "person", "vehicle", "other", "clam", "barnacle" };

std::vector<double> const scores  = { 0.65, 0.6, 0.07, 0.055, 0.005 };

struct test_class_map_tag {};

}

template class kwiver::vital::class_map< test_class_map_tag >;
using test_class_map = class_map< test_class_map_tag >;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(class_map, api)
{
  test_class_map cm( names, scores );

  EXPECT_EQ( 0.07, cm.score( "other" ) );

  std::string ml_name;
  double ml_score;
  cm.get_most_likely( ml_name, ml_score );

  EXPECT_EQ( "person", ml_name );
  EXPECT_EQ( 0.65, ml_score );

  for ( size_t i = 0; i < names.size(); ++i )
  {
    SCOPED_TRACE(
      "For score " + std::to_string( i ) + " ('" + names[i] + "')" );

    EXPECT_EQ( scores[i], cm.score( names[i] ) );
  }

  EXPECT_EQ( 0.055, cm.score( "clam" ) );

  cm.set_score( "clam", 1.23 );
  EXPECT_EQ( 1.23, cm.score( "clam" ) );

  EXPECT_EQ( 5, cm.class_names().size() );

  EXPECT_NO_THROW( cm.score( "other" ) ); // make sure this entry exists
  cm.delete_score( "other" );
  EXPECT_THROW( cm.score("other"), std::runtime_error )
    << "Accessing deleted class name";

  EXPECT_EQ( 4, cm.class_names().size() );

  for ( auto const& name : cm.class_names() )
  {
    std::cout << " -- " << name << "    score: "
              << cm.score( name ) << std::endl;
  }

}

// ----------------------------------------------------------------------------
TEST(class_map, creation_error)
{
  auto wrong_size_scores = scores;
  wrong_size_scores.resize( 4 );

  EXPECT_THROW(
    test_class_map cm( {}, {} ),
    std::invalid_argument );

  EXPECT_THROW(
    test_class_map cm( names, wrong_size_scores ),
    std::invalid_argument );
}

// ----------------------------------------------------------------------------
TEST(class_map, name_pool)
{
  test_class_map cm( names, scores );

  std::vector<std::string> alt_names =
    { "a-person", "a-vehicle", "a-other", "a-clam", "a-barnacle" };

  test_class_map cm_2( alt_names, scores );

  EXPECT_EQ( 10, test_class_map::all_class_names().size() );
  EXPECT_EQ( 0, activity_type::all_class_names().size() );
  EXPECT_EQ( 0, detected_object_type::all_class_names().size() );

  for ( auto const& name : test_class_map::all_class_names() )
  {
    std::cout << "  --  " << name << std::endl;
  }

}
