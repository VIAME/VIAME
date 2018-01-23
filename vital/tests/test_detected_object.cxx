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

#include <vital/types/detected_object.h>

#include <gtest/gtest.h>

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(detected_object, creation)
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d bb( tl, 100, 100 );

  kwiver::vital::detected_object dobj( bb ); // using defaults

  EXPECT_EQ( 1, dobj.confidence() );
  EXPECT_EQ( bb.upper_left(), dobj.bounding_box().upper_left() );
  EXPECT_EQ( bb.lower_right(), dobj.bounding_box().lower_right() );
  EXPECT_EQ( nullptr, dobj.type() );

  // Test associated API
  EXPECT_EQ( 0, dobj.index() );

  dobj.set_index( 1234 );
  EXPECT_EQ( 1234, dobj.index() );

  EXPECT_EQ( "", dobj.detector_name() );

  dobj.set_detector_name( "foo detector" );
  EXPECT_EQ( "foo detector", dobj.detector_name() );
}

// ----------------------------------------------------------------------------
TEST(detected_object, modification)
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d bb( tl, 100, 100 );

  kwiver::vital::detected_object doby( bb ); // start with defaults

  doby.set_confidence( 0.5546 );
  EXPECT_EQ( 0.5546, doby.confidence() );

  std::vector< std::string > names;
  names.push_back( "person" );
  names.push_back( "vehicle" );
  names.push_back( "other" );
  names.push_back( "clam" );
  names.push_back( "barnacle" );

  std::vector< double > score;
  score.push_back( .65 );
  score.push_back( .6 );
  score.push_back( .07 );
  score.push_back( .0055 );
  score.push_back( .005 );

  auto dot = std::make_shared< kwiver::vital::detected_object_type >( names, score );
  doby.set_type( dot );

  EXPECT_NE( nullptr, doby.type() );

  kwiver::vital::bounding_box_d::vector_type tr( 20, 10 );
  kwiver::vital::translate( bb, tr );

  doby.set_bounding_box( bb );
}
