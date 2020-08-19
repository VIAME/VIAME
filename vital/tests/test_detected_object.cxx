/*ckwg +29
 * Copyright 2016-2017, 2019 by Kitware, Inc.
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
#include <vital/types/geodesy.h>

#include <gtest/gtest.h>

static auto const loc = kwiver::vital::vector_3d{ -73.759291, 42.849631, 50 };
static auto constexpr crs = kwiver::vital::SRID::lat_lon_WGS84;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(detected_object, creation)
{
  kwiver::vital::bounding_box_d::vector_type tl{ 12, 23 };
  kwiver::vital::bounding_box_d bb{ tl, 100, 100 };

  kwiver::vital::detected_object dobj{ bb };

  // Test expected values passed to constructor
  EXPECT_EQ( 1.0, dobj.confidence() );
  EXPECT_EQ( bb.upper_left(), dobj.bounding_box().upper_left() );
  EXPECT_EQ( bb.lower_right(), dobj.bounding_box().lower_right() );
  EXPECT_EQ( nullptr, dobj.type() );

  // Test expected default values
  EXPECT_EQ( 0, dobj.index() );
  EXPECT_EQ( "", dobj.detector_name() );

  kwiver::vital::geo_point gp{ loc, crs };
  kwiver::vital::detected_object dobj2{ gp };

  // Test expected values passed to constructor
  EXPECT_EQ( loc, dobj2.geo_point().location( crs ) );
}

// ----------------------------------------------------------------------------
TEST(detected_object, modification)
{
  kwiver::vital::bounding_box_d::vector_type tl{ 12, 23 };
  kwiver::vital::bounding_box_d bb{ tl, 100, 100 };

  kwiver::vital::detected_object dobj{ bb };

  dobj.set_index( 1234 );
  EXPECT_EQ( 1234, dobj.index() );

  dobj.set_detector_name( "foo detector" );
  EXPECT_EQ( "foo detector", dobj.detector_name() );

  dobj.set_confidence( 0.5546 );
  EXPECT_EQ( 0.5546, dobj.confidence() );

  auto const names = std::vector< std::string >{
    "person",
    "vehicle",
    "other",
    "clam",
    "barnacle",
  };

  auto const scores = std::vector< double >{
    0.65,
    0.6,
    0.07,
    0.0055,
    0.005,
  };

  auto const dot =
    std::make_shared< kwiver::vital::detected_object_type >( names, scores );
  dobj.set_type( dot );
  EXPECT_EQ( dot, dobj.type() );

  kwiver::vital::bounding_box_d::vector_type offset{ 20, 10 };
  kwiver::vital::translate( bb, offset );
  dobj.set_bounding_box( bb );
  EXPECT_EQ( bb.upper_left(), dobj.bounding_box().upper_left() );
  EXPECT_EQ( bb.lower_right(), dobj.bounding_box().lower_right() );

  dobj.set_geo_point( { loc, crs } );
  EXPECT_EQ( loc, dobj.geo_point().location( crs ) );
}

// ----------------------------------------------------------------------------
TEST(detected_object, keypoints)
{
  kwiver::vital::detected_object dobj;

  dobj.add_keypoint( "head", { 11.2, 13.1 } );
  dobj.add_keypoint( "head", { 4.2, 9.5 } );

  auto const& keypoints = dobj.keypoints();
  EXPECT_EQ( 1, keypoints.size() );
  ASSERT_EQ( 1, keypoints.count( "head" ) );

  EXPECT_EQ( 4.2, keypoints.find( "head" )->second.value()[ 0 ] );
  EXPECT_EQ( 9.5, keypoints.find( "head" )->second.value()[ 1 ] );
}

// ----------------------------------------------------------------------------
TEST(detected_object, notes)
{
  kwiver::vital::detected_object dobj;

  EXPECT_EQ( dobj.notes().size(), 0 );

  dobj.add_note( "Dogs have owners." );
  dobj.add_note( "Cats have staff." );

  auto const& notes = dobj.notes();
  ASSERT_EQ( notes.size(), 2 );
  EXPECT_EQ( "Dogs have owners.", notes[ 0 ] );
  EXPECT_EQ( "Cats have staff.", notes[ 1 ] );

  dobj.clear_notes();
  EXPECT_EQ( dobj.notes().size(), 0 );
}
