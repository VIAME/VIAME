/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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

#include <test_common.h>

#include <vital/types/detected_object.h>

#define TEST_ARGS       ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(detected_object_creation)
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d bb1( tl, 100, 100 );

  kwiver::vital::detected_object dobj( bb1 ); // using defaults

  TEST_EQUAL( "Expected confidence", dobj.confidence(), 1 );

  kwiver::vital::bounding_box_d bb2 = dobj.bounding_box();

  auto ul1 = bb1.upper_left();
  auto lr1 = bb1.lower_right();

  auto ul2 = bb2.upper_left();
  auto lr2 = bb2.lower_right();

  if ( ul1 != ul2 || lr1 != lr2 )
  {
    TEST_ERROR( "Bounding box returned incorrectly" );
  }

  auto dot = dobj.type();

  if ( dot )
  {
    TEST_ERROR( "detected object type returned incorrectly" );
  }

}


// ------------------------------------------------------------------
IMPLEMENT_TEST(detected_object_modification)
{
  kwiver::vital::bounding_box_d::vector_type tl( 12, 23 );
  kwiver::vital::bounding_box_d bb1( tl, 100, 100 );

  kwiver::vital::detected_object doby( bb1 ); // start with defaults

  doby.set_confidence( .5546 );

  TEST_EQUAL( "Expected confidence", doby.confidence(), .5546 );

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

  dot = doby.type();

  if ( !dot )
  {
    TEST_ERROR( "detected object type returned incorrectly" );
  }

  kwiver::vital::bounding_box_d::vector_type tr( 20, 10 );
  kwiver::vital::translate( bb1, tr );

  doby.set_bounding_box( bb1 );
}
