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

#include <vital/types/detected_object_set.h>
#include <vital/vital_foreach.h>

#define TEST_ARGS       ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}

namespace {

  kwiver::vital::detected_object_type_sptr create_dot( const char* n[], const double s[] )
  {
    std::vector< std::string > names;
    std::vector< double > scores;

    for ( size_t i = 0; n[i] != 0; ++i )
    {
      names.push_back( std::string( n[i] ) );
      scores.push_back( s[i] );
    } // end for

    return  std::make_shared< kwiver::vital::detected_object_type >( names, scores );
  }

} // end namespace

// ------------------------------------------------------------------
IMPLEMENT_TEST(object_creation)
{
  kwiver::vital::detected_object_set do_set;

  kwiver::vital::bounding_box_d bb1( 10, 20, 30, 40 );

  const char* n[]  = { "person", "vehicle", "other", "clam", "barnacle", 0 };
  double s[] = {   .65,      .6,       .005,    .07,     .005,     0 };

  auto dot = create_dot( n, s );

  auto detection = std::make_shared< kwiver::vital::detected_object >( bb1 ); // using defaults
  do_set.add( detection );

  TEST_EQUAL( "set size one", do_set.size(), 1 );

  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.65, dot  );
  do_set.add( detection );

  double s1[] = {   .0065,      .006,       .005,    .775,     .605,     0 };
  dot = create_dot( n, s1 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.75, dot  );
  do_set.add( detection );

  double s2[] = {   .0065,      .006,       .005,    .605,     .775,     0 };
  dot = create_dot( n, s2 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.78, dot  );
  do_set.add( detection );

  double s3[] = {   .5065,      .006,       .005,    .775,     .605,     0 };
  dot = create_dot( n, s3 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.70, dot  );
  do_set.add( detection );

  // get whole list sorted by confidence
  auto list_1 = do_set.select();

  TEST_EQUAL( "size of list", list_1->size(), 5 );
  auto it = list_1->cbegin();
  TEST_EQUAL( "expected confidence 0", (*it++)->confidence(), 1.0 );
  TEST_EQUAL( "expected confidence 1", (*it++)->confidence(), 0.78 );
  TEST_EQUAL( "expected confidence 2", (*it++)->confidence(), 0.75 );
  TEST_EQUAL( "expected confidence 3", (*it++)->confidence(), 0.70 );
  TEST_EQUAL( "expected confidence 4", (*it++)->confidence(), 0.65 );

  list_1 = do_set.select( 0.75 );

  TEST_EQUAL( "size of list", list_1->size(), 3 );

  it = list_1->cbegin();
  TEST_EQUAL( "expected confidence 0", (*it++)->confidence(), 1.0 );
  TEST_EQUAL( "expected confidence 1", (*it++)->confidence(), 0.78 );
  TEST_EQUAL( "expected confidence 2", (*it++)->confidence(), 0.75 );

  list_1 = do_set.select( "clam" );

  TEST_EQUAL( "size of list", list_1->size(), 4 );

  it = list_1->cbegin();
  dot = (*it++)->type();
  double score = dot->score( "clam" );
  TEST_EQUAL( "expected score 0", score, 0.775 );

  dot = (*it++)->type();
  score = dot->score( "clam" );
  TEST_EQUAL( "expected score 1", score, 0.775 );

  dot = (*it++)->type();
  score = dot->score( "clam" );
  TEST_EQUAL( "expected score 2", score, 0.605 );

  dot = (*it++)->type();
  score = dot->score( "clam" );
  TEST_EQUAL( "expected score 3", score, 0.07 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(set_copy)
{
  kwiver::vital::detected_object_set do_set;

  kwiver::vital::bounding_box_d bb1( 10, 20, 30, 40 );

  const char* n[]  = { "person", "vehicle", "other", "clam", "barnacle", 0 };
  double s[] = {   .65,      .6,       .005,    .07,     .005,     0 };

  auto dot = create_dot( n, s );

  auto detection = std::make_shared< kwiver::vital::detected_object >( bb1 ); // using defaults
  do_set.add( detection );

  TEST_EQUAL( "set size one", do_set.size(), 1 );

  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.65, dot  );
  do_set.add( detection );

  double s1[] = {   .0065,      .006,       .005,    .775,     .605,     0 };
  dot = create_dot( n, s1 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.75, dot  );
  do_set.add( detection );

  double s2[] = {   .0065,      .006,       .005,    .605,     .775,     0 };
  dot = create_dot( n, s2 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.78, dot  );
  do_set.add( detection );

  double s3[] = {   .5065,      .006,       .005,    .775,     .605,     0 };
  dot = create_dot( n, s3 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.70, dot  );
  do_set.add( detection );

  auto do_set_clone = do_set.clone();

  // get whole list sorted by confidence
  auto list_1 = do_set_clone->select();

  TEST_EQUAL( "size of list", list_1->size(), 5 );

  auto it = list_1->cbegin();
  TEST_EQUAL( "expected confidence 0", (*it++)->confidence(), 1.0 );
  TEST_EQUAL( "expected confidence 1", (*it++)->confidence(), 0.78 );
  TEST_EQUAL( "expected confidence 2", (*it++)->confidence(), 0.75 );
  TEST_EQUAL( "expected confidence 3", (*it++)->confidence(), 0.70 );
  TEST_EQUAL( "expected confidence 4", (*it++)->confidence(), 0.65 );

  list_1 = do_set_clone->select( 0.75 );

  TEST_EQUAL( "size of list", list_1->size(), 3 );

  it = list_1->cbegin();
  TEST_EQUAL( "expected confidence 0", (*it++)->confidence(), 1.0 );
  TEST_EQUAL( "expected confidence 1", (*it++)->confidence(), 0.78 );
  TEST_EQUAL( "expected confidence 2", (*it++)->confidence(), 0.75 );

  list_1 =do_set_clone->select( "clam" );

  TEST_EQUAL( "size of list", list_1->size(), 4 );

  it = list_1->cbegin();
  dot = (*it++)->type();
  double score = dot->score( "clam" );
  TEST_EQUAL( "expected score 0", score, 0.775 );

  dot = (*it++)->type();
  score = dot->score( "clam" );
  TEST_EQUAL( "expected score 1", score, 0.775 );

  dot = (*it++)->type();
  score = dot->score( "clam" );
  TEST_EQUAL( "expected score 2", score, 0.605 );

  dot = (*it++)->type();
  score = dot->score( "clam" );
  TEST_EQUAL( "expected score 3", score, 0.07 );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(clone_2)
{
  kwiver::vital::detected_object_set do_set;

  kwiver::vital::bounding_box_d bb1( 10, 20, 30, 40 );

  const char* n[]  = { "person", "vehicle", "other", "clam", "barnacle", 0 };
  double s[] = {   .65,      .6,       .005,    .07,     .005,     0 };

  auto dot = create_dot( n, s );

  auto detection = std::make_shared< kwiver::vital::detected_object >( bb1 ); // using defaults
  do_set.add( detection );

  auto do_set2 =  do_set.clone();
  TEST_EQUAL( "(1) set size one", do_set2->size(), 1 );

  auto attr_set = std::make_shared< kwiver::vital::attribute_set >();
  do_set.set_attributes( attr_set );

  do_set2 = do_set.clone();
  TEST_EQUAL( "(2) set size one", do_set2->size(), 1 );
}
