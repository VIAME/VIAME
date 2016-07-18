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
 * \brief test detected_object_type class
 */

#include <test_common.h>

#include <vital/types/detected_object_type.h>
#include <stdexcept>

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
IMPLEMENT_TEST(DOT_creation)
{
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

  kwiver::vital::detected_object_type dot( names, score );

  TEST_EQUAL( "expected score", dot.score( "other" ), 0.07 );

  std::string ml_name;
  double ml_score;
  dot.get_most_likely( ml_name, ml_score );

  TEST_EQUAL( "expected most likely name", ml_name, "person" );
  TEST_EQUAL( "expected most likely score", ml_score, .65 );

  for (size_t i = 0; i < names.size(); i++)
  {
    double s = dot.score( names[i] );
    std::string msg;
    msg = "Expected score for " + names[i];
    TEST_EQUAL( msg, s, score[i] );
  }

  double old_cs = dot.score( "clam" );
  dot.set_score( "clam", 1.23 );
  double new_cs = dot.score( "clam" );

  if ( old_cs != 0.0055 || new_cs != 1.23 )
  {
    TEST_ERROR( "failure setting new score on old class." );
  }

  auto lbl  = kwiver::vital::detected_object_type::all_class_names();
  TEST_EQUAL( "expected name count", lbl.size(), 5 );

  dot.score( "other" ); // make sure this entry exists

  dot.delete_score( "other" );

  EXPECT_EXCEPTION( std::runtime_error,
                    dot.score("other"),
                    "accessing deleted class name" );

  lbl  = kwiver::vital::detected_object_type::all_class_names();
  TEST_EQUAL( "expected new name count", lbl.size(), 4 );

  for (size_t i = 0; i < lbl.size(); i++)
  {
    std::cout << " -- " << lbl[i]
              << "    score: "  << dot.score( lbl[i] ) << std::endl;
  }

}


// ------------------------------------------------------------------
IMPLEMENT_TEST(DOT_creation_error)
{
  std::vector< std::string > names;
  std::vector< double > score;

  EXPECT_EXCEPTION( std::invalid_argument,
                    kwiver::vital::detected_object_type dot( names, score ),
                    "empty initialization vectors" );

  names.push_back( "person" );
  names.push_back( "vehicle" );
  names.push_back( "other" );
  names.push_back( "clam" );
  names.push_back( "barnacle" );

  score.push_back( .65 );
  score.push_back( .6 );
  score.push_back( .07 );
  score.push_back( .0055 );

  EXPECT_EXCEPTION( std::invalid_argument,
                    kwiver::vital::detected_object_type dot( names, score ),
                    "invalid initialization of object" );
}


// ----------------------------------------------------------------


// ------------------------------------------------------------------
IMPLEMENT_TEST(DOT_name_pool)
{
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

  kwiver::vital::detected_object_type dot( names, score );

  std::vector< std::string > names_2;
  names_2.push_back( "a-person" );
  names_2.push_back( "a-vehicle" );
  names_2.push_back( "a-other" );
  names_2.push_back( "a-clam" );
  names_2.push_back( "a-barnacle" );

  std::vector< double > score_2;
  score_2.push_back( .65 );
  score_2.push_back( .6 );
  score_2.push_back( .07 );
  score_2.push_back( .0055 );
  score_2.push_back( .005 );

  kwiver::vital::detected_object_type dot_2( names_2, score_2 );

  auto list = kwiver::vital::detected_object_type::all_class_names();

  TEST_EQUAL( "Expected master list size", list.size(), 10 );

  auto it = list.begin();
  auto eit = list.end();
  for ( ; it != eit; ++it )
  {
    std::cout << "  --  " << *it << std::endl;
  }

}
