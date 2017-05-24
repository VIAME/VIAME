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
 * \brief test attribute_set functionality
 */

#include <test_common.h>

#include <vital/attribute_set.h>

#define TEST_ARGS      ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST( test_api )
{
  kwiver::vital::attribute_set set;

  TEST_EQUAL( "Size of empty set", set.size(), 0 );
  TEST_EQUAL( "Empty set", set.empty(), true );

  set.add( "first", 1 );

  TEST_EQUAL( "Size of one", set.size(), 1 );

  // replace entry
  set.add( "first", (double) 3.14159 );
  TEST_EQUAL( "Size of one(2)", set.size(), 1 );

  TEST_EQUAL( "has(first) for existing entry", set.has( "first" ), true );
  TEST_EQUAL( "has() for non-existing entry", set.has( "the-first" ), false );

  set.add( "second", 2 );
  TEST_EQUAL( "Size after add another", set.size(), 2 );
  TEST_EQUAL( "has(second) for existing entry", set.has( "second" ), true );
  TEST_EQUAL( "Non-empty set", set.empty(), false );

  int sec = set.get<int>( "second" );

  TEST_EQUAL( "Returned data value", sec, 2 );

  TEST_EQUAL( "Item type test", set.is_type<int>("second"), true );

  // Test returning wrong type, exception
  EXPECT_EXCEPTION(
    kwiver::vital::bad_any_cast,
    double val = set.get<double>( "second" );
    (void) val;
    , "Cast to wrong data type" );

  // test data() accessor
  EXPECT_EXCEPTION(
    kwiver::vital::attribute_set_exception,
    auto set_data = set.data( "second_one" );
    , "Accessing data() for non-existent type" );

  // test iterators
  auto ie = set.end();
  for (auto it = set.begin(); it != ie; ++it )
  {
    std::cout << it->first << std::endl;
  }
}
