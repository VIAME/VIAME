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
 * \brief test core any class
 */

#include <test_common.h>
#include <vital/any.h>

#define TEST_ARGS ()

DECLARE_TEST_MAP();

int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


IMPLEMENT_TEST(any_api)
{
  kwiver::vital::any any_one;
  TEST_EQUAL( "Empty any", any_one.empty(), true );

  kwiver::vital::any any_string( std::string ("this is a string") );
  TEST_EQUAL( "Not empty any", any_string.empty(), false );

  any_string.clear();
  TEST_EQUAL( "Cleared ant is empty any", any_one.empty(), true );

  TEST_EQUAL( "Type of cleared any", (any_string.type() == any_one.type()), true );

  kwiver::vital::any any_double(3.14159);
  double dval = kwiver::vital::any_cast<double>( any_double );
  TEST_EQUAL( "Cast to double", dval, 3.14159 );

  EXPECT_EXCEPTION( kwiver::vital::bad_any_cast,
                    std::string sval = kwiver::vital::any_cast<std::string >( any_double ),
                    "converting incompatible types" );

  kwiver::vital::any new_double = any_double;
  TEST_EQUAL( "Type of copied double any", (new_double.type() == any_double.type()), true );

  double* dptr = kwiver::vital::any_cast<double>( &any_double );
  TEST_EQUAL( "Pointer to any", *dptr, 3.14159 );

  // Extract pointer to value in any
  *dptr = 6.28;
  dval = kwiver::vital::any_cast<double>( any_double );
  TEST_EQUAL( "Update pointer to any", *dptr, 6.28 );

  double const* cdptr = kwiver::vital::any_cast<double>( &any_double );
  TEST_EQUAL( "CONST Pointer to any", *cdptr, 6.28 );

  any_double = 3.14159;
  dval = kwiver::vital::any_cast<double>( any_double );
  TEST_EQUAL( "Assign new value to any double", dval, 3.14159 );

  // reset value and type of any
  any_double = 3456; // convert to int
  dval = kwiver::vital::any_cast<int>( any_double );
  TEST_EQUAL( "Assign int value", dval, 3456.0 );
}
