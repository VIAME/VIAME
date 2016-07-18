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
 * \brief test util any_converter class
 */
#include <test_common.h>

#include <vital/util/any_converter.h>


#define TEST_ARGS ()

DECLARE_TEST_MAP();

// ------------------------------------------------------------------
int
main(int argc, char* argv[])
{
  CHECK_ARGS(1);

  testname_t const testname = argv[1];

  RUN_TEST(testname);
}


// ------------------------------------------------------------------
IMPLEMENT_TEST(test_converter)
{
  kwiver::vital::any_converter<int> any_to_int;

  any_to_int.add_converter<uint8_t>();  // add converter from uint8_t;
  any_to_int.add_converter<float>();    // add converter from float;

  kwiver::vital::any ui8 = (uint8_t) 123;
  kwiver::vital::any fl = (float) 123.45;
  kwiver::vital::any cp = std::string("string");

  TEST_EQUAL( "Convertable char", any_to_int.can_convert( ui8 ), true );
  TEST_EQUAL( "Convert char to int", any_to_int.convert( ui8 ), 123);

  TEST_EQUAL( "Convertable float", any_to_int.can_convert( fl ), true );
  TEST_EQUAL( "Convert float to int", any_to_int.convert( fl ), 123);

  TEST_EQUAL( "Unconvertable", any_to_int.can_convert( cp ), false );

  EXPECT_EXCEPTION( kwiver::vital::bad_any_cast,
                    any_to_int.convert( cp ),
                    "Converting the unconvertable" );
}
