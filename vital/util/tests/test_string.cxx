/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief test util string_editor class
 */
#include <test_common.h>

#include <vital/util/string.h>
#include <sstream>


#define TEST_ARGS ( )

DECLARE_TEST_MAP();

// ------------------------------------------------------------------
int
main( int argc, char* argv[] )
{
  CHECK_ARGS( 1 );

  testname_t const testname = argv[1];

  RUN_TEST( testname );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( test_starts_with )
{
  TEST_EQUAL( kwiver::vital::starts_with( "input_string", "input" ), true, true );

  if ( kwiver::vital::starts_with( "input_string", " input" ) )
  {
    TEST_ERROR( "kwiver::vital::starts_with( \"input_string\", \" input\" ) returned incorrect result."  );
  }

  TEST_EQUAL( kwiver::vital::starts_with( " input_string", " input" ), true, true );

  if( kwiver::vital::starts_with( "input_string", " string" ) )
  {
    TEST_ERROR( "kwiver::vital::starts_with( \"input_string\", \" string\" ) returned incorrect result." );
  }
}



// ------------------------------------------------------------------
IMPLEMENT_TEST( test_string_format )
{

  std::string result = kwiver::vital::string_format( "%d %d", 1, 2);
  std::string expected = "1 2";
  if (result != expected )
  {
    std::stringstream str;
    str << "Result: '" << result << "' not as expected '" << expected;
    TEST_ERROR(str.str());
  }

  result = kwiver::vital::string_format( " %d %d", 1, 2);
  expected = " 1 2";
  if (result != expected )
  {
    std::stringstream str;
    str << "Result: '" << result << "' not as expected '" << expected;
    TEST_ERROR(str.str());
  }

  const std::string long_string( "this is a very long string - relatively speaking" );
  result = kwiver::vital::string_format( "%s", long_string.c_str());
  expected = long_string;
  if (result != expected )
  {
    std::stringstream str;
    str << "Result: '" << result << "' not as expected '" << expected;
    TEST_ERROR(str.str());
  }
}
