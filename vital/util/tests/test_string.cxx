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
  TEST_EQUAL( "prefix match", kwiver::vital::starts_with( "input_string", "input" ), true );
  TEST_EQUAL( "leading space", kwiver::vital::starts_with( "input_string", " input" ), false );
  TEST_EQUAL( "leading space match", kwiver::vital::starts_with( " input_string", " input" ), true);
  TEST_EQUAL( "mismatch", kwiver::vital::starts_with( "input_string", " string" ), false );
}



// ------------------------------------------------------------------
IMPLEMENT_TEST( test_string_format )
{
  TEST_EQUAL( "Numeric values", kwiver::vital::string_format( "%d %d", 1, 2), "1 2" );
  TEST_EQUAL( "Leading space", kwiver::vital::string_format( " %d %d", 1, 2), " 1 2" );

  const std::string long_string( "this is a very long string - relatively speaking" );
  TEST_EQUAL( "result longer than format string", kwiver::vital::string_format( "%s", long_string.c_str()), long_string );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( test_string_join )
{
  std::vector<std::string> input;

  TEST_EQUAL( "Empty vector", kwiver::vital::join( input, ", " ), "" );

  input.push_back( "one" );

  TEST_EQUAL( "One element vector", kwiver::vital::join( input, std::string(", ") ), "one" );

  input.push_back( "two" );
  input.push_back( "three" );

  TEST_EQUAL( "Three element vector", kwiver::vital::join( input, std::string(", ") ), "one, two, three" );
}
