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

#include <vital/util/token_expander.h>
#include <vital/util/token_type_symtab.h>
#include <vital/util/token_type_sysenv.h>
#include <vital/util/token_type_env.h>

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
IMPLEMENT_TEST( test_expander )
{
  kwiver::vital::token_expander expander;

  expander.add_token_type( new kwiver::vital::token_type_env() );
  expander.add_token_type( new kwiver::vital::token_type_sysenv() );

  kwiver::vital::token_type_symtab* symtab = new kwiver::vital::token_type_symtab("LOCAL");
  expander.add_token_type( symtab );

  symtab->add_entry( "key", "value" );
  symtab->add_entry( "key2", "value2" );

  std::string input = "no-subs here";

  TEST_EQUAL( "No substitution", expander.expand_token( input ), input );

  input = "$LOCAL{key}";
  TEST_EQUAL( "Full substitution", expander.expand_token( input ), "value" );

  input = "prefix$LOCAL{key}";
  TEST_EQUAL( "Substitution prefix", expander.expand_token( input ), "prefixvalue" );

  input = "$LOCAL{key}postfix";
  TEST_EQUAL( "Substitution postfix", expander.expand_token( input ), "valuepostfix" );

  input = "pre$LOCAL{key}post";
  TEST_EQUAL( "Substitution bookends", expander.expand_token( input ), "prevaluepost" );

  input = "$LOCAL{key}$LOCAL{key2}";
  TEST_EQUAL( "Substitution 2 times", expander.expand_token( input ), "valuevalue2" );
}


// ------------------------------------------------------------------
IMPLEMENT_TEST( test_expander_missing_default )
{
  kwiver::vital::token_expander expander;

  expander.add_token_type( new kwiver::vital::token_type_env() );
  expander.add_token_type( new kwiver::vital::token_type_sysenv() );

  kwiver::vital::token_type_symtab* symtab = new kwiver::vital::token_type_symtab("LOCAL");
  expander.add_token_type( symtab );

  symtab->add_entry( "key", "value" );
  symtab->add_entry( "key2", "value2" );

  std::string input = "$LOCAL{not-here}";
  TEST_EQUAL( "Entry not here", expander.expand_token( input ), input );

  input = "$FOO{not-here}";
  TEST_EQUAL( "Provider not here", expander.expand_token( input ), input );
}


// ==================================================================
class expander_no_fill
  : public kwiver::vital::token_expander
{
public:
  expander_no_fill()
    : kwiver::vital::token_expander()
  { }

protected:
  virtual bool handle_missing_entry( const std::string& provider, const std::string& entry )
  { return false; }
  virtual bool handle_missing_provider( const std::string& provider, const std::string& entry )
  { return false; }

};


// ------------------------------------------------------------------
IMPLEMENT_TEST( test_expander_missing_override )
{
  expander_no_fill expander;

  expander.add_token_type( new kwiver::vital::token_type_env() );
  expander.add_token_type( new kwiver::vital::token_type_sysenv() );

  kwiver::vital::token_type_symtab* symtab = new kwiver::vital::token_type_symtab("LOCAL");
  expander.add_token_type( symtab );

  symtab->add_entry( "key", "value" );
  symtab->add_entry( "key2", "value2" );

  std::string input = "A$LOCAL{not_here}B";
  std::string exp_value = expander.expand_token( input );
  TEST_EQUAL( "Entry not here", exp_value, "AB" );

  input = "A$FOO{not_here}B";
  exp_value = expander.expand_token( input );
  TEST_EQUAL( "Provider not here", exp_value, "AB" );
}
