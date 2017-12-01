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
#include <vital/util/token_expander.h>
#include <vital/util/token_type_symtab.h>
#include <vital/util/token_type_sysenv.h>
#include <vital/util/token_type_env.h>

#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {

// ----------------------------------------------------------------------------
void initialize( token_expander& expander )
{
  auto* symtab = new token_type_symtab{ "LOCAL" };
  expander.add_token_type( symtab );

  symtab->add_entry( "key", "value" );
  symtab->add_entry( "key2", "value2" );
}

// ----------------------------------------------------------------------------
class expander_no_fill : public token_expander
{
protected:
  virtual bool handle_missing_entry(
    const std::string& provider, const std::string& entry ) override
  { std::cerr << provider << ':' << entry << std::endl; return false; }

  virtual bool handle_missing_provider(
    const std::string& provider, const std::string& entry ) override
  { std::cerr << provider << ':' << entry << std::endl; return false; }
};

}

// ----------------------------------------------------------------------------
TEST(token_expander, basic)
{
  token_expander expander;
  initialize( expander );

  EXPECT_EQ(
    "no-subs here",
    expander.expand_token( "no-subs here" ) );

  EXPECT_EQ(
    "value",
    expander.expand_token( "$LOCAL{key}" ) );

  EXPECT_EQ(
    "prefixvalue",
    expander.expand_token( "prefix$LOCAL{key}" ) );

  EXPECT_EQ(
    "valuesuffix",
    expander.expand_token( "$LOCAL{key}suffix" ) );

  EXPECT_EQ(
    "prefixvaluesuffix",
    expander.expand_token( "prefix$LOCAL{key}suffix" ) );

  EXPECT_EQ(
    "valuevalue2",
    expander.expand_token( "$LOCAL{key}$LOCAL{key2}" ) );
}


// ----------------------------------------------------------------------------
TEST(token_expander, missing_default)
{
  token_expander expander;
  initialize( expander );

  EXPECT_EQ(
    "$LOCAL{bad_key}",
    expander.expand_token( "$LOCAL{bad_key}" ) );

  EXPECT_EQ(
    "$BAD_PROVIDER{key}",
    expander.expand_token( "$BAD_PROVIDER{key}" ) );
}

// ----------------------------------------------------------------------------
TEST(token_expander, missing_override)
{
  expander_no_fill expander;
  initialize( expander );

  EXPECT_EQ(
    "AB",
    expander.expand_token( "A$LOCAL{bad_key}B" ) );

  EXPECT_EQ(
    "AB",
    expander.expand_token( "A$BAD_PROVIDER{key}B" ) );
}

// TODO:
// - Test ENV expander
// - Test SYSENV expander?
// - Test keys with special characters (':', '.')
// - Test malformatted keys, providers
