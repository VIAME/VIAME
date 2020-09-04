/*ckwg +29
 * Copyright 2011-2018 by Kitware, Inc.
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
 * \brief vital config_block tests
 */

#include <vital/config/config_block.h>
#include <vital/util/enum_converter.h>
#include <vital/types/vector.h>

#include <gtest/gtest.h>

#include <vector>
#include <functional>

using namespace kwiver::vital;

namespace {

// Some test keys/values, so we don't have to declare them in every test case
auto const keya = config_block_key_t{ "keya" };
auto const keyb = config_block_key_t{ "keyb" };
auto const keyc = config_block_key_t{ "keyc" };

auto const valuea = config_block_value_t{ "valuea" };
auto const valueb = config_block_value_t{ "valueb" };
auto const valuec = config_block_value_t{ "valuec" };

auto const block1_name = config_block_key_t{ "block1" };
auto const block2_name = config_block_key_t{ "block2" };

}

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(config_block, block_sep_size)
{
  // Quite a few places assume that the block separator size is 1
  EXPECT_EQ( 1, config_block::block_sep().size() );
}

// ----------------------------------------------------------------------------
TEST(config_block, has_value)
{
  auto const config = config_block::empty_config();

  config->set_value( keya, valuea );

  EXPECT_TRUE( config->has_value( keya ) );
  EXPECT_FALSE( config->has_value( keyb ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, get_value)
{
  auto const config = config_block::empty_config();

  config->set_value( keya, valuea );
  EXPECT_EQ( valuea, config->get_value<config_block_value_t>( keya ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, get_value_nested)
{
  auto const config = config_block::empty_config();

  config->set_value( keya + config_block::block_sep() + keyb, valuea );
  EXPECT_EQ(
    valuea,
    config->subblock( keya )->get_value<config_block_value_t>( keyb ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, get_value_no_exist)
{
  auto const config = config_block::empty_config();

  auto const no_such_key = config_block_key_t{ "lalala" };

  EXPECT_THROW(
    config->get_value<config_block_value_t>( no_such_key ),
    no_such_configuration_value_exception );

  EXPECT_EQ(
    valueb,
    config->get_value<config_block_value_t>( no_such_key, valueb ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, get_value_type_mismatch)
{
  auto const config = config_block::empty_config();

  auto const str_value_key = keya;
  auto const str_value = config_block_value_t{ "hello" };
  int const int_value = 100;

  config->set_value( str_value_key, str_value );

  EXPECT_THROW(
    config->get_value<int>( str_value_key ),
    bad_config_block_cast_exception );

  EXPECT_EQ( int_value, config->get_value<int>( str_value_key, int_value ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, value_conversion)
{
  auto const config = config_block::empty_config();
  auto const key = config_block_key_t{ "key" };

  config->set_value( key, 123.456 );
  EXPECT_EQ( 123.456, config->get_value<double>( key ) );

  config->set_value( key, 1234567 );
  EXPECT_EQ( 1234567, config->get_value<int>( key ) );

  /*
  config->set_value( key, vector_2d{ 2.34, 0.0567 } );
  EXPECT_EQ( ( vector_2d{ 2.34, 0.0567 } ),
             config->get_value<vector_2d>( key ) );
  */

  config->set_value( key, "some string" );
  EXPECT_EQ( "some string", config->get_value<std::string>( key ) );

  config_block_value_t value{ "Some value string" };
  config->set_value( key, value );
  EXPECT_EQ( value, config->get_value<config_block_key_t>( key ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, bool_conversion)
{
  auto const config = config_block::empty_config();

  auto const key = config_block_key_t{ "key" };

  config->set_value( key, config_block_value_t{ "true" } );
  EXPECT_EQ( true, config->get_value<bool>( key ) );

  config->set_value(key, config_block_value_t{ "false" });
  EXPECT_EQ( false, config->get_value<bool>( key ) );

  config->set_value(key, config_block_value_t{ "True" });
  EXPECT_EQ( true, config->get_value<bool>( key ) );

  config->set_value(key, config_block_value_t{ "False" });
  EXPECT_EQ( false, config->get_value<bool>( key ) );

  config->set_value( key, config_block_value_t{ "1" } );
  EXPECT_EQ( true, config->get_value<bool>( key ) );

  config->set_value( key, config_block_value_t{ "0" } );
  EXPECT_EQ( false, config->get_value<bool>( key ) );

  config->set_value( key, config_block_value_t{ "yes" } );
  EXPECT_EQ( true, config->get_value<bool>( key ) );

  config->set_value( key, config_block_value_t{ "no" } );
  EXPECT_EQ( false, config->get_value<bool>( key ) );

  config->set_value( key, true );
  EXPECT_EQ( true, config->get_value<bool>( key ) );

  config->set_value( key, false );
  EXPECT_EQ( false, config->get_value<bool>( key ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, unset_value)
{
  auto const config = config_block::empty_config();

  config->set_value( keya, valuea );
  config->set_value( keyb, valueb );

  config->unset_value( keya );

  EXPECT_THROW( config->get_value<config_block_value_t>( keya ),
                no_such_configuration_value_exception );

  // Check that an unrelated value wasn't also unset
  EXPECT_EQ( valueb, config->get_value<config_block_value_t>( keyb ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, available_values)
{
  auto const config = config_block::empty_config();

  config->set_value( keya, valuea );
  config->set_value( keyb, valueb );

  config_block_keys_t keys;

  keys.push_back( keya );
  keys.push_back( keyb );

  EXPECT_EQ( keys.size(), config->available_values().size() );
}

// ----------------------------------------------------------------------------
TEST(config_block, read_only)
{
  auto const config = config_block::empty_config();

  config->set_value( keya, valuea );

  config->mark_read_only( keya );

  EXPECT_THROW( config->set_value( keya, valueb ),
                set_on_read_only_value_exception );

  EXPECT_EQ( valuea, config->get_value<config_block_value_t>( keya ) );

  EXPECT_THROW( config->unset_value( keya ),
                unset_on_read_only_value_exception );

  EXPECT_EQ( valuea, config->get_value<config_block_value_t>( keya ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock)
{
  auto const config = config_block::empty_config();

  config->set_value( block1_name + config_block::block_sep() + keya, valuea );
  config->set_value( block1_name + config_block::block_sep() + keyb, valueb );
  config->set_value( block2_name + config_block::block_sep() + keyc, valuec );

  auto const subblock = config->subblock( block1_name );

  [&]{
    ASSERT_TRUE( subblock->has_value( keya ) );
    EXPECT_EQ( valuea, subblock->get_value<config_block_value_t>( keya ) );
  }();

  [&]{
    ASSERT_TRUE( subblock->has_value( keyb ) );
    EXPECT_EQ( valueb, subblock->get_value<config_block_value_t>( keyb ) );
  }();

  EXPECT_FALSE( subblock->has_value( keyc ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_nested)
{
  auto const config = config_block::empty_config();

  auto const nested_block_name =
    block1_name + config_block::block_sep() + block2_name;

  config->set_value(
    nested_block_name + config_block::block_sep() + keya, valuea );
  config->set_value(
    nested_block_name + config_block::block_sep() + keyb, valueb );

  auto const subblock = config->subblock( nested_block_name );

  [&]{
    ASSERT_TRUE( subblock->has_value( keya ) );
    EXPECT_EQ( valuea, subblock->get_value<config_block_value_t>( keya ) );
  }();

  [&]{
    ASSERT_TRUE( subblock->has_value( keyb ) );
    EXPECT_EQ( valueb, subblock->get_value<config_block_value_t>( keyb ) );
  }();
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_match)
{
  auto const config = config_block::empty_config();

  config->set_value( block1_name, valuea );

  EXPECT_TRUE( config->subblock( block1_name )->available_values().empty() );
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_prefix_match)
{
  auto const config = config_block::empty_config();

  config->set_value( block1_name + keya, valuea );

  EXPECT_TRUE( config->subblock( block1_name )->available_values().empty() );
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_view)
{
  auto const config = config_block::empty_config();

  config->set_value( block1_name + config_block::block_sep() + keya, valuea );
  config->set_value( block1_name + config_block::block_sep() + keyb, valueb );
  config->set_value( block2_name + config_block::block_sep() + keyc, valuec );

  config_block_sptr const subblock = config->subblock_view( block1_name );

  EXPECT_TRUE( subblock->has_value( keya ) );
  EXPECT_TRUE( subblock->has_value( keyb ) );
  EXPECT_FALSE( subblock->has_value( keyc ) );

  config->set_value( block1_name + config_block::block_sep() + keya, valueb );
  EXPECT_EQ( valueb, subblock->get_value<config_block_value_t>( keya ) );

  subblock->set_value( keya, valuea );
  EXPECT_EQ( valuea, config->get_value<config_block_value_t>(
                       block1_name + config_block::block_sep() + keya ) );

  subblock->unset_value( keyb );
  EXPECT_FALSE(
    config->has_value( block1_name + config_block::block_sep() + keyb ) );

  config->set_value( block1_name + config_block::block_sep() + keyc, valuec );

  config_block_keys_t keys;

  keys.push_back( keya );
  keys.push_back( keyc );

  EXPECT_EQ( keys.size(), subblock->available_values().size() );
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_view_nested)
{
  auto const config = config_block::empty_config();

  auto const nested_block_name =
    block1_name + config_block::block_sep() + block2_name;

  config->set_value(
    nested_block_name + config_block::block_sep() + keya, valuea );
  config->set_value(
    nested_block_name + config_block::block_sep() + keyb, valueb );

  auto const subblock = config->subblock_view( nested_block_name );

  [&]{
    ASSERT_TRUE( subblock->has_value( keya ) );
    EXPECT_EQ( valuea, subblock->get_value<config_block_value_t>( keya ) );
  }();

  [&]{
    ASSERT_TRUE( subblock->has_value( keyb ) );
    EXPECT_EQ( valueb, subblock->get_value<config_block_value_t>( keyb ) );
  }();
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_view_match)
{
  auto const config = config_block::empty_config();

  config->set_value( block1_name, valuea );

  EXPECT_TRUE(
    config->subblock_view( block1_name )->available_values().empty() );
}

// ----------------------------------------------------------------------------
TEST(config_block, subblock_view_prefix_match)
{
  auto const config = config_block::empty_config();

  config->set_value( block1_name + keya, valuea );

  EXPECT_TRUE(
    config->subblock_view( block1_name )->available_values().empty() );
}

// ----------------------------------------------------------------------------
TEST(config_block, merge_config)
{
  auto const configa = config_block::empty_config();
  auto const configb = config_block::empty_config();

  auto configa_sp = std::make_shared<std::string>( "configa");
  auto configb_sp = std::make_shared<std::string>( "configb");

  configa->set_value( keya, valuea );
  configa->set_location( keya, configa_sp, 1 );

  configa->set_value( keyb, valuea );
  configa->set_location( keyb, configa_sp, 2 );

  configb->set_value( keyb, valueb );
  configb->set_location( keyb, configb_sp, 3 );

  configb->set_value( keyc, valuec );
  configb->set_location( keyc, configb_sp, 4 );

  configa->merge_config( configb );

  std::string file;
  int line;

  EXPECT_EQ( valuea, configa->get_value<config_block_value_t>( keya ) );
  configa->get_location( keya, file, line );
  EXPECT_EQ( "configa", file );

  EXPECT_EQ( valueb, configa->get_value<config_block_value_t>( keyb ) );
  configa->get_location( keyb, file, line );
  EXPECT_EQ( "configb", file );

  EXPECT_EQ( valuec, configa->get_value<config_block_value_t>( keyc ) );
  configa->get_location( keyc, file, line );
  EXPECT_EQ( "configb", file );
}

// ----------------------------------------------------------------------------
TEST(config_block, difference_config)
{
  auto const configa = config_block::empty_config();
  auto const configb = config_block::empty_config();

  auto filename = std::make_shared<std::string>( __FILE__ );
  configa->set_value( keya, valuea );
  configa->set_location( keya, filename, __LINE__ );

  configa->set_value( keyb, valueb );
  configa->set_location( keyb, filename, __LINE__ );

  configb->set_value( keyb, valueb );
  configb->set_location( keyb, filename, __LINE__ );

  configb->set_value( keyc, valuec );
  configb->set_location( keyc, filename, __LINE__ );

  auto configa_diffb = configa->difference_config(configb);

  // should be (a - b) => keya
  EXPECT_TRUE( configa_diffb->has_value( keya ) );
  EXPECT_FALSE( configa_diffb->has_value( keyb ) );

  auto configb_diffa = configb->difference_config(configa);

  // should be (b - a) => keyc
  EXPECT_TRUE( configb_diffa->has_value( keyc ) );
  EXPECT_FALSE( configb_diffa->has_value( keya ) );
}

// ----------------------------------------------------------------------------
TEST(config_block, set_value_description)
{
  auto const config = config_block::empty_config();

  auto const keyx = block1_name + config_block::block_sep() + keyb;

  auto const descra = config_block_description_t{ "This is config value A" };
  auto const descrx = config_block_description_t{ "This is config value X" };

  config->set_value( keya, valuea, descra );
  config->set_value( keyx, valueb, descrx );
  config->set_value( keyc, valuec );

  EXPECT_EQ( descra, config->get_description( keya ) );
  EXPECT_EQ( config_block_description_t{}, config->get_description( keyc ) );

  config_block_sptr subblock = config->subblock_view( block1_name );

  EXPECT_EQ( descrx, subblock->get_description( keyb ) );

  EXPECT_THROW(
    config->get_description( config_block_key_t{ "not_a_key" } ),
    no_such_configuration_value_exception );

  config->unset_value( keya );

  EXPECT_THROW(
    config->get_description( keya ),
    no_such_configuration_value_exception );
}

// ----------------------------------------------------------------------------
// Test macro
ENUM_CONVERTER( my_ec, int,
          // init stuff
          { "one",   1 },
          { "two",   2 },
          { "three", 3 },
          { "four",  4 },
          { "five",  5 }
  )

// ----------------------------------------------------------------------------
TEST(config_block, enum_conversion)
{
  auto const config = config_block::empty_config();

  config->set_value( keya, "three" );
  config->set_value( keyb, "foo" );

  EXPECT_EQ( 3, config->get_enum_value < my_ec >( keya ) );

  EXPECT_THROW(
    config->get_enum_value < my_ec >( keyb ),
    std::runtime_error );

  EXPECT_THROW(
    config->get_enum_value< my_ec >( config_block_key_t{ "not_a_key" } ),
    no_such_configuration_value_exception );

  EXPECT_EQ(3, config->get_enum_value < my_ec >(keya, 2));

  EXPECT_EQ(2, config->get_enum_value < my_ec >(keyb, 2));
}


// ------------------------------------------------------------------
TEST(config_block, as_vector)
{
  config_block_sptr const config = config_block::empty_config();

  config->set_value( keya, "0.0 1.0 2.0 3.0" );
  config->set_value( keyb, "1, 2, 3, 4, 5, 6" );

  [&]{
    auto const& values = config->get_value_as_vector<double>( keya );

    ASSERT_EQ( 4, values.size() );

    EXPECT_EQ( 0.0, values[0] );
    EXPECT_EQ( 1.0, values[1] );
    EXPECT_EQ( 2.0, values[2] );
    EXPECT_EQ( 3.0, values[3] );
  }();

  auto const& values = config->get_value_as_vector<double>( keyb, ", " );

  EXPECT_EQ( 6, values.size() );

  for ( size_t i = 0; i < values.size(); ++i )
  {
    SCOPED_TRACE( "At " + std::to_string( i ) );
    EXPECT_EQ( i + 1, values[i] );
  }
}
