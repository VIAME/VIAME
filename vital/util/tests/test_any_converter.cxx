/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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

#include <vital/util/any_converter.h>
#include <vital/types/uid.h>

#include <unordered_map>

#include <gtest/gtest.h>

using namespace kwiver::vital;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST(any_converter, conversions)
{
  any_converter<int> any_to_int;

  any_to_int.add_converter<uint8_t>();  // add converter from uint8_t;
  any_to_int.add_converter<float>();    // add converter from float;

  any ui8 = uint8_t{ 123 };
  any fl = float{ 123.45f };
  any cp = std::string{ "string" };

  EXPECT_TRUE( any_to_int.can_convert( uint8_t{ 123 } ) );
  EXPECT_EQ( 123,  any_to_int.convert( uint8_t{ 123 } ) );

  EXPECT_TRUE( any_to_int.can_convert( float{ 123.45f } ) );
  EXPECT_EQ( 123,  any_to_int.convert( float{ 123.45f } ) );

  EXPECT_FALSE( any_to_int.can_convert( std::string{ "123" } ) );
  EXPECT_THROW( any_to_int.convert( std::string{ "123" } ),
                bad_any_cast );
}

// make a custom specialization
namespace kwiver {
namespace vital {
namespace any_convert {

// ----------------------------------------------------------------------------
template <>
struct converter<bool, std::string>
  : public convert_base<bool>
{
  // --------------------------------------------------------------------------
  converter()
  {
    convert_map.emplace( "yes",   true );
    convert_map.emplace( "YES",   true );
    convert_map.emplace( "no",    false );
    convert_map.emplace( "NO",    false );
    convert_map.emplace( "0",     false );
    convert_map.emplace( "zero",  false );
    convert_map.emplace( "1",     true );
    convert_map.emplace( "one",   true );
    convert_map.emplace( "on",    true );
    convert_map.emplace( "ON",    true );
    convert_map.emplace( "off",   false );
    convert_map.emplace( "OFF",   false );
    convert_map.emplace( "ja",    true );
    convert_map.emplace( "nein",  false );
    convert_map.emplace( "up",    true );
    convert_map.emplace( "down",  false );
    convert_map.emplace( "true",  true );
    convert_map.emplace( "false", false );
  }

  // --------------------------------------------------------------------------
  virtual ~converter() VITAL_DEFAULT_DTOR

  // --------------------------------------------------------------------------
  virtual bool can_convert( any const & data ) const
  {
    return ( data.type() == typeid( std::string ) ) &&
           convert_map.find( any_cast<std::string>( data ) ) != convert_map.end();
  }

  // --------------------------------------------------------------------------
  virtual bool convert( any const& data ) const
  {
    auto const it = convert_map.find( any_cast<std::string>( data ) );
    if ( it != convert_map.end() )
    {
      return it->second;
    }
    throw bad_any_cast( typeid( bool ).name(), typeid( std::string ).name() );
  }

private:
  std::unordered_map<std::string, bool> convert_map;
};

} } }     // end namespace

// ----------------------------------------------------------------------------
TEST(any_converter, custom_converter)
{
  any_converter<bool> convert_to_bool;

  convert_to_bool.add_converter<bool>(); // self type needs to be added too
  convert_to_bool.add_converter<int>();
  convert_to_bool.add_converter<std::string>(); // Use custom converter

  std::string value;

  EXPECT_TRUE(  convert_to_bool.can_convert( std::string{ "yes" } ) );
  EXPECT_EQ( true,  convert_to_bool.convert( std::string{ "yes" } ) );

  EXPECT_TRUE(  convert_to_bool.can_convert( std::string{ "up" } ) );
  EXPECT_EQ( true,  convert_to_bool.convert( std::string{ "up" } ) );

  EXPECT_TRUE(  convert_to_bool.can_convert( std::string{ "false" } ) );
  EXPECT_EQ( false, convert_to_bool.convert( std::string{ "false" } ) );

  EXPECT_TRUE(  convert_to_bool.can_convert( std::string{ "yes" } ) );
  EXPECT_EQ( true,  convert_to_bool.convert( std::string{ "yes" } ) );

  EXPECT_TRUE(  convert_to_bool.can_convert( std::string{ "true" } ) );
  EXPECT_EQ( true,  convert_to_bool.convert( std::string{ "true" } ) );

  EXPECT_TRUE(  convert_to_bool.can_convert( 10 ) );
  EXPECT_EQ( true,  convert_to_bool.convert( 10 ) );

  EXPECT_TRUE(  convert_to_bool.can_convert( true ) );
  EXPECT_EQ( true,  convert_to_bool.convert( true ) );

  EXPECT_FALSE( convert_to_bool.can_convert( std::string{ "foo" } ) );
  EXPECT_THROW( convert_to_bool.convert( std::string{ "foo" } ),
                bad_any_cast );
}
