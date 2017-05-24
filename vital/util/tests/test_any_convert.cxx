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
#include <vital/types/uid.h>

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
IMPLEMENT_TEST(converter)
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


// ==================================================================
// make a custom specialization
namespace kwiver {
namespace vital {
namespace any_convert {

template < >
struct converter< bool, std::string >
  : public convert_base< bool >
{
  converter()
  {
    convert_map.insert( std::pair< std::string, bool > ( "yes", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "YES", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "no", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "NO", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "0", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "zero", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "1", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "one", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "on", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "ON", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "off", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "OFF", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "ja", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "nein", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "up", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "down", false ) );
    convert_map.insert( std::pair< std::string, bool > ( "true", true ) );
    convert_map.insert( std::pair< std::string, bool > ( "false", false ) );
  }


  virtual ~converter() VITAL_DEFAULT_DTOR

  virtual bool can_convert( kwiver::vital::any const & data ) const
  {
    return ( data.type() == typeid( std::string ) ) &&
           convert_map.find( kwiver::vital::any_cast< std::string > ( data ) ) != convert_map.end();
  }


  virtual bool convert( kwiver::vital::any const& data ) const
  {
    auto it = convert_map.find( kwiver::vital::any_cast< std::string > ( data ) );
    if ( it != convert_map.end() )
    {
      return it->second;
    }
    throw kwiver::vital::bad_any_cast( typeid( bool ).name(), typeid( std::string ).name() );
  }


private:
  std::map< std::string, bool > convert_map;

};

} } }     // end namespace

// ------------------------------------------------------------------
IMPLEMENT_TEST( test_custom_converter )
{
  kwiver::vital::any_converter< bool > convert_to_bool;


  convert_to_bool.add_converter< bool > ();      // self type needs to be added too
  convert_to_bool.add_converter< int > ();
  convert_to_bool.add_converter< std::string > ();      // Use custom converter

  std::string value;

  value = "yes";
  TEST_EQUAL( "Convertable string", convert_to_bool.can_convert( value ), true );
  TEST_EQUAL( "Convert string to bool", convert_to_bool.convert( value ), true );

  value = "up";
  TEST_EQUAL( "Convertable string", convert_to_bool.can_convert( value ), true );
  TEST_EQUAL( "Convert string to bool", convert_to_bool.convert( value ), true );

  value = "false";
  TEST_EQUAL( "Convertable string", convert_to_bool.can_convert( value ), true );
  TEST_EQUAL( "Convert string to bool", convert_to_bool.convert( value ), false );

  value = "true";
  TEST_EQUAL( "Convertable string", convert_to_bool.can_convert( value ), true );
  TEST_EQUAL( "Convert string to bool", convert_to_bool.convert( value ), true );

  TEST_EQUAL( "Convertable int", convert_to_bool.can_convert( 10 ), true );
  TEST_EQUAL( "Convert int to bool", convert_to_bool.convert( 10 ), true );

  TEST_EQUAL( "Convertable bool", convert_to_bool.can_convert( true ), true );
  TEST_EQUAL( "Convert bool to bool", convert_to_bool.convert( true ), true );

  TEST_EQUAL( "UnConvertable string", convert_to_bool.can_convert( std::string( "yup" ) ), false );
}

#if 0
//
// Custom converter object
//
struct uuid_converter
  : public kwiver::vital::any_convert::convert_base< std::string >
{
  virtual bool can_convert( kwiver::vital::any const& data ) const
  {
    return data.type() == typeid( kwiver::vital::uuid );
  }

  virtual std::string convert( kwiver::vital::any const& data ) const
  {
    return  kwiver::vital::any_cast< kwiver::vital::uuid > ( data ).format();
  }
};


// ------------------------------------------------------------------
IMPLEMENT_TEST(explicit_specialization)
{
  kwiver::vital::any_converter<std::string> any_to_string;
  kwiver::vital::any_converter<std::string> other_to_string;

  other_to_string.add_converter( new uuid_converter() );
  any_to_string.add_converter<float>();    // add converter from float;
  any_to_string.add_converter<kwiver::vital::uuid>();


  kwiver::vital::uuid::uuid_data_t udata = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 };
  kwiver::vital::uuid uu(udata);
  kwiver::vital::any a_uuid(uu);
  kwiver::vital::any fl = (float) 123.45;

  std::string suu = any_to_string.convert( a_uuid );
  std::cout << "uuid - " << suu << std::endl;

  suu = other_to_string.convert( a_uuid );
  std::cout << "other uuid - " << suu << std::endl;

  suu = any_to_string.convert( fl );
  std::cout << "float - " << suu << std::endl;
}
#endif
