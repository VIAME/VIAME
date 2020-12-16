// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test detected object io
 */

#include <arrows/core/detected_object_set_input_csv.h>
#include <arrows/core/detected_object_set_output_csv.h>

#include <gtest/gtest.h>

#include <iostream>
#include <sstream>
#include <memory>
#include <string>

namespace algo = kwiver::vital::algo;
namespace kac = kwiver::arrows::core;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

namespace {

// ----------------------------------------------------------------------------
kwiver::vital::detected_object_type_sptr
create_dot( const char* n[], const double s[] )
{
  std::vector< std::string > names;
  std::vector< double > scores;

  for ( size_t i = 0; n[i] != 0; ++i )
  {
    names.push_back( std::string( n[i] ) );
    scores.push_back( s[i] );
  } // end for

  return  std::make_shared< kwiver::vital::detected_object_type >( names, scores );
}

// ----------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
make_dos()
{
  kwiver::vital::detected_object_set_sptr do_set = std::make_shared<kwiver::vital::detected_object_set>();

  kwiver::vital::bounding_box_d bb1( 10, 20, 30, 40 );

  const char* n[]  = { "person", "vehicle", "other", "clam", "barnacle", 0 };
  double s[] = {   .65,      .6,       .005,    .07,     .005,     0 };

  auto dot = create_dot( n, s );

  auto detection = std::make_shared< kwiver::vital::detected_object >( bb1 ); // using defaults
  do_set->add( detection );

  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.65, dot  );
  do_set->add( detection );

  double s1[] = {   .0065,      .006,       .005,    .775,     .605,     0 };
  dot = create_dot( n, s1 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.75, dot  );
  do_set->add( detection );

  double s2[] = {   .0065,      .006,       .005,    .605,     .775,     0 };
  dot = create_dot( n, s2 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.78, dot  );
  do_set->add( detection );

  double s3[] = {   .5065,      .006,       .005,    .775,     .605,     0 };
  dot = create_dot( n, s3 );
  detection = std::make_shared< kwiver::vital::detected_object >( bb1, 0.70, dot  );
  do_set->add( detection );

  return do_set;
}

} // end namespace

// ----------------------------------------------------------------------------
TEST(detected_object_io, stream_io)
{

  kac::detected_object_set_input_csv reader;
  kac::detected_object_set_output_csv writer;

  // Create some detections.
  auto dos = make_dos();

  // setup stream
  std::stringstream str;
  writer.use_stream( &str );

  writer.write_set( dos, "image_name_1" );
  writer.write_set( dos, "image_name_2" );

  writer.close();

  std::cout << str.str() << std::endl;

  reader.use_stream( &str );
  kwiver::vital::detected_object_set_sptr idos;
  std::string name;
  EXPECT_TRUE( reader.read_set( idos, name ) );
  EXPECT_EQ( "image_name_1" , name );

  EXPECT_EQ( false , reader.at_eof() );

  EXPECT_TRUE( reader.read_set( idos, name ) );
  EXPECT_EQ( "image_name_2", name );

  EXPECT_FALSE( reader.read_set( idos, name ) );
  EXPECT_TRUE( reader.at_eof() );
}
