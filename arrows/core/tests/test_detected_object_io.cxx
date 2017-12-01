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
