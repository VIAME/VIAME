/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * \brief test json serializers
 */

#include <gtest/gtest.h>

#include <arrows/serialize/protobuf/bounding_box.h>
#include <arrows/serialize/protobuf/detected_object_type.h>
#include <arrows/serialize/protobuf/detected_object.h>
#include <arrows/serialize/protobuf/detected_object_set.h>
#include <arrows/serialize/protobuf/timestamp.h>
#include <arrows/serialize/protobuf/image.h>
#include <arrows/serialize/protobuf/string.h>
#include <vital/types/bounding_box.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/detected_object.h>
#include <vital/types/timestamp.h>

#include <vital/util/hex_dump.h>
#include <vital/util/string.h>

namespace kasp = kwiver::arrows::serialize::protobuf;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST( serialize, bounding_box )
{
  kasp::bounding_box bbox_ser;
  kwiver::vital::bounding_box_d bbox { 1, 2, 3, 4 };

  kwiver::vital::any bb_any( bbox );
  auto mes = bbox_ser.serialize( bb_any );

  // std::cout << "Serialized bbox: \"" << *mes << "\"\n";

  auto dser = bbox_ser.deserialize( *mes );
  kwiver::vital::bounding_box_d bbox_dser = kwiver::vital::any_cast< kwiver::vital::bounding_box_d >( dser );

  /*
  std::cout << "bbox_dser { " << bbox_dser.min_x() << ", "
            << bbox_dser.min_y() << ", "
            << bbox_dser.max_x() << ", "
            << bbox_dser.max_y() << "}\n";
  */
  EXPECT_EQ( bbox, bbox_dser );
}

// ----------------------------------------------------------------------------
TEST( serialize, detected_object_type )
{
  kasp::detected_object_type dot_ser; // get serializer
  kwiver::vital::detected_object_type dot;

  dot.set_score( "first", 1 );
  dot.set_score( "second", 10 );
  dot.set_score( "third", 101 );
  dot.set_score( "last", 121 );

  kwiver::vital::any dot_any( dot );
  auto mes = dot_ser.serialize( dot_any );

  // std::cout << "Serialized dot: \"" << *mes << "\"\n";

  auto dser = dot_ser.deserialize( *mes );
  kwiver::vital::detected_object_type dot_dser =
    kwiver::vital::any_cast< kwiver::vital::detected_object_type >( dser );

  EXPECT_EQ( dot.size(), dot_dser.size() );

  auto o_it = dot.begin();
  auto d_it = dot_dser.begin();

  for (size_t i = 0; i < dot.size(); ++i )
  {
    EXPECT_EQ( *(o_it->first), *(d_it->first) );
    EXPECT_EQ( o_it->second, d_it->second );
  }
}

// ----------------------------------------------------------------------------
TEST( serialize, detected_object )
{
  kasp::detected_object obj_ser; // get serializer

  auto dot = std::make_shared<kwiver::vital::detected_object_type>();

  dot->set_score( "first", 1 );
  dot->set_score( "second", 10 );
  dot->set_score( "third", 101 );
  dot->set_score( "last", 121 );

  auto obj = std::make_shared< kwiver::vital::detected_object>(
    kwiver::vital::bounding_box_d{ 1, 2, 3, 4 }, 3.14159, dot );
  obj->set_detector_name( "test_detector" );
  obj->set_index( 1234 );

  kwiver::vital::any obj_any( *obj );
  auto mes = obj_ser.serialize( obj_any );

  // std::cout << "Serialized dot: \"" << *mes << "\"\n";

  auto dser = obj_ser.deserialize( *mes );
  auto obj_dser = kwiver::vital::any_cast< kwiver::vital::detected_object_sptr >( dser );

  EXPECT_EQ( obj->bounding_box(), obj_dser->bounding_box() );
  EXPECT_EQ( obj->index(), obj_dser->index() );
  EXPECT_EQ( obj->confidence(), obj_dser->confidence() );
  EXPECT_EQ( obj->detector_name(), obj_dser->detector_name() );

  dot = obj->type();
  if (dot)
  {
    auto dot_dser = obj_dser->type();

    EXPECT_EQ( dot->size(), dot_dser->size() );

    auto o_it = dot->begin();
    auto d_it = dot_dser->begin();

    for (size_t i = 0; i < dot->size(); ++i )
    {
      EXPECT_EQ( *(o_it->first), *(d_it->first) );
      EXPECT_EQ( o_it->second, d_it->second );
    }
  }
}

// ----------------------------------------------------------------------------
TEST( serialize, detected_object_set )
{
  kasp::detected_object_set obj_ser; // get serializer

  auto ser_dos_sptr = std::make_shared< kwiver::vital::detected_object_set >();
  for ( int i=0; i < 10; i++ )
  {
    auto dot_sptr = std::make_shared<kwiver::vital::detected_object_type>();

    dot_sptr->set_score( "first", 1 + i );
    dot_sptr->set_score( "second", 10 + i );
    dot_sptr->set_score( "third", 101 + i );
    dot_sptr->set_score( "last", 121 + i );

    auto det_object_sptr = std::make_shared< kwiver::vital::detected_object>(
      kwiver::vital::bounding_box_d{ 1.0 + i, 2.0 + i, 3.0 + i, 4.0 + i }, 3.14159, dot_sptr );
    det_object_sptr->set_detector_name( "test_detector" );
    det_object_sptr->set_index( 1234 + i);

    ser_dos_sptr->add( det_object_sptr );
  }

  kwiver::vital::any obj_any( ser_dos_sptr );
  auto mes = obj_ser.serialize( obj_any );

  // std::cout << "Serialized dot: \"" << *mes << "\"\n";

  auto dser = obj_ser.deserialize( *mes );
  auto deser_dos_sptr = kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr >( dser );

  for ( int i = 0; i < 10; i++ )
  {
    auto ser_do_sptr = ser_dos_sptr->at(i);
    auto deser_do_sptr = deser_dos_sptr->at(i);

    EXPECT_EQ( ser_do_sptr->bounding_box(), deser_do_sptr->bounding_box() );
    EXPECT_EQ( ser_do_sptr->index(), deser_do_sptr->index() );
    EXPECT_EQ( ser_do_sptr->confidence(), deser_do_sptr->confidence() );
    EXPECT_EQ( ser_do_sptr->detector_name(), deser_do_sptr->detector_name() );

    auto ser_dot_sptr = ser_do_sptr->type();
    auto deser_dot_sptr = deser_do_sptr->type();

    if ( ser_dot_sptr )
    {
      EXPECT_EQ( ser_dot_sptr->size(),deser_dot_sptr->size() );

      auto ser_it = ser_dot_sptr->begin();
      auto deser_it = deser_dot_sptr->begin();

      for ( size_t i = 0; i < ser_dot_sptr->size(); ++i )
      {
        EXPECT_EQ( *(ser_it->first), *(ser_it->first) );
        EXPECT_EQ( deser_it->second, deser_it->second );
      }
    }
  } // end for
}

// ----------------------------------------------------------------------------
TEST (serialize, timestamp)
{
  kasp::timestamp tstamp_ser;
  kwiver::vital::timestamp tstamp{1, 1};

  kwiver::vital::any tstamp_any(tstamp);

  auto mes = tstamp_ser.serialize( tstamp_any );
  auto dser = tstamp_ser.deserialize( *mes );

  kwiver::vital::timestamp tstamp_dser =
    kwiver::vital::any_cast< kwiver::vital::timestamp > ( dser );

  // std::cout << tstamp_dser.pretty_print() << std::endl;

  EXPECT_EQ (tstamp, tstamp_dser);
}

// ----------------------------------------------------------------------------
TEST( serialize, image)
{
  kasp::image image_ser;
  kwiver::vital::image img{200, 300, 3};

  char* cp = static_cast< char* >(img.memory()->data() );
  for ( size_t i = 0; i < img.size(); ++i )
  {
    *cp++ = i;
  }

  kwiver::vital::image_container_sptr img_container =
    std::make_shared< kwiver::vital::simple_image_container >( img );
  kwiver::vital::any img_any(img_container);

  auto mes = image_ser.serialize( img_any );
  auto dser = image_ser.deserialize( *mes );

  auto img_dser = kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( dser );

  // Check the content of images
  EXPECT_TRUE ( kwiver::vital::equal_content( img_container->get_image(), img_dser->get_image()) );
}

// ----------------------------------------------------------------------------
TEST (serialize, string)
{
  kasp::string str_ser;
  std::string str("Test string");

  kwiver::vital::any str_any(str);

  auto mes = str_ser.serialize( str_any );
  auto dser = str_ser.deserialize( *mes );

  std::string str_dser =
    kwiver::vital::any_cast< std::string > ( dser );

  // std::cout << tstamp_dser.pretty_print() << std::endl;

  EXPECT_EQ (str, str_dser);
}
