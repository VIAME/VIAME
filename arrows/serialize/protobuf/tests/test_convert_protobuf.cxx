/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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

#include <gtest/gtest.h>

#include <arrows/serialize/protobuf/convert_protobuf.h>
#include <arrows/serialize/protobuf/convert_protobuf_point.h>

#include <vital/types/bounding_box.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/geodesy.h>
#include <vital/types/image_container.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_tags.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/object_track_set.h>
#include <vital/types/polygon.h>
#include <vital/types/timestamp.h>
#include <vital/types/track.h>
#include <vital/types/track_set.h>
#include <vital/vital_types.h>

#include <vital/types/protobuf/bounding_box.pb.h>
#include <vital/types/protobuf/detected_object.pb.h>
#include <vital/types/protobuf/detected_object_set.pb.h>
#include <vital/types/protobuf/detected_object_type.pb.h>
#include <vital/types/protobuf/geo_point.pb.h>
#include <vital/types/protobuf/geo_polygon.pb.h>
#include <vital/types/protobuf/image.pb.h>
#include <vital/types/protobuf/metadata.pb.h>
#include <vital/types/protobuf/metadata.pb.h>
#include <vital/types/protobuf/object_track_set.pb.h>
#include <vital/types/protobuf/object_track_state.pb.h>
#include <vital/types/protobuf/polygon.pb.h>
#include <vital/types/protobuf/string.pb.h>
#include <vital/types/protobuf/timestamp.pb.h>
#include <vital/types/protobuf/track.pb.h>
#include <vital/types/protobuf/track_set.pb.h>
#include <vital/types/protobuf/track_state.pb.h>

#include <sstream>
#include <iostream>

namespace kasp = kwiver::arrows::serialize::protobuf;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, bounding_box )
{
  kwiver::vital::bounding_box_d bbox { 1, 2, 3, 4 };

  kwiver::protobuf::bounding_box bbox_proto;
  kwiver::vital::bounding_box_d bbox_dser { 11, 12, 13, 14 };

  kasp::convert_protobuf( bbox, bbox_proto );
  kasp::convert_protobuf( bbox_proto, bbox_dser );

  EXPECT_EQ( bbox, bbox_dser );
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, detected_object_type )
{
  kwiver::vital::detected_object_type dot;

  dot.set_score( "first", 1 );
  dot.set_score( "second", 10 );
  dot.set_score( "third", 101 );
  dot.set_score( "last", 121 );

  kwiver::protobuf::detected_object_type dot_proto;
  kwiver::vital::detected_object_type dot_dser;

  kasp::convert_protobuf( dot, dot_proto );
  kasp::convert_protobuf( dot_proto, dot_dser );

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
TEST( convert_protobuf, detected_object )
{
  auto dot = std::make_shared<kwiver::vital::detected_object_type>();

  dot->set_score( "first", 1 );
  dot->set_score( "second", 10 );
  dot->set_score( "third", 101 );
  dot->set_score( "last", 121 );

  kwiver::vital::detected_object dobj( kwiver::vital::bounding_box_d{ 1, 2, 3, 4 }, 3.14159, dot );
  dobj.set_detector_name( "test_detector" );
  dobj.set_index( 1234 );

  kwiver::protobuf::detected_object dobj_proto;
  kwiver::vital::detected_object dobj_dser( kwiver::vital::bounding_box_d{ 11, 12, 13, 14 }, 13.14159, dot );

  kasp::convert_protobuf( dobj, dobj_proto );
  kasp::convert_protobuf( dobj_proto, dobj_dser );

  EXPECT_EQ( dobj.bounding_box(), dobj_dser.bounding_box() );
  EXPECT_EQ( dobj.index(), dobj_dser.index() );
  EXPECT_EQ( dobj.confidence(), dobj_dser.confidence() );
  EXPECT_EQ( dobj.detector_name(), dobj_dser.detector_name() );

  dot = dobj.type();
  if (dot)
  {
    auto dot_dser = dobj_dser.type();

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

// ---------------------------------------------------------------------------
TEST( convert_protobuf, detected_object_set )
{
  kwiver::vital::detected_object_set dos;
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

    dos.add( det_object_sptr );
  }

  kwiver::protobuf::detected_object_set dos_proto;
  kwiver::vital::detected_object_set dos_dser;

  kasp::convert_protobuf( dos, dos_proto );
  kasp::convert_protobuf( dos_proto, dos_dser );

  for ( int i = 0; i < 10; i++ )
  {
    auto ser_do_sptr = dos.at(i);
    auto dser_do_sptr = dos_dser.at(i);

    EXPECT_EQ( ser_do_sptr->bounding_box(), dser_do_sptr->bounding_box() );
    EXPECT_EQ( ser_do_sptr->index(), dser_do_sptr->index() );
    EXPECT_EQ( ser_do_sptr->confidence(), dser_do_sptr->confidence() );
    EXPECT_EQ( ser_do_sptr->detector_name(), dser_do_sptr->detector_name() );

    auto ser_dot_sptr = ser_do_sptr->type();
    auto dser_dot_sptr = dser_do_sptr->type();

    if ( ser_dot_sptr )
    {
      EXPECT_EQ( ser_dot_sptr->size(),dser_dot_sptr->size() );

      auto ser_it = ser_dot_sptr->begin();
      auto dser_it = dser_dot_sptr->begin();

      for ( size_t i = 0; i < ser_dot_sptr->size(); ++i )
      {
        EXPECT_EQ( *(ser_it->first), *(ser_it->first) );
        EXPECT_EQ( dser_it->second, dser_it->second );
      }
    }
  } // end for
}

// ----------------------------------------------------------------------------
TEST (convert_protobuf, timestamp)
{
  kwiver::vital::timestamp tstamp{1, 1};

  kwiver::protobuf::timestamp ts_proto;
  kwiver::vital::timestamp ts_dser;

  kasp::convert_protobuf( tstamp, ts_proto );
  kasp::convert_protobuf( ts_proto, ts_dser );

  EXPECT_EQ (tstamp, ts_dser);
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, image)
{
  kwiver::vital::image img{200, 300, 3};

  char* cp = static_cast< char* >(img.memory()->data() );
  for ( size_t i = 0; i < img.size(); ++i )
  {
    *cp++ = i;
  }

  kwiver::vital::image_container_sptr img_container =
    std::make_shared< kwiver::vital::simple_image_container >( img );

  kwiver::protobuf::image image_proto;
  kwiver::vital::image_container_sptr img_dser;

  kasp::convert_protobuf( img_container, image_proto );
  kasp::convert_protobuf( image_proto, img_dser );

  // Check the content of images
  EXPECT_TRUE ( kwiver::vital::equal_content( img_container->get_image(), img_dser->get_image()) );
}

// ----------------------------------------------------------------------------
TEST (convert_protobuf, string)
{
  std::string str("Test string");

  kwiver::protobuf::string str_proto;
  std::string str_dser;

  kasp::convert_protobuf( str, str_proto );
  kasp::convert_protobuf( str_proto, str_dser );

  EXPECT_EQ (str, str_dser);
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, polygon )
{
  kwiver::vital::polygon obj;
  obj.push_back( 100, 100 );
  obj.push_back( 400, 100 );
  obj.push_back( 400, 400 );
  obj.push_back( 100, 400 );

  kwiver::protobuf::polygon obj_proto;
  kwiver::vital::polygon obj_dser;

  kasp::convert_protobuf( obj, obj_proto );
  kasp::convert_protobuf( obj_proto, obj_dser );

  EXPECT_EQ( obj.num_vertices(), obj_dser.num_vertices() );
  EXPECT_EQ( obj.at(0), obj_dser.at(0) );
  EXPECT_EQ( obj.at(1), obj_dser.at(1) );
  EXPECT_EQ( obj.at(2), obj_dser.at(2) );
  EXPECT_EQ( obj.at(3), obj_dser.at(3) );
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, geo_point_2d )
{
  // --- 2d variant ---
  kwiver::vital::geo_point::geo_2d_point_t geo( 42.50, 73.54 );
  kwiver::vital::geo_point obj( geo, kwiver::vital::SRID::lat_lon_WGS84 );

  kwiver::protobuf::geo_point obj_proto;
  kwiver::vital::geo_point::geo_2d_point_t geo_dser( 0, 0 );
  kwiver::vital::geo_point obj_dser( geo_dser, 0 );

  kasp::convert_protobuf( obj, obj_proto );
  kasp::convert_protobuf( obj_proto, obj_dser );

  EXPECT_EQ( obj.location(), obj_dser.location() );
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, geo_point_raw )
{
  // --- 3d variant ---
  kwiver::vital::geo_point::geo_3d_point_t geo( 42.50, 73.54, 16.33 );
  kwiver::vital::geo_point obj( geo, kwiver::vital::SRID::lat_lon_WGS84 );

  kwiver::protobuf::geo_point obj_proto;
  kwiver::vital::geo_point::geo_3d_point_t geo_dser( 0, 0, 0 );
  kwiver::vital::geo_point obj_dser( geo_dser, 0 );

  kasp::convert_protobuf( obj, obj_proto );
  kasp::convert_protobuf( obj_proto, obj_dser );

  EXPECT_EQ( obj.location(), obj_dser.location() );
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, geo_polygon )
{
  kwiver::vital::polygon raw_obj;
  raw_obj.push_back( 100, 100 );
  raw_obj.push_back( 400, 100 );
  raw_obj.push_back( 400, 400 );
  raw_obj.push_back( 100, 400 );

  kwiver::vital::geo_polygon obj( raw_obj, kwiver::vital::SRID::lat_lon_WGS84 );

  kwiver::protobuf::geo_polygon obj_proto;
  kwiver::vital::geo_polygon obj_dser;

  kasp::convert_protobuf( obj, obj_proto );
  kasp::convert_protobuf( obj_proto, obj_dser );

  kwiver::vital::polygon dser_raw_obj = obj_dser.polygon();

  EXPECT_EQ( raw_obj.num_vertices(), dser_raw_obj.num_vertices() );
  EXPECT_EQ( raw_obj.at(0), dser_raw_obj.at(0) );
  EXPECT_EQ( raw_obj.at(1), dser_raw_obj.at(1) );
  EXPECT_EQ( raw_obj.at(2), dser_raw_obj.at(2) );
  EXPECT_EQ( raw_obj.at(3), dser_raw_obj.at(3) );
  EXPECT_EQ( obj_dser.crs(), kwiver::vital::SRID::lat_lon_WGS84 );
}

// ----------------------------------------------------------------------------
TEST( convert_protobuf, metadata )
{
  static kwiver::vital::metadata_traits traits;
  kwiver::vital::metadata meta;

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_METADATA_ORIGIN );
    auto* item = info.create_metadata_item( kwiver::vital::any(std::string ("test-source")) );
    meta.add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_UNIX_TIMESTAMP );
    auto* item = info.create_metadata_item( kwiver::vital::any((uint64_t)12345678) );
    meta.add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_SENSOR_VERTICAL_FOV );
    auto* item = info.create_metadata_item( kwiver::vital::any((double)12345.678) );
    meta.add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_FRAME_CENTER );

    kwiver::vital::geo_point::geo_2d_point_t geo_2d( 42.50, 73.54 );
    kwiver::vital::geo_point pt ( geo_2d, kwiver::vital::SRID::lat_lon_WGS84 );
    auto* item = info.create_metadata_item( kwiver::vital::any(pt) );
    meta.add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_FRAME_CENTER );

    kwiver::vital::geo_point::geo_3d_point_t geo( 42.50, 73.54, 16.33 );
    kwiver::vital::geo_point pt ( geo, kwiver::vital::SRID::lat_lon_WGS84 );
    auto* item = info.create_metadata_item( kwiver::vital::any(pt) );
    meta.add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_CORNER_POINTS );
    kwiver::vital::polygon raw_obj;
    raw_obj.push_back( 100, 100 );
    raw_obj.push_back( 400, 100 );
    raw_obj.push_back( 400, 400 );
    raw_obj.push_back( 100, 400 );

    kwiver::vital::geo_polygon poly( raw_obj, kwiver::vital::SRID::lat_lon_WGS84 );
    auto* item = info.create_metadata_item( kwiver::vital::any(poly) );
    meta.add( item );
  }

  kwiver::protobuf::metadata obj_proto;
  kwiver::vital::metadata meta_dser;

  kasp::convert_protobuf( meta, obj_proto );
  kasp::convert_protobuf( obj_proto, meta_dser );

  //+ test for equality - TBD

}

//----------------------------------------------------------------------------
TEST( convert_protobuf, track_state )
{
  kwiver::vital::track_state trk_state{ 1 };

  kwiver::protobuf::track_state proto_trk_state;
  kwiver::vital::track_state dser_trk_state;

  kasp::convert_protobuf( trk_state, proto_trk_state );
  kasp::convert_protobuf( proto_trk_state, dser_trk_state );

  EXPECT_EQ ( trk_state, dser_trk_state );

}

//----------------------------------------------------------------------------
TEST( convert_protobuf, object_track_state )
{
  auto dot_sptr = std::make_shared< kwiver::vital::detected_object_type >();

  dot_sptr->set_score( "first", 1 );
  dot_sptr->set_score( "second", 10 );
  dot_sptr->set_score( "third", 101 );
  dot_sptr->set_score( "last", 121 );

  auto dobj_sptr = std::make_shared< kwiver::vital::detected_object>(
    kwiver::vital::bounding_box_d{ 1, 2, 3, 4 },
    3.14159265, dot_sptr );

  dobj_sptr->set_detector_name( "test_detector" );
  dobj_sptr->set_index( 1234 );

  kwiver::vital::object_track_state obj_trk_state( 1, 1, dobj_sptr );

  obj_trk_state.set_image_point( ::kwiver::vital::point_2d( 123, 321 ) );
  obj_trk_state.set_track_point( ::kwiver::vital::point_3d( 123, 234, 345 ) );

  kwiver::protobuf::object_track_state proto_obj_trk_state;
  kwiver::vital::object_track_state obj_trk_state_dser;

  // do conversion to and fro
  kasp::convert_protobuf( obj_trk_state, proto_obj_trk_state );
  kasp::convert_protobuf( proto_obj_trk_state, obj_trk_state_dser );

  auto do_sptr = obj_trk_state.detection();
  auto do_sptr_dser = obj_trk_state_dser.detection();

  EXPECT_EQ( do_sptr->bounding_box(), do_sptr_dser->bounding_box() );
  EXPECT_EQ( do_sptr->index(), do_sptr_dser->index() );
  EXPECT_EQ( do_sptr->confidence(), do_sptr_dser->confidence() );
  EXPECT_EQ( do_sptr->detector_name(), do_sptr_dser->detector_name() );

  auto dot_sptr_dser = do_sptr_dser->type();

  if ( dot_sptr )
  {
    EXPECT_EQ( dot_sptr->size(), dot_sptr_dser->size() );

    auto it = dot_sptr->begin();
    auto it_dser = dot_sptr_dser->begin();

    for ( size_t i = 0; i < dot_sptr->size(); ++i )
    {
      EXPECT_EQ( *(it->first), *(it_dser->first) );
      EXPECT_EQ( it->second, it_dser->second );
    }
  }
  EXPECT_EQ(obj_trk_state.time(), obj_trk_state_dser.time());
  EXPECT_EQ(obj_trk_state.frame(), obj_trk_state_dser.frame());
  EXPECT_EQ( obj_trk_state.image_point().value(), obj_trk_state_dser.image_point().value() );
  EXPECT_EQ( obj_trk_state.track_point().value(), obj_trk_state_dser.track_point().value() );
}

// ---------------------------------------------------------------------------
TEST( convert_protobuf, track )
{
  // test track with object track state

  kwiver::protobuf::track proto_obj_trk;
  auto trk_dser=kwiver::vital::track::create();
  auto trk = kwiver::vital::track::create();

  trk->set_id(1);
  for (int i=0; i<10; i++)
  {
    auto dot = std::make_shared<kwiver::vital::detected_object_type>();

    dot->set_score( "first", 1 );
    dot->set_score( "second", 10 );
    dot->set_score( "third", 101 );
    dot->set_score( "last", 121 );

    auto dobj_sptr = std::make_shared< kwiver::vital::detected_object>(
                            kwiver::vital::bounding_box_d{ 1, 2, 3, 4 },
                                3.14159265, dot );
    dobj_sptr->set_detector_name( "test_detector" );
    dobj_sptr->set_index( 1234 );
    auto obj_trk_state_sptr = std::make_shared< kwiver::vital::object_track_state >
                                ( i, i, dobj_sptr );

    bool insert_success = trk->insert( obj_trk_state_sptr );
    if ( !insert_success )
    {
      std::cerr << "Failed to insert object track state" << std::endl;
    }
  }
  // Convert track to protobuf and back
  kasp::convert_protobuf( trk, proto_obj_trk );
  kasp::convert_protobuf( proto_obj_trk, trk_dser );

  // Check track id
  EXPECT_EQ( trk->id(), trk_dser->id() );
  for ( int i=0; i<10; i++ )
  {
    auto trk_state_sptr = *trk->find( i );
    auto dser_trk_state_sptr = *trk_dser->find( i );

    EXPECT_EQ( trk_state_sptr->frame(), dser_trk_state_sptr->frame() );
    auto obj_trk_state_sptr = kwiver::vital::object_track_state::downcast( trk_state_sptr );
    auto dser_obj_trk_state_sptr = kwiver::vital::object_track_state::
                                                    downcast( dser_trk_state_sptr );


    auto ser_do_sptr = obj_trk_state_sptr->detection();
    auto dser_do_sptr = dser_obj_trk_state_sptr->detection();

    EXPECT_EQ( ser_do_sptr->bounding_box(), dser_do_sptr->bounding_box() );
    EXPECT_EQ( ser_do_sptr->index(), dser_do_sptr->index() );
    EXPECT_EQ( ser_do_sptr->confidence(), dser_do_sptr->confidence() );
    EXPECT_EQ( ser_do_sptr->detector_name(), dser_do_sptr->detector_name() );

    auto ser_dot_sptr = ser_do_sptr->type();
    auto dser_dot_sptr = dser_do_sptr->type();

    if ( ser_dot_sptr )
    {
      EXPECT_EQ( ser_dot_sptr->size(),dser_dot_sptr->size() );

      auto ser_it = ser_dot_sptr->begin();
      auto dser_it = dser_dot_sptr->begin();

      for ( size_t i = 0; i < ser_dot_sptr->size(); ++i )
      {
        EXPECT_EQ( *(ser_it->first), *(ser_it->first) );
        EXPECT_EQ( dser_it->second, dser_it->second );
      }
    }
  }

  // test track with track state
  kwiver::protobuf::track proto_trk;
  trk = kwiver::vital::track::create();
  trk_dser = kwiver::vital::track::create();
  trk->set_id( 2 );
  for ( int i=0; i<10; i++ )
  {
    auto trk_state_sptr = std::make_shared< kwiver::vital::track_state>( i );
    bool insert_success = trk->insert( trk_state_sptr );
    if ( !insert_success )
    {
      std::cerr << "Failed to insert track state" << std::endl;
    }
  }

  kasp::convert_protobuf( trk, proto_trk );
  kasp::convert_protobuf( proto_trk, trk_dser );

  EXPECT_EQ( trk->id(), trk_dser->id() );

  for ( int i=0; i<10; i++ )
  {
    auto obj_trk_state_sptr = *trk->find( i );
    auto dser_trk_state_sptr = *trk_dser->find( i );

    EXPECT_EQ( obj_trk_state_sptr->frame(), dser_trk_state_sptr->frame() );
  }

}
// ---------------------------------------------------------------------------
TEST( convert_protobuf, track_set )
{
  kwiver::protobuf::track_set proto_trk_set;
  auto trk_set_sptr = std::make_shared< kwiver::vital::track_set >();
  auto trk_set_sptr_dser = std::make_shared< kwiver::vital::track_set >();
  for ( kwiver::vital::track_id_t trk_id=1; trk_id<5; ++trk_id )
  {
    auto trk = kwiver::vital::track::create();
    trk->set_id( trk_id );

    for ( int i=trk_id*10; i < ( trk_id+1 )*10; i++ )
    {
      auto trk_state_sptr = std::make_shared< kwiver::vital::track_state>( i );
      bool insert_success = trk->insert( trk_state_sptr );
      if ( !insert_success )
      {
        std::cerr << "Failed to insert track state" << std::endl;
      }
    }
    trk_set_sptr->insert(trk);
  }

  kasp::convert_protobuf( trk_set_sptr, proto_trk_set );
  kasp::convert_protobuf( proto_trk_set, trk_set_sptr_dser );

  for ( kwiver::vital::track_id_t trk_id=1; trk_id<5; ++trk_id )
  {
    auto trk = trk_set_sptr->get_track( trk_id );
    auto trk_dser = trk_set_sptr_dser->get_track( trk_id );
    EXPECT_EQ( trk->id(), trk_dser->id() );
    for ( int i=trk_id*10; i < ( trk_id+1 )*10; i++ )
    {
      auto obj_trk_state_sptr = *trk->find( i );
      auto dser_trk_state_sptr = *trk_dser->find( i );

      EXPECT_EQ( obj_trk_state_sptr->frame(), dser_trk_state_sptr->frame() );
    }
  }
}
// ---------------------------------------------------------------------------
TEST( convert_protobuf, object_track_set )
{
  kwiver::protobuf::object_track_set proto_obj_trk_set;
  auto obj_trk_set_sptr = std::make_shared< kwiver::vital::object_track_set >();
  auto obj_trk_set_sptr_dser = std::make_shared< kwiver::vital::object_track_set >();
  for ( kwiver::vital::track_id_t trk_id=1; trk_id<5; ++trk_id )
  {
    auto trk = kwiver::vital::track::create();
    trk->set_id( trk_id );
    for ( int i=trk_id*10; i < ( trk_id+1 )*10; i++ )
    {
      auto dot = std::make_shared<kwiver::vital::detected_object_type>();

      dot->set_score( "first", 1 );
      dot->set_score( "second", 10 );
      dot->set_score( "third", 101 );
      dot->set_score( "last", 121 );

      auto dobj_sptr = std::make_shared< kwiver::vital::detected_object>(
                              kwiver::vital::bounding_box_d{ 1, 2, 3, 4 },
                                  3.14159265, dot );
      dobj_sptr->set_detector_name( "test_detector" );
      dobj_sptr->set_index( 1234 );
      auto obj_trk_state_sptr = std::make_shared< kwiver::vital::object_track_state >
                                  ( i, i, dobj_sptr );

      bool insert_success = trk->insert( obj_trk_state_sptr );
      if ( !insert_success )
      {
        std::cerr << "Failed to insert object track state" << std::endl;
      }
    }
    obj_trk_set_sptr->insert(trk);
  }

  kasp::convert_protobuf( obj_trk_set_sptr, proto_obj_trk_set );
  kasp::convert_protobuf( proto_obj_trk_set, obj_trk_set_sptr_dser );

  for ( kwiver::vital::track_id_t trk_id=1; trk_id<5; ++trk_id )
  {
    auto trk = obj_trk_set_sptr->get_track( trk_id );
    auto trk_dser = obj_trk_set_sptr_dser->get_track( trk_id );
    EXPECT_EQ( trk->id(), trk_dser->id() );
    for ( int i=trk_id*10; i < ( trk_id+1 )*10; i++ )
    {
      auto trk_state_sptr = *trk->find( i );
      auto dser_trk_state_sptr = *trk_dser->find( i );

      EXPECT_EQ( trk_state_sptr->frame(), dser_trk_state_sptr->frame() );
      auto obj_trk_state_sptr = kwiver::vital::object_track_state::downcast( trk_state_sptr );
      auto dser_obj_trk_state_sptr = kwiver::vital::object_track_state::
                                                      downcast( dser_trk_state_sptr );


      auto ser_do_sptr = obj_trk_state_sptr->detection();
      auto dser_do_sptr = dser_obj_trk_state_sptr->detection();

      EXPECT_EQ( ser_do_sptr->bounding_box(), dser_do_sptr->bounding_box() );
      EXPECT_EQ( ser_do_sptr->index(), dser_do_sptr->index() );
      EXPECT_EQ( ser_do_sptr->confidence(), dser_do_sptr->confidence() );
      EXPECT_EQ( ser_do_sptr->detector_name(), dser_do_sptr->detector_name() );

      auto ser_dot_sptr = ser_do_sptr->type();
      auto dser_dot_sptr = dser_do_sptr->type();

      if ( ser_dot_sptr )
      {
        EXPECT_EQ( ser_dot_sptr->size(),dser_dot_sptr->size() );

        auto ser_it = ser_dot_sptr->begin();
        auto dser_it = dser_dot_sptr->begin();

        for ( size_t i = 0; i < ser_dot_sptr->size(); ++i )
        {
          EXPECT_EQ( *(ser_it->first), *(ser_it->first) );
          EXPECT_EQ( dser_it->second, dser_it->second );
        }
      }
    }
  }
}

// ----------------------------------------------------------------------------
TEST (convert_protobuf, covariance)
{

#define TEST_COV(T, ... )                               \
do {                                                    \
  ::kwiver::vital::T::matrix_type val;                  \
  val << __VA_ARGS__;                                   \
  ::kwiver::vital::T obj( val );                        \
  ::kwiver::protobuf::covariance obj_proto;             \
  ::kwiver::vital::T obj_dser;                          \
                                                        \
  kasp::convert_protobuf( obj, obj_proto );             \
  kasp::convert_protobuf( obj_proto, obj_dser );        \
                                                        \
  EXPECT_EQ( obj, obj_dser );                           \
} while (0)

TEST_COV( covariance_2d, 1, 2, 3, 4 );
TEST_COV( covariance_2f, 1, 2, 3, 4 );
TEST_COV( covariance_3d, 1, 2, 3, 4, 5, 6, 7, 8, 9 );
TEST_COV( covariance_3f, 1, 2, 3, 4, 5, 6, 7, 8, 9 );
TEST_COV( covariance_4d, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
TEST_COV( covariance_4f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );

#undef TEST_COV

}

// ----------------------------------------------------------------------------
TEST (convert_protobuf, points)
{
  {
    ::kwiver::vital::point_2i obj( 1, 2 );
    ::kwiver::protobuf::point_i obj_proto;
    ::kwiver::vital::point_2i obj_dser;

    kasp::convert_protobuf( obj, obj_proto );
    kasp::convert_protobuf( obj_proto, obj_dser );

    EXPECT_EQ( obj.value(), obj_dser.value() );
  }


#define TEST_POINT(T, ... )                             \
do {                                                    \
  ::kwiver::vital::T obj( __VA_ARGS__ );                \
  ::kwiver::protobuf::point_d obj_proto;                \
  ::kwiver::vital::T obj_dser;                          \
                                                        \
  kasp::convert_protobuf( obj, obj_proto );             \
  kasp::convert_protobuf( obj_proto, obj_dser );        \
                                                        \
  EXPECT_EQ( obj.value(), obj_dser.value() );           \
} while (0)

  TEST_POINT( point_2d, 1, 2 );
  TEST_POINT( point_2f, 1, 2 );
  TEST_POINT( point_3d, 1, 2, 3 );
  TEST_POINT( point_3f, 1, 2, 3 );
  TEST_POINT( point_4d, 1, 2, 3, 4 );
  TEST_POINT( point_4f, 1, 2, 3, 4 );

#undef TEST_POINT

}
