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

#include "../load_save.h"

#include <gtest/gtest.h>

#include <vital/types/metadata.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/metadata_tags.h>
#include <vital/types/polygon.h>
#include <vital/types/timestamp.h>
#include <vital/types/geodesy.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>
#include <iostream>

namespace kasj = kwiver::arrows::serialize::json;

#define DEBUG 1

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// ----------------------------------------------------------------------------
TEST( load_save, bounding_box )
{
  kwiver::vital::bounding_box_d obj { 1, 2, 3, 4 };

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    kasj::save( ar, obj );
  }

#if DEBUG
  std::cout << "bbox as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::bounding_box_d obj_dser { 0, 0, 0, 0 };
  {
    cereal::JSONInputArchive ar( msg );
    kasj::load( ar, obj_dser );
  }

  EXPECT_EQ( obj, obj_dser );
}

// ----------------------------------------------------------------------------
TEST( load_save, polygon )
{
  kwiver::vital::polygon obj;
  obj.push_back( 100, 100 );
  obj.push_back( 400, 100 );
  obj.push_back( 400, 400 );
  obj.push_back( 100, 400 );

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    kasj::save( ar, obj );
  }

#if DEBUG
  std::cout << "polygon as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::polygon obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    kasj::load( ar, obj_dser );
  }

  EXPECT_EQ( obj.num_vertices(), obj_dser.num_vertices() );
  EXPECT_EQ( obj.at(0), obj_dser.at(0) );
  EXPECT_EQ( obj.at(1), obj_dser.at(1) );
  EXPECT_EQ( obj.at(2), obj_dser.at(2) );
  EXPECT_EQ( obj.at(3), obj_dser.at(3) );
}

// ----------------------------------------------------------------------------
TEST( load_save, geo_point )
{
  kwiver::vital::geo_point obj( {42.50, 73.54}, kwiver::vital::SRID::lat_lon_WGS84 );

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    kasj::save( ar, obj );
  }

#if DEBUG
  std::cout << "geo_point as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::geo_point obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    kasj::load( ar, obj_dser );
  }

  EXPECT_EQ( obj.location(), obj_dser.location() );
}

// ----------------------------------------------------------------------------
TEST( load_save, geo_polygon )
{
  kwiver::vital::polygon raw_obj;
  raw_obj.push_back( 100, 100 );
  raw_obj.push_back( 400, 100 );
  raw_obj.push_back( 400, 400 );
  raw_obj.push_back( 100, 400 );

  kwiver::vital::geo_polygon obj( raw_obj, kwiver::vital::SRID::lat_lon_WGS84 );

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    kasj::save( ar, obj );
  }

#if DEBUG
  std::cout << "geo_polygon as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::geo_polygon obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    kasj::load( ar, obj_dser );
  }

  kwiver::vital::polygon dser_raw_obj = obj_dser.polygon();

  EXPECT_EQ( raw_obj.num_vertices(), dser_raw_obj.num_vertices() );
  EXPECT_EQ( raw_obj.at(0), dser_raw_obj.at(0) );
  EXPECT_EQ( raw_obj.at(1), dser_raw_obj.at(1) );
  EXPECT_EQ( raw_obj.at(2), dser_raw_obj.at(2) );
  EXPECT_EQ( raw_obj.at(3), dser_raw_obj.at(3) );
  EXPECT_EQ( obj_dser.crs(), kwiver::vital::SRID::lat_lon_WGS84 );
}

// ----------------------------------------------------------------------------
TEST( load_save, metadata )
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
    kwiver::vital::geo_point pt ( { 42.50, 73.54 }, kwiver::vital::SRID::lat_lon_WGS84 );
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

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    kasj::save( ar, meta);
  }

#if DEBUG
  std::cout << "geo_point as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::metadata obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    kasj::load( ar, obj_dser );
  }


}
