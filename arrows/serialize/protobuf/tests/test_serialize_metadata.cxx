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
 * \brief test protobuf serializers
 */

#include <gtest/gtest.h>

#include <arrows/serialize/protobuf/metadata.h>
#include <arrows/serialize/protobuf/convert_protobuf.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_tags.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/polygon.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/geodesy.h>


namespace kasp = kwiver::arrows::serialize::protobuf;

// ----------------------------------------------------------------------------
int main(int argc, char** argv)
{
  ::testing::InitGoogleTest( &argc, argv );
  return RUN_ALL_TESTS();
}

// 1) Test single metadata collection in vector
// 2) Test two different metadata collections in vector
// 3) Build metadata collection with one element of each supported type.

// ----------------------------------------------------------------------------
TEST( serialize_metadata, metadata )
{
  // Create a metadata collection
  // duplicate that collection and make some modifications
  // put both collections in a vector

  static kwiver::vital::metadata_traits traits;
  auto meta_sptr = std::make_shared< kwiver::vital::metadata>();

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_METADATA_ORIGIN );
    auto* item = info.create_metadata_item( kwiver::vital::any(std::string ("test-source")) );
    meta_sptr->add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_UNIX_TIMESTAMP );
    auto* item = info.create_metadata_item( kwiver::vital::any((uint64_t)12345678) );
    meta_sptr->add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_SENSOR_VERTICAL_FOV );
    auto* item = info.create_metadata_item( kwiver::vital::any((double)12345.678) );
    meta_sptr->add( item );
  }

  {
    const auto& info = traits.find( kwiver::vital::VITAL_META_FRAME_CENTER );

    kwiver::vital::geo_point::geo_3d_point_t geo_3d( 42.50, 73.54, 100 );
    kwiver::vital::geo_point pt (  geo_3d, kwiver::vital::SRID::lat_lon_WGS84 );
    auto* item = info.create_metadata_item( kwiver::vital::any(pt) );
    meta_sptr->add( item );
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
    meta_sptr->add( item );
  }

  kasp::metadata meta_ser;      // The serializer

  kwiver::vital::metadata_vector mvec;
  mvec.push_back( meta_sptr );
  mvec.push_back( meta_sptr ); // just so there is more than one

  kwiver::vital::any meta_any( mvec );
  auto mes = meta_ser.serialize( meta_any );
  auto meta_dser_any = meta_ser.deserialize( *mes );

  kwiver::vital::metadata_vector meta_dser =
    kwiver::vital::any_cast< kwiver::vital::metadata_vector >( meta_dser_any );

  // test for equality
  EXPECT_TRUE( test_equal_content( *meta_sptr, *meta_dser[0] ));
}
