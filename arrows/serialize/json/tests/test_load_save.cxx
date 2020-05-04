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

#include <arrows/serialize/json/load_save.h>
#include <arrows/serialize/json/load_save_point.h>
#include <arrows/serialize/json/load_save_track_state.h>
#include <arrows/serialize/json/load_save_track_set.h>

#include <gtest/gtest.h>

#include <vital/types/bounding_box.h>
#include <vital/types/covariance.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/geodesy.h>
#include <vital/types/image_container.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_tags.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/object_track_set.h>
#include <vital/types/point.h>
#include <vital/types/polygon.h>
#include <vital/types/timestamp.h>
#include <vital/types/track.h>
#include <vital/types/track_set.h>
#include <vital/vital_types.h>

#include <arrows/serialize/json/track.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>
#include <iostream>

#define DEBUG 0

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
    cereal::save( ar, obj );
  }

#if DEBUG
  std::cout << "bbox as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::bounding_box_d obj_dser { 0, 0, 0, 0 };
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
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
    cereal::save( ar, obj );
  }

#if DEBUG
  std::cout << "polygon as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::polygon obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
  }

  EXPECT_EQ( obj.num_vertices(), obj_dser.num_vertices() );
  EXPECT_EQ( obj.at(0), obj_dser.at(0) );
  EXPECT_EQ( obj.at(1), obj_dser.at(1) );
  EXPECT_EQ( obj.at(2), obj_dser.at(2) );
  EXPECT_EQ( obj.at(3), obj_dser.at(3) );
}

// ----------------------------------------------------------------------------
TEST( load_save, geo_point_2d )
{
  kwiver::vital::geo_point::geo_2d_point_t geo( 42.50, 73.54 );
  kwiver::vital::geo_point obj( geo, kwiver::vital::SRID::lat_lon_WGS84 );

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, obj );
  }

#if DEBUG
  std::cout << "geo_point as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::geo_point obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
  }

  EXPECT_EQ( obj.location(), obj_dser.location() );
}

// ----------------------------------------------------------------------------
TEST( load_save, geo_point_3d )
{
  kwiver::vital::geo_point::geo_3d_point_t geo( 42.50, 73.54, 16.33 );
  kwiver::vital::geo_point obj( geo, kwiver::vital::SRID::lat_lon_WGS84 );

  std::stringstream msg;
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, obj );
  }

#if DEBUG
  std::cout << "geo_point as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::geo_point obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
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
    cereal::save( ar, obj );
  }

#if DEBUG
  std::cout << "geo_polygon as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::geo_polygon obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
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
kwiver::vital::metadata create_meta_collection()
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
    kwiver::vital::geo_point::geo_2d_point_t geo( 42.50, 73.54 );
    kwiver::vital::geo_point pt ( geo, kwiver::vital::SRID::lat_lon_WGS84 );
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

  return meta;
}

// ----------------------------------------------------------------------------
void compare_meta_collection( const kwiver::vital::metadata& lhs,
                              const kwiver::vital::metadata& rhs )
{
  // Check to make sure they are the same
  for ( const auto& it : lhs )
  {
    const auto lhs_item = it.second;

    EXPECT_TRUE( rhs.has( lhs_item->tag() ) );

    const auto& rhs_item = rhs.find( lhs_item->tag() );

    // test for data being the same
    EXPECT_EQ( lhs_item->data().type(), rhs_item.data().type() );

    //+ TBD check data values for equal
  } // end for

}

// ----------------------------------------------------------------------------
TEST( load_save, metadata )
{
  kwiver::vital::metadata meta = create_meta_collection();

  std::stringstream msg;

  try
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, meta);
  }
  catch(std::exception const& e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }

#if DEBUG
  std::cout << "metadata as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::metadata obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
  }

  compare_meta_collection( meta, obj_dser );
}


// ----------------------------------------------------------------------------
TEST( load_save, metadata_vector )
{
  kwiver::vital::metadata_sptr meta = std::make_shared<kwiver::vital::metadata>( create_meta_collection() );
  kwiver::vital::metadata_vector meta_vect;

  meta_vect.push_back( meta );
  meta_vect.push_back( meta );


  std::stringstream msg;

  try
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, meta_vect);
  }
  catch(std::exception const& e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  }

#if DEBUG
  std::cout << "metadata vector as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::metadata_vector obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
  }

  EXPECT_EQ( meta_vect.size(), obj_dser.size() );

  // Check to make sure they are the same
  for (size_t i = 0; i < meta_vect.size(); i++)
  {
    compare_meta_collection( *meta_vect[i], *obj_dser[i] );
  }
}

// ----------------------------------------------------------------------------
TEST( load_save, track_state)
{
  kwiver::vital::track_state trk_state{1};
  std::stringstream msg;

  try
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, trk_state);
  }
  catch(std::exception const& e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  }

#if DEBUG
  std::cout << "track state as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::track_state obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
  }


  EXPECT_EQ( trk_state.frame(), obj_dser.frame() );
}
// ----------------------------------------------------------------------------
TEST( load_save, object_track_state)
{
  auto dot = std::make_shared< kwiver::vital::detected_object_type >();

  dot->set_score( "first", 1 );
  dot->set_score( "second", 10 );
  dot->set_score( "third", 101 );
  dot->set_score( "last", 121 );

  // create detected object
  auto obj = std::make_shared< kwiver::vital::detected_object >(
    kwiver::vital::bounding_box_d{ 1, 2, 3, 4 }, 3.14159, dot );
  obj->set_detector_name( "test_detector" );
  obj->set_index( 1234 );
  obj->add_note( "this is a note" );

  ::kwiver::vital::point_2d p2d( 123, 456 );
  obj->add_keypoint( "keypoint-1", p2d );

  ::kwiver::vital::geo_point::geo_3d_point_t g3d( 123, 234, 345 );
  obj->set_geo_point( ::kwiver::vital::geo_point( g3d, 42 ) );

  // Create object track state
  kwiver::vital::object_track_state obj_trk_state( 1, 1, obj );

  obj_trk_state.set_image_point( ::kwiver::vital::point_2d( 123, 321 ) );
  obj_trk_state.set_track_point( ::kwiver::vital::point_3d( 123, 234, 345 ) );

  std::stringstream msg;
  try
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, obj_trk_state );
  }
  catch ( std::exception const& e )
  {
    std::cout << "exception caught: " << e.what() << std::endl;
  }

#if DEBUG

  std::cout << "object track state as json - " << msg.str() << std::endl;
#endif

  kwiver::vital::object_track_state obj_dser;
  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, obj_dser );
  }

  auto do_sptr = obj_trk_state.detection();
  auto do_sptr_dser = obj_dser.detection();

  EXPECT_EQ( do_sptr->bounding_box(), do_sptr_dser->bounding_box() );
  EXPECT_EQ( do_sptr->index(), do_sptr_dser->index() );
  EXPECT_EQ( do_sptr->confidence(), do_sptr_dser->confidence() );
  EXPECT_EQ( do_sptr->detector_name(), do_sptr_dser->detector_name() );

  EXPECT_EQ( do_sptr->notes().size(), do_sptr_dser->notes().size() );
  EXPECT_EQ( do_sptr->notes()[ 0 ], do_sptr_dser->notes()[ 0 ] );

  EXPECT_EQ( do_sptr->keypoints().size(), do_sptr_dser->keypoints().size() );

  auto kpts = do_sptr_dser->keypoints();
  EXPECT_EQ( kpts.count( "keypoint-1" ), 1 );

  auto dot_sptr_dser = do_sptr_dser->type();

  if ( dot )
  {
    EXPECT_EQ( dot->size(), dot_sptr_dser->size() );

    auto it = dot->begin();
    auto it_dser = dot_sptr_dser->begin();

    for ( size_t i = 0; i < dot->size(); ++i )
    {
      EXPECT_EQ( *( it->first ), *( it_dser->first ) );
      EXPECT_EQ( it->second, it_dser->second );
    }
  }

  EXPECT_EQ( obj_trk_state.time(), obj_dser.time() );
  EXPECT_EQ( obj_trk_state.frame(), obj_dser.frame() );
  EXPECT_EQ( obj_trk_state.image_point().value(), obj_dser.image_point().value() );
  EXPECT_EQ( obj_trk_state.track_point().value(), obj_dser.track_point().value() );
}
// ============================================================================
TEST( load_save, track_set )
{
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
  std::stringstream msg;

  try
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, *trk_set_sptr);
  }
  catch(std::exception const& e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  }

#if DEBUG
  std::cout << "track set as json - " << msg.str() << std::endl;
#endif

  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, *trk_set_sptr_dser );
  }

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
  } // end for

}

// ---------------------------------------------------------------------------
TEST( load_save, object_track_set )
{
  auto obj_trk_set_sptr = std::make_shared< kwiver::vital::object_track_set >();
  auto obj_trk_set_sptr_dser = std::make_shared< kwiver::vital::object_track_set >();
  for ( kwiver::vital::track_id_t trk_id = 1; trk_id < 3; ++trk_id )
  {
    auto trk = kwiver::vital::track::create();
    trk->set_id( trk_id );
    for ( int i = trk_id * 2; i < ( trk_id+1 )*2; i++ )
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
  std::stringstream msg;

  try
  {
    cereal::JSONOutputArchive ar( msg );
    cereal::save( ar, *obj_trk_set_sptr);
  }
  catch(std::exception const& e) {
    std::cout << "exception caught: " << e.what() << std::endl;
  }

#if DEBUG
  std::cout << "object track set as json - " << msg.str() << std::endl;
#endif

  {
    cereal::JSONInputArchive ar( msg );
    cereal::load( ar, *obj_trk_set_sptr_dser );
  }

  for ( kwiver::vital::track_id_t trk_id=1; trk_id<3; ++trk_id )
  {
    auto trk = obj_trk_set_sptr->get_track( trk_id );
    auto trk_dser = obj_trk_set_sptr_dser->get_track( trk_id );
    EXPECT_EQ( trk->id(), trk_dser->id() );
    for ( int i=trk_id*2; i < ( trk_id+1 )*2; i++ )
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

// ---------------------------------------------------------------------------
TEST( load_save, covariance )
{

#define TEST_COV(T, ... )                                               \
do {                                                                    \
  ::kwiver::vital::T::matrix_type val;                                  \
  val << __VA_ARGS__;                                                   \
  ::kwiver::vital::T obj( val );                                        \
  std::stringstream msg;                                                \
  {                                                                     \
    cereal::JSONOutputArchive ar( msg );                                \
    cereal::save( ar, obj );                                            \
  }                                                                     \
                                                                        \
  if (DEBUG)                                                            \
  {                                                                     \
    std::cout << # T << " as json - " << msg.str() << std::endl;        \
  }                                                                     \
                                                                        \
  ::kwiver::vital::T obj_dser;                                          \
  {                                                                     \
    cereal::JSONInputArchive ar( msg );                                 \
    cereal::load( ar, obj_dser );                                       \
  }                                                                     \
                                                                        \
  EXPECT_EQ( obj, obj_dser );                                           \
} while(0)


TEST_COV( covariance_2d, 1, 2, 3, 4 );
TEST_COV( covariance_2f, 1, 2, 3, 4 );
TEST_COV( covariance_3d, 1, 2, 3, 4, 5, 6, 7, 8, 9 );
TEST_COV( covariance_3f, 1, 2, 3, 4, 5, 6, 7, 8, 9 );
TEST_COV( covariance_4d, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );
TEST_COV( covariance_4f, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16 );

#undef TEST_COV

}

// ---------------------------------------------------------------------------
TEST( load_save, points )
{

#define TEST_POINT(T, ... )                                             \
do {                                                                    \
  ::kwiver::vital::T obj{ __VA_ARGS__ };                                \
    std::stringstream msg;                                              \
  {                                                                     \
    cereal::JSONOutputArchive ar( msg );                                \
    cereal::save( ar, obj );                                            \
  }                                                                     \
                                                                        \
  if (DEBUG)                                                            \
  {                                                                     \
    std::cout << # T << " as json - " << msg.str() << std::endl;        \
  }                                                                     \
                                                                        \
  ::kwiver::vital::T obj_dser;                                          \
  {                                                                     \
    cereal::JSONInputArchive ar( msg );                                 \
    cereal::load( ar, obj_dser );                                       \
  }                                                                     \
                                                                        \
  EXPECT_EQ( obj.value(), obj_dser.value() );                           \
} while(0)


  TEST_POINT( point_2i, 1, 2 );
  TEST_POINT( point_2d, 1, 2 );
  TEST_POINT( point_2f, 1, 2 );
  TEST_POINT( point_3d, 1, 2, 3 );
  TEST_POINT( point_3f, 1, 2, 3 );
  TEST_POINT( point_4d, 1, 2, 3, 4 );
  TEST_POINT( point_4f, 1, 2, 3, 4 );

#undef TEST_POINT

}
