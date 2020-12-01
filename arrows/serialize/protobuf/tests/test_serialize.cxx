// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief test protobuf serializers
 */

#include <gtest/gtest.h>

#include <arrows/serialize/protobuf/activity.h>
#include <arrows/serialize/protobuf/activity_type.h>
#include <arrows/serialize/protobuf/bounding_box.h>
#include <arrows/serialize/protobuf/detected_object_type.h>
#include <arrows/serialize/protobuf/detected_object.h>
#include <arrows/serialize/protobuf/detected_object_set.h>
#include <arrows/serialize/protobuf/timestamp.h>
#include <arrows/serialize/protobuf/image.h>
#include <arrows/serialize/protobuf/string.h>
#include <arrows/serialize/protobuf/track_state.h>
#include <arrows/serialize/protobuf/track.h>
#include <arrows/serialize/protobuf/track_set.h>
#include <arrows/serialize/protobuf/object_track_state.h>
#include <arrows/serialize/protobuf/object_track_set.h>
#include <arrows/serialize/protobuf/convert_protobuf.h>

#include <vital/types/activity.h>
#include <vital/types/activity_type.h>
#include <vital/types/bounding_box.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/timestamp.h>
#include <vital/types/track.h>
#include <vital/types/track_set.h>
#include <vital/types/object_track_set.h>
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
TEST( serialize, activity_default )
{
  // This tests the behavior when participants
  // and detected_object_type are set to NULL
  auto const act = kwiver::vital::activity{};
  auto act_ser = kasp::activity{};

  auto const mes = act_ser.serialize( kwiver::vital::any( act ) );

  auto const dser = act_ser.deserialize( *mes );
  auto const act_dser =
    kwiver::vital::any_cast< kwiver::vital::activity >( dser );

  // Check members
  EXPECT_EQ( act.id(), act_dser.id() );
  EXPECT_EQ( act.label(), act_dser.label() );
  EXPECT_EQ( act.type(), act_dser.type() );
  EXPECT_EQ( act.participants(), act_dser.participants() );
  EXPECT_DOUBLE_EQ( act.confidence(), act_dser.confidence() );

  // Timestamps are invalid so can't do a direct comparison
  auto const start = act.start();
  auto const end = act.end();
  auto const start_dser = act_dser.start();
  auto const end_dser = act_dser.end();

  EXPECT_EQ( start.get_time_seconds(), start_dser.get_time_seconds() );
  EXPECT_EQ( start.get_frame(), start_dser.get_frame() );
  EXPECT_EQ( start.get_time_domain_index(), start_dser.get_time_domain_index() );

  EXPECT_EQ( end.get_time_seconds(), end_dser.get_time_seconds() );
  EXPECT_EQ( end.get_frame(), end_dser.get_frame() );
  EXPECT_EQ( end.get_time_domain_index(), end_dser.get_time_domain_index() );
}

// ----------------------------------------------------------------------------
TEST( serialize, activity )
{
  auto at_sptr = std::make_shared< kwiver::vital::activity_type >();
  at_sptr->set_score( "first", 1 );
  at_sptr->set_score( "second", 10 );
  at_sptr->set_score( "third", 101 );

  // Create object_track_set consisting of
  // 1 track_sptr with 10 track states
  auto track_sptr = kwiver::vital::track::create();
  track_sptr->set_id( 1 );
  for ( int i = 0; i < 10; i++ )
  {
    auto const bbox =
      kwiver::vital::bounding_box_d{ 10.0 + i, 10.0 + i, 20.0 + i, 20.0 + i };

    auto dobj_dot_sptr = std::make_shared< kwiver::vital::detected_object_type >();
    dobj_dot_sptr->set_score( "key", i / 10.0 );

    auto const dobj_sptr =
      std::make_shared< kwiver::vital::detected_object >( bbox, i / 10.0, dobj_dot_sptr );

    auto const ots_sptr =
      std::make_shared< kwiver::vital::object_track_state >( i, i, dobj_sptr );

    track_sptr->append( ots_sptr );
  }

  auto const tracks = std::vector< kwiver::vital::track_sptr >{ track_sptr };
  auto const obj_trk_set_sptr =
    std::make_shared< kwiver::vital::object_track_set >( tracks );

  // Now both timestamps
  auto const start = kwiver::vital::timestamp{ 1, 1 };
  auto const end = kwiver::vital::timestamp{ 2, 2 };

  // Now construct activity
  auto const act =
    kwiver::vital::activity{ 5, "test_label", 3.1415, at_sptr, start, end, obj_trk_set_sptr };

  auto act_ser = kasp::activity{};

  auto const mes = act_ser.serialize( kwiver::vital::any( act ) );

  auto const dser = act_ser.deserialize( *mes );
  auto const act_dser =
    kwiver::vital::any_cast< kwiver::vital::activity >( dser );

  // Now check equality
  EXPECT_EQ( act.id(), act_dser.id() );
  EXPECT_EQ( act.label(), act_dser.label() );
  EXPECT_DOUBLE_EQ( act.confidence(), act_dser.confidence() );
  EXPECT_EQ( act.start(), act_dser.start() );
  EXPECT_EQ( act.end(), act_dser.end() );

  // Check values in the retrieved class map
  auto const act_type = act.type();
  auto const act_type_dser = act_dser.type();
  EXPECT_EQ( act_type->size(), act_type_dser->size() );
  EXPECT_DOUBLE_EQ( act_type->score( "first" ),  act_type_dser->score( "first" ) );
  EXPECT_DOUBLE_EQ( act_type->score( "second" ), act_type_dser->score( "second" ) );
  EXPECT_DOUBLE_EQ( act_type->score( "third" ),  act_type_dser->score( "third" ) );

  // Now the object_track_set
  auto const parts = act.participants();
  auto const parts_dser = act_dser.participants();

  EXPECT_EQ( parts->size(), parts_dser->size() );

  auto const trk = parts->get_track( 1 );
  auto const trk_dser = parts_dser->get_track( 1 );

  // Iterate over the track_states
  for ( int i = 0; i < 10; i++ )
  {
    auto const trk_state_sptr = *trk->find( i );
    auto const trk_state_dser_sptr = *trk_dser->find( i );

    EXPECT_EQ( trk_state_sptr->frame(), trk_state_dser_sptr->frame() );

    auto const obj_trk_state_sptr =
      kwiver::vital::object_track_state::downcast( trk_state_sptr );
    auto const obj_trk_state_dser_sptr =
      kwiver::vital::object_track_state::downcast( trk_state_dser_sptr );

    EXPECT_EQ( obj_trk_state_sptr->time(), obj_trk_state_dser_sptr->time() );

    auto const do_ser_sptr = obj_trk_state_sptr->detection();
    auto const do_dser_sptr = obj_trk_state_dser_sptr->detection();

    EXPECT_EQ( do_ser_sptr->bounding_box(), do_dser_sptr->bounding_box() );
    EXPECT_EQ( do_ser_sptr->confidence(), do_dser_sptr->confidence() );

    auto const at_ser_sptr = do_ser_sptr->type();
    auto const at_dser_sptr = do_dser_sptr->type();

    if ( at_ser_sptr )
    {
      EXPECT_EQ( at_ser_sptr->size(), at_dser_sptr->size() );
      EXPECT_EQ( at_ser_sptr->score( "key" ), at_dser_sptr->score( "key" ) );
    }
  }
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
TEST( serialize, activity_type )
{
  kasp::activity_type at_ser; // get serializer
  kwiver::vital::activity_type at;

  at.set_score( "first", 1 );
  at.set_score( "second", 10 );
  at.set_score( "third", 101 );
  at.set_score( "last", 121 );

  kwiver::vital::any at_any( at );
  auto mes = at_ser.serialize( at_any );

  auto dser = at_ser.deserialize( *mes );
  kwiver::vital::activity_type at_dser =
    kwiver::vital::any_cast< kwiver::vital::activity_type >( dser );

  EXPECT_EQ( at.size(), at_dser.size() );

  auto o_it = at.begin();
  auto d_it = at_dser.begin();

  for (size_t i = 0; i < at.size(); ++i )
  {
    EXPECT_EQ( *(o_it->first), *(d_it->first) );
    EXPECT_EQ( o_it->second, d_it->second );
    ++o_it;
    ++d_it;
  }
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
    ++o_it;
    ++d_it;
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

  obj->add_note( "This is a note" );
  obj->add_note( "This is another note" );

  obj->add_keypoint( "point-1", ::kwiver::vital::point_2d( 123,234 ) );
  obj->add_keypoint( "point-2", ::kwiver::vital::point_2d( 234,123 ) );

  kwiver::vital::any obj_any( *obj );
  auto mes = obj_ser.serialize( obj_any );

  auto dser = obj_ser.deserialize( *mes );
  auto obj_dser = kwiver::vital::any_cast< kwiver::vital::detected_object_sptr >( dser );

  EXPECT_EQ( obj->bounding_box(), obj_dser->bounding_box() );
  EXPECT_EQ( obj->index(), obj_dser->index() );
  EXPECT_EQ( obj->confidence(), obj_dser->confidence() );
  EXPECT_EQ( obj->detector_name(), obj_dser->detector_name() );

  // Notes
  {
    auto obj_notes = obj->notes();
    auto dser_notes = obj_dser->notes();
    EXPECT_EQ( obj_notes.size(), dser_notes.size() );
    for ( size_t i = 0; i < obj_notes.size(); ++i )
    {
      EXPECT_EQ( obj_notes[i], dser_notes[i] );
    }
  }

  // keypoints
  {
    auto obj_kp = obj->keypoints();
    auto dser_kp = obj_dser->keypoints();
    EXPECT_EQ( obj_kp.size(), dser_kp.size() );
    EXPECT_EQ( obj_kp, dser_kp );
  }

  // class map
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
      ++o_it;
      ++d_it;
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

      for ( size_t j = 0; j < ser_dot_sptr->size(); ++j )
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

  EXPECT_EQ (tstamp, tstamp_dser);
}

// ----------------------------------------------------------------------------
TEST( serialize, image )
{
  kasp::image image_ser;
  kwiver::vital::image img{200, 300, 3};

  char* cp = static_cast< char* >(img.memory()->data() );
  for ( size_t i = 0; i < img.size(); ++i )
  {
    *cp++ = i;
  }

  {
    kwiver::vital::image_container_sptr img_container =
      std::make_shared< kwiver::vital::simple_image_container >( img );
    kwiver::vital::any img_any(img_container);

    auto mes = image_ser.serialize( img_any );
    auto dser = image_ser.deserialize( *mes );

    auto img_dser = kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( dser );

    // Check the content of images
    EXPECT_TRUE ( kwiver::vital::equal_content( img_container->get_image(), img_dser->get_image()) );
  }

  {
    kwiver::vital::image other( img.memory(),
                                ((char *)img.first_pixel() + 32),
                                100, 200, img.depth(),
                                img.w_step(), img.h_step(), img.d_step(),
                                img.pixel_traits() );

    kwiver::vital::image_container_sptr img_container =
      std::make_shared< kwiver::vital::simple_image_container >( other );
    kwiver::vital::any img_any(img_container);

    auto mes = image_ser.serialize( img_any );
    auto dser = image_ser.deserialize( *mes );

    auto img_dser = kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( dser );

    // Check the content of images
    EXPECT_TRUE( kwiver::vital::equal_content( img_container->get_image(), img_dser->get_image()) );
  }

  {
    kwiver::vital::image other( img.memory(),
                                ((char *)img.first_pixel() + (3 * img.width()) ),
                                img.width(), 200, img.depth(),
                                img.w_step(), img.h_step(), img.d_step(),
                                img.pixel_traits() );

    kwiver::vital::image_container_sptr img_container =
      std::make_shared< kwiver::vital::simple_image_container >( other );
    kwiver::vital::any img_any(img_container);

    auto mes = image_ser.serialize( img_any );
    auto dser = image_ser.deserialize( *mes );

    auto img_dser = kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( dser );

    // Check the content of images
    EXPECT_TRUE( kwiver::vital::equal_content( img_container->get_image(), img_dser->get_image()) );
  }
}

// ----------------------------------------------------------------------------
TEST ( serialize, string )
{
  kasp::string str_ser;
  std::string str("Test string");

  kwiver::vital::any str_any(str);

  auto mes = str_ser.serialize( str_any );
  auto dser = str_ser.deserialize( *mes );

  std::string str_dser =
    kwiver::vital::any_cast< std::string > ( dser );

  EXPECT_EQ (str, str_dser);
}

// ----------------------------------------------------------------------------
TEST (serialize, track_state)
{
  kasp::track_state trk_state_ser;
  kwiver::vital::track_state trk_state( 1 );

  kwiver::vital::any trk_state_any( trk_state );

  auto mes = trk_state_ser.serialize( trk_state_any );
  auto dser = trk_state_ser.deserialize( *mes );

  kwiver::vital::track_state trk_state_dser =
    kwiver::vital::any_cast< kwiver::vital::track_state >( dser );

  EXPECT_EQ( trk_state, trk_state_dser );
}

// ----------------------------------------------------------------------------
TEST (serialize, object_track_state)
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
  kwiver::vital::object_track_state obj_trk_state( 1, 1, dobj_sptr );

  kasp::object_track_state obj_trk_state_ser;
  kwiver::vital::any obj_trk_state_any(obj_trk_state);

  auto mes = obj_trk_state_ser.serialize( obj_trk_state_any );
  auto dser = obj_trk_state_ser.deserialize( *mes );

  kwiver::vital::object_track_state obj_trk_state_dser =
    kwiver::vital::any_cast< kwiver::vital::object_track_state > ( dser );

  auto ser_do_sptr = obj_trk_state.detection();
  auto deser_do_sptr = obj_trk_state_dser.detection();

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
  EXPECT_EQ( obj_trk_state.time(), obj_trk_state_dser.time() );
  EXPECT_EQ( obj_trk_state.frame(), obj_trk_state_dser.frame() );
}
// ---------------------------------------------------------------------------
TEST( serialize, track )
{
  // test track with object track state
  auto obj_trk = kwiver::vital::track::create();
  obj_trk->set_id( 1 );
  for ( int i=0; i<10; i++ )
  {
    auto dot = std::make_shared< kwiver::vital::detected_object_type >();

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

    bool insert_success = obj_trk->insert( obj_trk_state_sptr );
    if ( !insert_success )
    {
      std::cerr << "Failed to insert object track state" << std::endl;
    }
  }

  kasp::track obj_trk_ser;
  kwiver::vital::any obj_trk_any( obj_trk );

  auto mes = obj_trk_ser.serialize( obj_trk_any );
  auto dser = obj_trk_ser.deserialize( *mes );

  auto obj_trk_dser_sptr = kwiver::vital::any_cast< kwiver::vital::track_sptr >( dser );

  // Check track id
  EXPECT_EQ( obj_trk->id(), obj_trk_dser_sptr->id() );

  for( int i=0; i<10; i++ )
  {
    auto trk_state_sptr = *obj_trk->find( i );
    auto dser_trk_state_sptr = *obj_trk_dser_sptr->find( i );

    EXPECT_EQ( trk_state_sptr->frame(), dser_trk_state_sptr->frame() );
    auto obj_trk_state_sptr = kwiver::vital::object_track_state::downcast(
                                                                trk_state_sptr );
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
      EXPECT_EQ( ser_dot_sptr->size(), dser_dot_sptr->size() );

      auto ser_it = ser_dot_sptr->begin();
      auto dser_it = dser_dot_sptr->begin();

      for ( size_t j = 0; j < ser_dot_sptr->size(); ++j )
      {
        EXPECT_EQ( *( ser_it->first ), *( ser_it->first ) );
        EXPECT_EQ( dser_it->second, dser_it->second );
      }
    }

  }

  // test track with track state
  auto trk = kwiver::vital::track::create();
  trk->set_id( 2 );
  for ( int ii = 0; ii < 10; ii++ )
  {
    auto trk_state_sptr = std::make_shared< kwiver::vital::track_state>( ii );
    bool insert_success = trk->insert( trk_state_sptr );
    if ( !insert_success )
    {
      std::cerr << "Failed to insert track state" << std::endl;
    }
  }

  kasp::track trk_ser;
  kwiver::vital::any trk_any( trk );

  auto trk_mes = trk_ser.serialize( trk_any );
  auto trk_dser = trk_ser.deserialize( *trk_mes );

  auto trk_dser_sptr =
    kwiver::vital::any_cast< kwiver::vital::track_sptr > ( trk_dser );

  EXPECT_EQ( trk->id(), trk_dser_sptr->id() );

  for ( int i=0; i<10; i++ )
  {
    auto obj_trk_state_sptr = *trk->find( i );
    auto trk_state_dser_sptr = *trk_dser_sptr->find( i );

    EXPECT_EQ( obj_trk_state_sptr->frame(), trk_state_dser_sptr->frame() );
  }

}

// ---------------------------------------------------------------------------
TEST( serialize, track_set )
{
  auto trk_set_sptr = std::make_shared< kwiver::vital::track_set >();
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

  kasp::track_set trk_set_ser;
  kwiver::vital::any trk_set_any( trk_set_sptr );
  auto trk_set_mes = trk_set_ser.serialize( trk_set_any );
  auto trk_set_dser = trk_set_ser.deserialize( *trk_set_mes );

  auto trk_set_sptr_dser =
    kwiver::vital::any_cast< kwiver::vital::track_set_sptr > ( trk_set_dser );

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
TEST( serialize, object_track_set )
{
  auto obj_trk_set_sptr = std::make_shared< kwiver::vital::object_track_set >();
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

  kasp::object_track_set obj_trk_set_ser;
  kwiver::vital::any obj_trk_set_any( obj_trk_set_sptr );
  auto trk_set_mes = obj_trk_set_ser.serialize( obj_trk_set_any );
  auto trk_set_dser = obj_trk_set_ser.deserialize( *trk_set_mes );

  auto obj_trk_set_sptr_dser =
    kwiver::vital::any_cast< kwiver::vital::object_track_set_sptr > ( trk_set_dser );

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

        for ( size_t j = 0; j < ser_dot_sptr->size(); ++j )
        {
          EXPECT_EQ( *(ser_it->first), *(ser_it->first) );
          EXPECT_EQ( dser_it->second, dser_it->second );
        }
      }
    }
  }
}
