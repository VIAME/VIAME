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

#include "convert_protobuf.h"
#include "convert_protobuf_point.h"

#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/polygon.h>
#include <vital/types/timestamp.h>
#include <vital/types/metadata.h>
#include <vital/types/metadata_traits.h>
#include <vital/types/track.h>
#include <vital/types/track_set.h>
#include <vital/types/object_track_set.h>
#include <vital/util/hex_dump.h>
#include <vital/exceptions.h>
#include <vital/vital_types.h>

#include <vital/types/protobuf/bounding_box.pb.h>
#include <vital/types/protobuf/detected_object.pb.h>
#include <vital/types/protobuf/detected_object_set.pb.h>
#include <vital/types/protobuf/detected_object_type.pb.h>
#include <vital/types/protobuf/geo_polygon.pb.h>
#include <vital/types/protobuf/geo_point.pb.h>
#include <vital/types/protobuf/metadata.pb.h>
#include <vital/types/protobuf/polygon.pb.h>
#include <vital/types/protobuf/timestamp.pb.h>
#include <vital/types/protobuf/metadata.pb.h>
#include <vital/types/protobuf/string.pb.h>
#include <vital/types/protobuf/image.pb.h>
#include <vital/types/protobuf/track.pb.h>
#include <vital/types/protobuf/track_set.pb.h>
#include <vital/types/protobuf/track_state.pb.h>
#include <vital/types/protobuf/object_track_state.pb.h>
#include <vital/types/protobuf/object_track_set.pb.h>

#include <zlib.h>
#include <cstddef>
#include <cstring>
#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::bounding_box&  proto_bbox,
                  ::kwiver::vital::bounding_box_d& bbox )
 {
   bbox = ::kwiver::vital::bounding_box_d( proto_bbox.xmin(),
                                         proto_bbox.ymin(),
                                         proto_bbox.xmax(),
                                         proto_bbox.ymax());
 }

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::bounding_box_d& bbox,
                  ::kwiver::protobuf::bounding_box&  proto_bbox )
{
  proto_bbox.set_xmin( bbox.min_x() );
  proto_bbox.set_ymin( bbox.min_y() );
  proto_bbox.set_xmax( bbox.max_x() );
  proto_bbox.set_ymax( bbox.max_y() );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::detected_object&  proto_det_object,
                  ::kwiver::vital::detected_object& det_object )
{
  det_object.set_confidence( proto_det_object.confidence() );

  ::kwiver::vital::bounding_box_d bbox{ 0, 0, 0, 0 };
  ::kwiver::protobuf::bounding_box proto_bbox = proto_det_object.bbox();
  convert_protobuf( proto_bbox, bbox );
  det_object.set_bounding_box( bbox );

  if ( proto_det_object.has_classifcations() )
  {
    auto new_dot = std::make_shared< ::kwiver::vital::detected_object_type >();
    ::kwiver::protobuf::detected_object_type proto_dot = proto_det_object.classifcations();
    convert_protobuf( proto_dot, *new_dot );
    det_object.set_type( new_dot );
  }

  if ( proto_det_object.has_index() )
  {
    det_object.set_index( proto_det_object.index() ) ;
  }


  if ( proto_det_object.has_detector_name() )
  {
    det_object.set_detector_name( proto_det_object.detector_name() );
  }

  // Notes
  det_object.clear_notes();
  for ( int i = 0; i < proto_det_object.notes_size(); ++i )
  {
    det_object.add_note( proto_det_object.notes(i) );
  } // end for

  // keypoints
  det_object.clear_keypoints();
  auto keypoint_map = proto_det_object.keypoints();
  for ( const auto & entry : keypoint_map )
  {
    ::kwiver::vital::point_2d pt;
    convert_protobuf( entry.second, pt );
    det_object.add_keypoint( entry.first, pt );
  } // end for

  // geo_point
  ::kwiver::vital::geo_point gpt;
  convert_protobuf( proto_det_object.geopoint(), gpt );
  det_object.set_geo_point( gpt );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::detected_object& det_object,
                  ::kwiver::protobuf::detected_object&  proto_det_object )
{
  proto_det_object.set_confidence( det_object.confidence() );

  auto* proto_bbox = proto_det_object.mutable_bbox();
  convert_protobuf( det_object.bounding_box(), *proto_bbox );


  // We're using type() in "const" (read only) way here.  There's utility
  // in having the source object parameter be const, but type() isn't because
  // its a pointer into the det_object.  Using const_cast here is a middle ground
  // though somewhat ugly
  if ( const_cast<::kwiver::vital::detected_object&>(det_object).type() != NULL )
  {
    auto* proto_dot = proto_det_object.mutable_classifcations();
    convert_protobuf( * const_cast<::kwiver::vital::detected_object&>(det_object).type(), *proto_dot );

  }

  proto_det_object.set_index( det_object.index() );

  proto_det_object.set_detector_name( det_object.detector_name() );

  // Notes
  auto l_notes = det_object.notes();
  if ( ! l_notes.empty() )
  {
    for ( auto& n : l_notes )
    {
      proto_det_object.add_notes( n );
    } // end for
  }

  // keypoints
  auto l_keyp = det_object.keypoints();
  if ( ! l_keyp.empty() )
  {
    auto* mkp = proto_det_object.mutable_keypoints();

    for ( auto it : l_keyp )
    {
      ::kwiver::protobuf::point_d proto_pt;
      convert_protobuf( it.second, proto_pt );

      (*mkp)[it.first] = proto_pt;

    } // end for
  }

  // geopoint
  auto proto_geop = proto_det_object.mutable_geopoint();
  convert_protobuf( det_object.geo_point(), *proto_geop );

}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::detected_object_set&  proto_dos,
                  ::kwiver::vital::detected_object_set& dos )
{
  const size_t count( proto_dos.detected_objects_size() );
  for (size_t i = 0; i < count; ++i )
  {
    auto det_object_sptr = std::make_shared< ::kwiver::vital::detected_object >(
      ::kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );
    auto proto_det_object = proto_dos.detected_objects( i );

    convert_protobuf(proto_det_object,*det_object_sptr);

    dos.add( det_object_sptr );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::detected_object_set& dos,
                  ::kwiver::protobuf::detected_object_set&  proto_dos )
{
  // We're using type() in "const" (read only) way here.  There's utility
  // in having the source object parameter be const, but type() isn't because
  // its a pointer into the det_object.  Using const_cast here is a middle ground
  // though somewhat ugly
  for ( const auto it: const_cast< ::kwiver::vital::detected_object_set& >( dos ) )
  {
    ::kwiver::protobuf::detected_object *proto_det_object_ptr = proto_dos.add_detected_objects();

    convert_protobuf( *it, *proto_det_object_ptr );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::detected_object_type&  proto_dot,
                       ::kwiver::vital::detected_object_type& dot )
 {
   const size_t count( proto_dot.name_size() );
   for (size_t i = 0; i < count; ++i )
   {
     dot.set_score( proto_dot.name(i), proto_dot.score(i) );
   }
 }

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::detected_object_type& dot,
                       ::kwiver::protobuf::detected_object_type&  proto_dot )
{
  for ( const auto it : dot )
  {
    proto_dot.add_name( *(it.first) );
    proto_dot.add_score( it.second);
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::image&      proto_img,
                       ::kwiver::vital::image_container_sptr& img )
{
  const size_t img_size( proto_img.size() );
  auto mem_sptr = std::make_shared<vital::image_memory>( img_size );

  // decompress the data
  uLongf out_size( img_size );
  Bytef* out_buf = reinterpret_cast< Bytef* >( mem_sptr->data() );
  Bytef const *in_buf = reinterpret_cast< Bytef const* >( proto_img.data().data() );
  uLongf in_size = proto_img.data().size(); // compressed size

  int z_rc = uncompress(out_buf, &out_size, // outputs
                        in_buf, in_size);   // inputs
  if (Z_OK != z_rc )
  {
    switch (z_rc)
    {
    case Z_MEM_ERROR:
      LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data." );
      break;
    } // end switch
    return;
  }

  if (static_cast<uLongf>( img_size ) != out_size)
  {
    LOG_ERROR( ::kwiver::vital::get_logger( "image_container" ),
               "Uncompressed data not expected size. Possible data corruption.");
    return;
  }

  // create pixel trait
  const ::kwiver::vital::image_pixel_traits pix_trait(
    static_cast<::kwiver::vital::image_pixel_traits::pixel_type>(proto_img.trait_type() ),
    proto_img.trait_num_bytes() );

  // create the image
  auto vital_image = ::kwiver::vital::image(
    mem_sptr, mem_sptr->data(),
    static_cast< std::size_t > ( proto_img.width() ),
    static_cast< std::size_t > ( proto_img.height() ),
    static_cast< std::size_t > ( proto_img.depth() ),
    static_cast< std::ptrdiff_t > ( proto_img.w_step() ),
    static_cast< std::ptrdiff_t > ( proto_img.h_step() ),
    static_cast< std::ptrdiff_t > ( proto_img.d_step() ),
    pix_trait
    );

  // return newly constructed image container
  img = std::make_shared< ::kwiver::vital::simple_image_container >( vital_image );

  // convert metadata if there is any
  if ( proto_img.has_image_metadata() )
  {
    auto meta_ptr = std::make_shared< ::kwiver::vital::metadata >();
    convert_protobuf( proto_img.image_metadata(), *meta_ptr );
    img->set_metadata( meta_ptr );
  }
}


// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::image_container_sptr img,
                       ::kwiver::protobuf::image&                  proto_img )
{
  ::kwiver::vital::image vital_image = img->get_image();
  ::kwiver::vital::image local_image;

  if ( vital_image.memory() == nullptr || ! vital_image.is_contiguous() )
  {
    // Either we do not own the memory or it is not contiguous.  We
    // need to consolidate the input image into a contiguous memory
    // block before it can be serialized.
    local_image.copy_from( vital_image );
  }
  else
  {
    local_image = vital_image;
  }

  // Compress raw pixel data
  const uLongf size = compressBound( local_image.size() );
  uLongf out_size(size);
  std::vector<uint8_t> image_data( size );
  Bytef *out_buf = new Bytef[size];
  Bytef const* in_buf = reinterpret_cast< Bytef * >(local_image.memory()->data());

  // Since the image is contiguous, we can calculate the size
  size_t local_size = local_image.width() * local_image.height()
    * local_image.depth() * local_image.pixel_traits().num_bytes;

  int z_rc = compress( out_buf, &out_size, // outputs
                       in_buf, local_size ); // inputs
  if (Z_OK != z_rc )
  {
    switch (z_rc)
    {
    case Z_MEM_ERROR:
      LOG_ERROR( ::kwiver::vital::get_logger( "image_container" ),
                 "Error compressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( ::kwiver::vital::get_logger( "image_container" ),
                 "Error compressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( ::kwiver::vital::get_logger( "image_container" ),
                 "Error compressing image data." );
      break;
    } // end switch
  }
  else
  {

    proto_img.set_width( static_cast< int64_t > ( local_image.width() ) );
    proto_img.set_height( static_cast< int64_t > ( local_image.height() ) );
    proto_img.set_depth( static_cast< int64_t > ( local_image.depth() ) );

    proto_img.set_w_step( static_cast< int64_t > ( local_image.w_step() ) );
    proto_img.set_h_step( static_cast< int64_t > ( local_image.h_step() ) );
    proto_img.set_d_step( static_cast< int64_t > ( local_image.d_step() ) );

    // Get pixel trait
    auto pixel_trait = local_image.pixel_traits();
    proto_img.set_trait_type( pixel_trait.type );
    proto_img.set_trait_num_bytes( pixel_trait.num_bytes );

    proto_img.set_size( local_image.size() ); // uncompressed size
    proto_img.set_data( out_buf, size );

    // serialize the metadata if there is any.
    if ( img->get_metadata() )
    {
      auto* proto_meta = proto_img.mutable_image_metadata();
      convert_protobuf( *img->get_metadata(), *proto_meta );
    }
  }
  delete []out_buf;
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::timestamp& proto_tstamp,
                       ::kwiver::vital::timestamp&          tstamp )
{
  tstamp = ::kwiver::vital::timestamp(
    static_cast< ::kwiver::vital::time_usec_t > ( proto_tstamp.time() ),
    static_cast< ::kwiver::vital::frame_id_t > ( proto_tstamp.frame() ) );
}


// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::timestamp&  tstamp,
                       ::kwiver::protobuf::timestamp&     proto_tstamp )
{
  proto_tstamp.set_time( static_cast< int64_t > ( tstamp.get_time_usec() ) );
  proto_tstamp.set_frame( static_cast< int64_t > ( tstamp.get_frame() ) );
}


// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::metadata_vector&  proto_mvec,
                  ::kwiver::vital::metadata_vector& mvec )

{
  for ( const auto& meta : proto_mvec.collection() )
  {
    auto meta_ptr = std::make_shared< ::kwiver::vital::metadata >();
    convert_protobuf( meta, *meta_ptr );
    mvec.push_back( meta_ptr );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::metadata_vector& mvec,
                  ::kwiver::protobuf::metadata_vector&  proto_mvec )
{
  // convert to proto
  for ( const auto& meta : mvec )
  {
    auto* proto_meta = proto_mvec.add_collection();
    convert_protobuf( *meta, *proto_meta );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::metadata&  proto,
                  ::kwiver::vital::metadata& metadata )
{
  static ::kwiver::vital::metadata_traits traits;

  // deserialize one metadata collection
  for ( const auto& mi : proto.items() )
  {
    const auto tag = static_cast< ::kwiver::vital::vital_metadata_tag >( mi.metadata_tag() );
    const auto& trait = traits.find( tag );
    ::kwiver::vital::any data;

    if ( trait.is_floating_point() )
    {
      data = ::kwiver::vital::any( mi.double_value() );
    }
    else if ( trait.is_integral() )
    {
      data = ::kwiver::vital::any( static_cast<uint64_t>(mi.int_value()) );
    }
    else if ( trait.tag_type() == typeid(std::string) )
    {
      // is natively a string
      data = ::kwiver::vital::any( mi.string_value() );
    }
    else if ( trait.tag_type() == typeid(::kwiver::vital::geo_point) )
    {
      ::kwiver::vital::geo_point pt;
      convert_protobuf( mi.geo_point_value(), pt );
      data = ::kwiver::vital::any( pt );
    }
    else if ( trait.tag_type() == typeid(::kwiver::vital::geo_polygon) )
    {
      ::kwiver::vital::geo_polygon poly;
      convert_protobuf( mi.geo_polygon_value(), poly );
      data = ::kwiver::vital::any( poly );
    }
    else
    {
      std::stringstream str;
      str << "Found unexpected data type \"" << trait.tag_type().name()
          << "\" in metadata collection for item name \""
          << trait.name() << "\".";
      VITAL_THROW( ::kwiver::vital::metadata_exception, str.str() );
    }

    metadata.add( trait.create_metadata_item( data ) );

  } // end for
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::metadata& metadata,
                  ::kwiver::protobuf::metadata&  proto_meta )
{
  static ::kwiver::vital::metadata_traits traits;

  // Serialize one metadata collection
  for ( const auto& mi : metadata )
  {
    auto* proto_item = proto_meta.add_items();

    // element is <tag, any>
    const auto tag = mi.first;
    const auto metap = mi.second;
    const auto& trait = traits.find( tag );

    proto_item->set_metadata_tag( tag );

    if ( metap->has_double() )
    {
      proto_item->set_double_value( metap->as_double() );
    }
    else if ( metap->has_uint64() )
    {
      proto_item->set_int_value( metap->as_uint64() );
    }
    else if ( metap->has_string() )
    {
      proto_item->set_string_value( metap->as_string() );
    }
    else if ( trait.tag_type() == typeid(::kwiver::vital::geo_point) )
    {
      ::kwiver::vital::geo_point pt;
      if ( ! metap->data<::kwiver::vital::geo_point>( pt ) )
      {
        std::stringstream str;
        str << "Error extracting data item from metadata. "
            << "Expected \"::kwiver::vital::geo_point\" but found \""
            << metap->data().type_name() << "\"." ;
        VITAL_THROW( ::kwiver::vital::metadata_exception, str.str() );
      }

      auto* proto_pt = proto_item->mutable_geo_point_value();
      convert_protobuf( pt, *proto_pt );
    }
    else if ( trait.tag_type() == typeid(::kwiver::vital::geo_polygon) )
    {
      ::kwiver::vital::geo_polygon poly;
      if ( ! metap->data<::kwiver::vital::geo_polygon>( poly ) )
      {
        std::stringstream str;
        str << "Error extracting data item from metadata. "
            << "Expected \"::kwiver::vital::geo_polygon\" but found \""
            << metap->data().type_name() << "\"." ;
        VITAL_THROW( ::kwiver::vital::metadata_exception, str.str() );

      }

      auto* proto_poly = proto_item->mutable_geo_polygon_value();
      convert_protobuf( poly, *proto_poly );
    }
    else
    {
      // encode as string.
      proto_item->set_string_value( metap->as_string() );
    }
  } //end for
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::geo_polygon&  proto_poly,
                       ::kwiver::vital::geo_polygon& poly )
{
  ::kwiver::vital::polygon raw_poly;
  convert_protobuf( proto_poly.point_list(), raw_poly );
  poly.set_polygon( raw_poly, proto_poly.crs() );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::geo_polygon& poly,
                       ::kwiver::protobuf::geo_polygon&  proto_poly )
{
  proto_poly.set_crs( poly.crs() );
  convert_protobuf( poly.polygon(), *proto_poly.mutable_point_list() );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::geo_point&  proto_point,
                       ::kwiver::vital::geo_point& point )
{
  if ( proto_point.has_crs() )
  {
    ::kwiver::vital::geo_point::geo_3d_point_t pt;
    pt[0] = proto_point.x();
    pt[1] = proto_point.y();
    pt[2] = proto_point.z();
    point.set_location( pt, proto_point.crs() );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::geo_point& point,
                       ::kwiver::protobuf::geo_point&  proto_point )
{
  // test for empty
  if ( ! point.is_empty() )
  {
    proto_point.set_crs( point.crs() );
    const auto loc = point.location( point.crs() );
    proto_point.set_x( loc[0] );
    proto_point.set_y( loc[1] );
    proto_point.set_z( loc[2] );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::polygon&  proto_poly,
                       ::kwiver::vital::polygon& poly )
{
  for ( const auto vert : proto_poly.point_list() )
  {
    poly.push_back( vert.x(), vert.y() );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::polygon& poly,
                       ::kwiver::protobuf::polygon&  proto_poly )
{
  const auto vertices = poly.get_vertices();
  for ( const auto vert : vertices )
  {
    auto* proto_item = proto_poly.add_point_list();
    proto_item->set_x( vert[0] );
    proto_item->set_y( vert[1] );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::string& proto_string,
                       std::string& str )
{
  str = std::string(proto_string.data());
}

// ----------------------------------------------------------------------------
void convert_protobuf( const std::string& str,
                       ::kwiver::protobuf::string& proto_string )
{
  proto_string.set_data(str);
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::track_state& proto_trk_state,
                       ::kwiver::vital::track_state& trk_state )
{
  trk_state.set_frame( proto_trk_state.frame_id() );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::track_state& trk_state,
                        ::kwiver::protobuf::track_state& proto_trk_state)
{
  proto_trk_state.set_frame_id( static_cast< int64_t >( trk_state.frame() ) );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::track_sptr& trk_sptr,
                        ::kwiver::protobuf::track& proto_trk )
{
  proto_trk.set_track_id( trk_sptr->id() );
  for ( auto trk_state_itr=trk_sptr->begin(); trk_state_itr!=trk_sptr->end();
        ++trk_state_itr ){
    auto trk_state = *trk_state_itr;
    auto obj_trk_state_sptr = std::dynamic_pointer_cast<
                              ::kwiver::vital::object_track_state>( trk_state );
    // Check if the track state is Object Track State
    if (obj_trk_state_sptr){
      ::kwiver::protobuf::object_track_state *proto_obj_trk_state =
                                          proto_trk.add_object_track_states();
      convert_protobuf( *obj_trk_state_sptr, *proto_obj_trk_state );
    }
    else
    {
      ::kwiver::protobuf::track_state *proto_trk_state = proto_trk.add_track_states();
      convert_protobuf( *trk_state, *proto_trk_state );
    }
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::track& proto_trk,
                        ::kwiver::vital::track_sptr& trk_sptr )
{
  trk_sptr->set_id( proto_trk.track_id() );

  // Supports single type of track states
  if ( proto_trk.track_states_size() > 0 )
  {
    const size_t count( proto_trk.track_states_size() );
    for (size_t index = 0; index < count; ++index )
    {
      auto trk_state = std::make_shared< ::kwiver::vital::track_state >();
      convert_protobuf( proto_trk.track_states( index ), *trk_state );
      bool trk_inserted = trk_sptr->insert( trk_state );
      if ( !trk_inserted )
      {
        LOG_ERROR( ::kwiver::vital::get_logger( "track" ),
                 "Failed to insert track state in track." );
      }
    }
  } else if ( proto_trk.object_track_states_size() > 0 )
  {
    const size_t count( proto_trk.object_track_states_size() );
    for (size_t index = 0; index < count; ++index )
    {
      auto object_trk_state = std::make_shared< ::kwiver::vital::object_track_state >();
      convert_protobuf( proto_trk.object_track_states( index ), *object_trk_state );
      bool trk_inserted = trk_sptr->insert( object_trk_state );
      if ( !trk_inserted )
      {
        LOG_ERROR( ::kwiver::vital::get_logger( "track" ),
                 "Failed to insert object track state in track." );
      }
    }
  }
}


// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::track_set_sptr& trk_set_sptr,
                        ::kwiver::protobuf::track_set& proto_trk_set )
{
  for (::kwiver::vital::track_id_t trk_id : trk_set_sptr->all_track_ids())
  {
    ::kwiver::protobuf::track *trk = proto_trk_set.add_tracks();
    convert_protobuf( trk_set_sptr->get_track(trk_id ), *trk);
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::track_set& proto_trk_set,
                        ::kwiver::vital::track_set_sptr& trk_set_sptr )
{
  const size_t count( proto_trk_set.tracks_size() );
  for ( size_t index = 0; index < count; ++index )
  {
      auto trk = ::kwiver::vital::track::create();
      convert_protobuf( proto_trk_set.tracks( index ), trk );
      trk_set_sptr->insert( trk );
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::object_track_state& proto_obj_trk_state,
                       ::kwiver::vital::object_track_state& obj_trk_state )
{
  ::kwiver::vital::frame_id_t frame_id =  static_cast< ::kwiver::vital::frame_id_t >(
                        proto_obj_trk_state.track_state().frame_id() );
  ::kwiver::vital::time_usec_t time =  static_cast< ::kwiver::vital::time_usec_t >(
                        proto_obj_trk_state.time() );
  // object track state detection might be nullptr
  if ( !obj_trk_state.detection() )
  {
    obj_trk_state.set_detection( std::make_shared<::kwiver::vital::detected_object>(
                      ::kwiver::vital::bounding_box_d{0, 0, 0, 0} ) );
  }
  convert_protobuf( proto_obj_trk_state.detection(), *obj_trk_state.detection() );

  obj_trk_state.set_frame( frame_id );
  obj_trk_state.set_time( time );

  // image point
  ::kwiver::vital::point_2d ip;
  convert_protobuf( proto_obj_trk_state.image_point(), ip );
  obj_trk_state.set_image_point( ip );

  // track point
  ::kwiver::vital::point_3d tp;
  convert_protobuf( proto_obj_trk_state.track_point(), tp );
  obj_trk_state.set_track_point( tp );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::object_track_state& obj_trk_state,
                        ::kwiver::protobuf::object_track_state& proto_obj_trk_state )
{
  proto_obj_trk_state.set_time( obj_trk_state.time() );

  const ::kwiver::vital::track_state trk_state =
                              ::kwiver::vital::track_state( obj_trk_state.frame() );

  auto* proto_trk_state = proto_obj_trk_state.mutable_track_state();
  convert_protobuf( trk_state, *proto_trk_state );

  auto* proto_det_obj = proto_obj_trk_state.mutable_detection();
  convert_protobuf( *obj_trk_state.detection(), *proto_det_obj );

  // image point
  auto* ip = proto_obj_trk_state.mutable_image_point();
  convert_protobuf( obj_trk_state.image_point(), *ip );

  // track point
  auto* tp = proto_obj_trk_state.mutable_track_point();
  convert_protobuf( obj_trk_state.track_point(), *tp );
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::vital::object_track_set_sptr& obj_trk_set_sptr,
                        ::kwiver::protobuf::object_track_set& proto_obj_trk_set )
{
  for (::kwiver::vital::track_id_t trk_id : obj_trk_set_sptr->all_track_ids())
  {
    ::kwiver::protobuf::track *trk = proto_obj_trk_set.add_tracks();
    convert_protobuf( obj_trk_set_sptr->get_track(trk_id ), *trk);
  }
}

// ----------------------------------------------------------------------------
void convert_protobuf( const ::kwiver::protobuf::object_track_set& proto_obj_trk_set,
                        ::kwiver::vital::object_track_set_sptr& obj_trk_set_sptr )
{
  const size_t count( proto_obj_trk_set.tracks_size() );
  for ( size_t index = 0; index < count; ++index )
  {
      auto trk = ::kwiver::vital::track::create();
      convert_protobuf( proto_obj_trk_set.tracks( index ), trk );
      obj_trk_set_sptr->insert( trk );
  }
}

} } } } // end namespace
