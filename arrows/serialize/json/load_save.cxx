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

#include <arrows/serialize/json/load_save.h>
#include <arrows/serialize/json/load_save_point.h>
#include <arrows/serialize/json/load_save_track_state.h>
#include <arrows/serialize/json/load_save_track_set.h>
#include <arrows/serialize/json/track_item.h>

#include <vital/exceptions.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/geo_point.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/polygon.h>
#include <vital/types/timestamp.h>
#include <vital/types/track_set.h>
#include <vital/types/object_track_set.h>
#include <vital/vital_types.h>
#include <vital/util/hex_dump.h>

#include <vital/logger/logger.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/types/map.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <zlib.h>
#include <iostream>


namespace cereal {

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::bounding_box_d& bbox )
{
  archive( ::cereal::make_nvp( "min_x", bbox.min_x() ),
           ::cereal::make_nvp( "min_y", bbox.min_y() ),
           ::cereal::make_nvp( "max_x", bbox.max_x() ),
           ::cereal::make_nvp( "max_y", bbox.max_y() ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::bounding_box_d& bbox )
{
  double min_x, min_y, max_x, max_y;

  archive( CEREAL_NVP( min_x ),
           CEREAL_NVP( min_y ),
           CEREAL_NVP( max_x ),
           CEREAL_NVP( max_y ) );

  bbox = ::kwiver::vital::bounding_box_d( min_x, min_y, max_x, max_y );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::detected_object& obj )
{
  // serialize bounding box
  save( archive, obj.bounding_box() );

  archive( ::cereal::make_nvp( "confidence", obj.confidence() ),
           ::cereal::make_nvp( "index", obj.index() ),
           ::cereal::make_nvp( "detector_name", obj.detector_name() ),
           ::cereal::make_nvp( "notes", obj.notes() ),
           ::cereal::make_nvp( "keypoints", obj.keypoints() ),
           ::cereal::make_nvp( "geo_point", obj.geo_point() )
    );

  // This pointer may be null
  const auto dot_ptr = const_cast< ::kwiver::vital::detected_object& >(obj).type();
  if ( dot_ptr )
  {
    save( archive, *dot_ptr );
  }
  else
  {
    ::kwiver::vital::detected_object_type empty_dot;
    save( archive, empty_dot );
  }

  // Currently skipping the image chip and descriptor.
  //+ TBD
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::detected_object& obj )
{
  // deserialize bounding box
  ::kwiver::vital::bounding_box_d bbox { 0, 0, 0, 0 };
  load( archive, bbox );
  obj.set_bounding_box( bbox );

  double confidence;
  uint64_t index;
  std::string detector_name;
  ::kwiver::vital::detected_object::notes_t notes;
  ::kwiver::vital::detected_object::keypoints_t keypoints;
  ::kwiver::vital::geo_point geo_point;

  archive( CEREAL_NVP( confidence ),
           CEREAL_NVP( index ),
           CEREAL_NVP( detector_name ),
           CEREAL_NVP( notes ),
           CEREAL_NVP( keypoints ),
           CEREAL_NVP( geo_point )
    );

  obj.set_confidence( confidence );
  obj.set_index( index );
  obj.set_detector_name( detector_name );

  for ( const auto& n : notes )
  {
    obj.add_note( n );
  }

  obj.set_geo_point( geo_point );

  for ( const auto& kp : keypoints )
  {
    obj.add_keypoint( kp.first, kp.second );
  }

  auto new_dot = std::make_shared< ::kwiver::vital::detected_object_type >();
  load( archive, *new_dot );
  obj.set_type( new_dot );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive&                archive,
           const ::kwiver::vital::detected_object_set& obj )
{
  archive( ::cereal::make_nvp( "size", obj.size() ) );

  using dos = ::kwiver::vital::detected_object_set;

  dos::const_iterator ie = obj.cend();
  dos::const_iterator element;

  for ( element = obj.cbegin(); element != ie; ++element )
  {
    save( archive, **element );
  }

  // currently not handling atributes
  //+ TBD
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive&           archive,
           ::kwiver::vital::detected_object_set& obj )
{
  ::cereal::size_type size;
  archive( CEREAL_NVP( size ) );

  for ( ::cereal::size_type i = 0; i < size; ++i )
  {
    auto new_obj = std::make_shared< ::kwiver::vital::detected_object > (
      ::kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );
    load( archive, *new_obj );

    obj.add( new_obj );
  }

  // currently not handling atributes
  //+ TBD
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::detected_object_type& dot )
{

  // recreate the class/score map so we don't break encapsulation.
  std::map< std::string, double > class_map;
  for ( auto entry : dot )
  {
    class_map[*(entry.first)] = entry.second;
  }

  archive( CEREAL_NVP( class_map ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::detected_object_type& dot )
{
  std::map< std::string, double > class_map;
  archive( CEREAL_NVP( class_map ) );

  for ( auto entry : class_map )
  {
    dot.set_score( entry.first, entry.second );
  }
}

// ----------------------------------------------------------------------------
/*
 * We still may have a problem in this serialization approach if the
 * two end point have different byte ordering.
 *
 * Since the image is compressed as a byte string, all notion of
 * multi-byte pixels is lost when the image is saved. When the image
 * is loaded, there is no indication of byte ordering from the
 * source. The sender may have the same byte ordering or it may be
 * different. There is no way of telling.
 *
 * Options:
 *
 * 1) Save image as a vector of correct pixel data types. This
 *    approach would preclude compression.
 *
 * 2) Write an uncompressed integer or other indicator into the
 *    stream which the receiver can use to determine the senders
 *    byte ordering and decode appropriately.
 *
 * 3) Refuse to serialize images with multi-byte pixels or say it
 *    does not work between systems with different byte ordering.
 *
 * 4) Use network byte ordering. ( htonl(), ntohl() )
 *
 * You can argue that endianess is a message property, but we can't
 * access message level attributes way down here in the serializer.
 *
 * Note that this is only a problem with images where the pixels are
 * multi-byte.
 */

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::image_container_sptr ctr )
{
  ::kwiver::vital::image vital_image = ctr->get_image();
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
  const uLongf size = compressBound( vital_image.size() );
  uLongf out_size(size);
  std::vector<uint8_t> image_data( size );
  Bytef* out_buf = reinterpret_cast< Bytef* >( &image_data[0] );
  Bytef const* in_buf = reinterpret_cast< Bytef * >( local_image.first_pixel() );

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
      LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data." );
      break;
    } // end switch
    return;
  }

  // Copy compressed image to vector
  uint8_t* cp = static_cast< uint8_t* >(out_buf );
  uint8_t* cpe = cp +out_size;
  image_data.assign( cp, cpe );

  // Get pixel trait
  auto pixel_trait = local_image.pixel_traits();

  archive( ::cereal::make_nvp( "width",  local_image.width() ),
           ::cereal::make_nvp( "height", local_image.height() ),
           ::cereal::make_nvp( "depth",  local_image.depth() ),

           ::cereal::make_nvp( "w_step", local_image.w_step() ),
           ::cereal::make_nvp( "h_step", local_image.h_step() ),
           ::cereal::make_nvp( "d_step", local_image.d_step() ),

           ::cereal::make_nvp( "trait_type", static_cast<int> (pixel_trait.type) ),
           ::cereal::make_nvp( "trait_num_bytes", pixel_trait.num_bytes ),

           ::cereal::make_nvp( "img_size", local_image.size() ), // uncompressed size
           ::cereal::make_nvp( "img_data", image_data ) );       // compressed image

}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::image_container_sptr& ctr )
{
  // deserialize image
  std::size_t width, height, depth, img_size;
  std::ptrdiff_t w_step, h_step, d_step;
  std::vector<uint8_t> img_data;
  int trait_type, trait_num_bytes;

  archive( CEREAL_NVP( width ),
           CEREAL_NVP( height ),
           CEREAL_NVP( depth ),

           CEREAL_NVP( w_step ),
           CEREAL_NVP( h_step ),
           CEREAL_NVP( d_step ),

           CEREAL_NVP( trait_type ),
           CEREAL_NVP( trait_num_bytes ),

           CEREAL_NVP( img_size ), // uncompressed size
           CEREAL_NVP( img_data )  // compressed image
    );


  auto img_mem = std::make_shared< ::kwiver::vital::image_memory >( img_size );

  const ::kwiver::vital::image_pixel_traits pix_trait(
    static_cast<::kwiver::vital::image_pixel_traits::pixel_type>(trait_type ),
    trait_num_bytes );

  // decompress the data
  Bytef* out_buf = reinterpret_cast< Bytef* >(img_mem->data());
  uLongf out_size( img_size );
  Bytef const* in_buf = reinterpret_cast< const Bytef *> (&img_data[0] );
  uLongf in_size = img_data.size(); // byte size

  int z_rc = uncompress( out_buf, &out_size, // outputs
                         in_buf, in_size ); // inputs
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

  if ( static_cast< uLongf >(img_size) != out_size )
  {
    LOG_ERROR( ::kwiver::vital::get_logger( "data_serializer" ),
               "Uncompressed data not expected size. Possible data corruption." );
    return;
  }

  auto vital_image = ::kwiver::vital::image( img_mem, img_mem->data(),
                                           width, height, depth,
                                           w_step, h_step, d_step,
                                           pix_trait );

  // return newly constructed image container
  ctr = std::make_shared< ::kwiver::vital::simple_image_container >( vital_image );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive&       archive,
           const ::kwiver::vital::timestamp&  tstamp )
{
  archive( ::cereal::make_nvp( "time", tstamp.get_time_usec() ),
           ::cereal::make_nvp( "frame", tstamp.get_frame() ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive&  archive,
           ::kwiver::vital::timestamp&  tstamp )
{
  int64_t time, frame;

  archive( CEREAL_NVP( time ),
           CEREAL_NVP( frame ) );
  tstamp = ::kwiver::vital::timestamp( static_cast< ::kwiver::vital::time_usec_t > ( time ),
                                     static_cast< ::kwiver::vital::frame_id_t > (
                                       frame ) );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::geo_polygon& poly )
{
  archive( ::cereal::make_nvp( "crs", poly.crs() ) );
  save( archive, poly.polygon() ); // save plain polygon
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::geo_polygon& poly )
{
  int crs;
  ::kwiver::vital::polygon raw_poly;

  archive( CEREAL_NVP( crs ) );
  load( archive, raw_poly ); // load raw polygon
  poly.set_polygon( raw_poly, crs );
}


// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::geo_point& point )
{
  if ( point.is_empty() )
  {
    const int crs(-1);
    archive( CEREAL_NVP(crs) );
  }
  else
  {
    const auto loc = point.location( point.crs() );

    archive( ::cereal::make_nvp( "crs", point.crs() ),
             ::cereal::make_nvp( "x", loc[0] ),
             ::cereal::make_nvp( "y", loc[1] ),
             ::cereal::make_nvp( "z", loc[2] )
      );
  }
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::geo_point& point )
{
  int crs;
  archive( CEREAL_NVP(crs) );

  if ( crs != -1 ) // empty marker
  {
    double x, y, z;
    archive( CEREAL_NVP( crs ),
             CEREAL_NVP( x ),
             CEREAL_NVP( y ),
             CEREAL_NVP( z )
      );

    const ::kwiver::vital::geo_point::geo_3d_point_t raw( x, y, z );
    point.set_location( raw, crs );
  }
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::polygon& poly )
{
  auto vert = poly.get_vertices();
  archive( ::cereal::make_nvp( "points", vert ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::polygon& poly )
{
  std::vector< ::kwiver::vital::polygon::point_t > points;
  archive( CEREAL_NVP( points ) );

  for ( const auto pt : points )
  {
    poly.push_back( pt );
  }
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::track_state& trk_state )
{
  ::kwiver::vital::frame_id_t frame = trk_state.frame();
  archive(  ::cereal::make_nvp( "track_frame", frame ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::track_state& trk_state )
{
  int64_t track_frame;
  archive( CEREAL_NVP( track_frame ) );
  trk_state.set_frame( track_frame );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive,
           const ::kwiver::vital::object_track_state& obj_trk_state )
{
  archive( ::cereal::base_class< ::kwiver::vital::track_state >( std::addressof( obj_trk_state ) ) );
  archive( ::cereal::make_nvp( "track_time", obj_trk_state.time() ) );
  archive( ::cereal::make_nvp( "image_point", obj_trk_state.image_point() ) );
  archive( ::cereal::make_nvp( "track_point", obj_trk_state.track_point() ) );

  save( archive, *obj_trk_state.detection() );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive,
           ::kwiver::vital::object_track_state& obj_trk_state )
{
  auto detection = std::make_shared< ::kwiver::vital::detected_object >(
                      ::kwiver::vital::bounding_box_d{0, 0, 0, 0});
  archive( ::cereal::base_class< ::kwiver::vital::track_state >( std::addressof( obj_trk_state ) ) );

  int64_t track_time;
  archive( CEREAL_NVP( track_time ) );
  obj_trk_state.set_time(track_time);

  // image point
  ::kwiver::vital::point_2d image_point;
  archive( CEREAL_NVP( image_point ) );
  obj_trk_state.set_image_point( image_point );

  // track point
  ::kwiver::vital::point_3d track_point;
  archive( CEREAL_NVP( track_point ) ) ;
  obj_trk_state.set_track_point( track_point );

  load(archive, *detection);
  obj_trk_state.set_detection(detection);
}


// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::track_set& trk_set )
{
  std::vector<::kwiver::arrows::serialize::json::track_item> track_items;
  for ( auto trk_sptr : trk_set.tracks())
  {
    track_items.push_back( ::kwiver::arrows::serialize::json::track_item( trk_sptr ) );
  }
  archive( ::cereal::make_nvp( "trk_items", track_items ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::track_set& trk_set )
{
  std::vector< ::kwiver::arrows::serialize::json::track_item > trk_items;
  archive ( CEREAL_NVP(trk_items) );
  std::vector< ::kwiver::vital::track_sptr > tracks;
  for (auto trk_item : trk_items)
  {
    tracks.push_back(trk_item.get_track());
  }
  trk_set.set_tracks(tracks);
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive,
          const ::kwiver::vital::object_track_set& obj_trk_set )
{
  //TBD: Inheritance is not working between track set and object track set
  // Causes the object associated with the track set to be a list
  //
  //archive( ::cereal::base_class< ::kwiver::vital::track_set >( std::addressof( obj_trk_set ) ) );
  std::vector<::kwiver::arrows::serialize::json::track_item> track_items;
  for ( auto trk_sptr : obj_trk_set.tracks())
  {
    track_items.push_back( ::kwiver::arrows::serialize::json::track_item( trk_sptr ) );
  }
  archive( ::cereal::make_nvp( "object_trk_items", track_items ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive,
            ::kwiver::vital::object_track_set& obj_trk_set )
{
  //archive( ::cereal::base_class< ::kwiver::vital::object_track_set >( std::addressof( obj_trk_set ) ) );
  std::vector< ::kwiver::arrows::serialize::json::track_item > object_trk_items;
  archive ( CEREAL_NVP(object_trk_items) );
  std::vector< ::kwiver::vital::track_sptr > object_tracks;
  for (auto object_trk_item : object_trk_items)
  {
    object_tracks.push_back(object_trk_item.get_track());
  }
  obj_trk_set.set_tracks(object_tracks);
}

} // end namespace
