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

#include <vital/exceptions.h>
#include <vital/types/detected_object.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/geo_point.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/polygon.h>
#include <vital/types/timestamp.h>
#include <vital/util/hex_dump.h>

#include <vital/logger/logger.h>

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/vector.hpp>
#include <vital/internal/cereal/types/map.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <zlib.h>

namespace cereal {

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::bounding_box_d& bbox )
{
  archive( ::cereal::make_nvp( "min_x", bbox.min_x() ),
           ::cereal::make_nvp( "min_y", bbox.min_y() ),
           ::cereal::make_nvp( "max_x", bbox.max_x() ),
           ::cereal::make_nvp( "max_y", bbox.max_y() ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::bounding_box_d& bbox )
{
  double min_x, min_y, max_x, max_y;

  archive( CEREAL_NVP( min_x ),
           CEREAL_NVP( min_y ),
           CEREAL_NVP( max_x ),
           CEREAL_NVP( max_y ) );

  bbox = kwiver::vital::bounding_box_d( min_x, min_y, max_x, max_y );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::detected_object& obj )
{
  // serialize bounding box
  save( archive, obj.bounding_box() );

  archive( ::cereal::make_nvp( "confidence", obj.confidence() ),
           ::cereal::make_nvp( "index", obj.index() ),
           ::cereal::make_nvp( "detector_name", obj.detector_name() ) );

  // This pointer may be null
  const auto dot_ptr = const_cast< kwiver::vital::detected_object& >(obj).type();
  if ( dot_ptr )
  {
    save( archive, *dot_ptr );
  }
  else
  {
    kwiver::vital::detected_object_type empty_dot;
    save( archive, empty_dot );
  }
  // Currently skipping the image chip and descriptor.
  //+ TBD
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::detected_object& obj )
{
  // deserialize bounding box
  kwiver::vital::bounding_box_d bbox { 0, 0, 0, 0 };
  load( archive, bbox );
  obj.set_bounding_box( bbox );

  double confidence;
  uint64_t index;
  std::string detector_name;

  archive( CEREAL_NVP( confidence ),
           CEREAL_NVP( index ),
           CEREAL_NVP( detector_name ) );

  obj.set_confidence( confidence );
  obj.set_index( index );
  obj.set_detector_name( detector_name );

  auto new_dot = std::make_shared< kwiver::vital::detected_object_type >();
  load( archive, *new_dot );
  obj.set_type( new_dot );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive&                archive,
           const kwiver::vital::detected_object_set& obj )
{
  archive( ::cereal::make_nvp( "size", obj.size() ) );

  for ( const auto& element : const_cast< kwiver::vital::detected_object_set& >(obj) )
  {
    save( archive, *element );
  }

  // currently not handling atributes
  //+ TBD
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive&           archive,
           kwiver::vital::detected_object_set& obj )
{
  ::cereal::size_type size;
  archive( CEREAL_NVP( size ) );

  for ( ::cereal::size_type i = 0; i < size; ++i )
  {
    auto new_obj = std::make_shared< kwiver::vital::detected_object > (
      kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );
    load( archive, *new_obj );

    obj.add( new_obj );
  }

  // currently not handling atributes
  //+ TBD
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::detected_object_type& dot )
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
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::detected_object_type& dot )
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
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::image_container_sptr ctr )
{
  kwiver::vital::image vital_image = ctr->get_image();

  // Compress raw pixel data
  const uLongf size = compressBound( vital_image.size() );
  uLongf out_size(size);
  std::vector<uint8_t> image_data( size );
  Bytef* out_buf = reinterpret_cast< Bytef* >( &image_data[0] );
  Bytef const* in_buf = reinterpret_cast< Bytef * >(vital_image.memory()->data());

  int z_rc = compress( out_buf, &out_size, // outputs
                       in_buf, vital_image.size() ); // inputs
  if (Z_OK != z_rc )
  {
    switch (z_rc)
    {
    case Z_MEM_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error compressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
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
  auto pixel_trait = vital_image.pixel_traits();

  archive( ::cereal::make_nvp( "width",  vital_image.width() ),
           ::cereal::make_nvp( "height", vital_image.height() ),
           ::cereal::make_nvp( "depth",  vital_image.depth() ),

           ::cereal::make_nvp( "w_step", vital_image.w_step() ),
           ::cereal::make_nvp( "h_step", vital_image.h_step() ),
           ::cereal::make_nvp( "d_step", vital_image.d_step() ),

           ::cereal::make_nvp( "trait_type", pixel_trait.type ),
           ::cereal::make_nvp( "trait_num_bytes", pixel_trait.num_bytes ),

           ::cereal::make_nvp( "img_size", vital_image.size() ), // uncompressed size
           ::cereal::make_nvp( "img_data", image_data ) // compressed image
    );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::image_container_sptr& ctr )
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


  auto img_mem = std::make_shared< kwiver::vital::image_memory >( img_size );

  const kwiver::vital::image_pixel_traits pix_trait(
    static_cast<kwiver::vital::image_pixel_traits::pixel_type>(trait_type ),
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
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data. Not enough memory." );
      break;

    case Z_BUF_ERROR:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data. Not enough room in output buffer." );
      break;

    default:
      LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
                 "Error decompressing image data." );
      break;
    } // end switch
    return;
  }

  if ( static_cast< uLongf >(img_size) != out_size )
  {
    LOG_ERROR( kwiver::vital::get_logger( "data_serializer" ),
               "Uncompressed data not expected size. Possible data corruption." );
    return;
  }

  auto vital_image = kwiver::vital::image( img_mem, img_mem->data(),
                                           width, height, depth,
                                           w_step, h_step, d_step,
                                           pix_trait );

  // return newly constructed image container
  ctr = std::make_shared< kwiver::vital::simple_image_container >( vital_image );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive&       archive,
           const kwiver::vital::timestamp&  tstamp )
{
  archive( ::cereal::make_nvp( "time", tstamp.get_time_usec() ),
           ::cereal::make_nvp( "frame", tstamp.get_frame() ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive&  archive,
           kwiver::vital::timestamp&  tstamp )
{
  int64_t time, frame;

  archive( CEREAL_NVP( time ),
           CEREAL_NVP( frame ) );
  tstamp = kwiver::vital::timestamp( static_cast< kwiver::vital::time_usec_t > ( time ),
                                     static_cast< kwiver::vital::frame_id_t > (
                                       frame ) );
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::geo_polygon& poly )
{
  archive( ::cereal::make_nvp( "crs", poly.crs() ) );
  save( archive, poly.polygon() ); // save plain polygon
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::geo_polygon& poly )
{
  int crs;
  kwiver::vital::polygon raw_poly;

  archive( CEREAL_NVP( crs ) );
  load( archive, raw_poly ); // load raw polygon
  poly.set_polygon( raw_poly, crs );
}


// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::geo_point& point )
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
             ::cereal::make_nvp( "y", loc[1] )
      );
  }
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::geo_point& point )
{
  int crs;
  archive( CEREAL_NVP(crs) );

  if ( crs != -1 ) // empty marker
  {
    double x, y;
    archive( CEREAL_NVP( crs ),
             CEREAL_NVP( x ),
             CEREAL_NVP( y )
      );

    const kwiver::vital::geo_point::geo_raw_point_t raw( x, y );
    point.set_location( raw, crs );
  }
}

// ============================================================================
void save( ::cereal::JSONOutputArchive& archive, const kwiver::vital::polygon& poly )
{
  auto vert = poly.get_vertices();
  archive( ::cereal::make_nvp( "points", vert ) );
}

// ----------------------------------------------------------------------------
void load( ::cereal::JSONInputArchive& archive, kwiver::vital::polygon& poly )
{
  std::vector< kwiver::vital::polygon::point_t > points;
  archive( CEREAL_NVP( points ) );

  for ( const auto pt : points )
  {
    poly.push_back( pt );
  }
}

} // end namespace
