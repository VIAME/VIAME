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

#include "ts_image.h"
#include "protobuf_util.h"

// protobuf converters
#include "timestamp.h"
#include "image.h"

namespace kasp = kwiver::arrows::serialize::protobuf;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
ts_image::
ts_image()
{
  m_element_names.insert( "timestamp" );
  m_element_names.insert( "image" );
}


ts_image::
~ts_image()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
ts_image::
serialize( const data_serializer::serialize_param_t& elements )
{
  // Make sure the expected elements names are present
  if ( ! check_element_names( elements ) ) // throws
  {
    // error TBD
  }

  std::ostringstream msg;
  msg << "ts_image "; // add type tag

  auto ts = kwiver::vital::any_cast< kwiver::vital::timestamp > ( elements.at( "timestamp" ) );
  kwiver::protobuf::timestamp proto_timestamp;
  kasp::timestamp::convert_protobuf( ts, proto_timestamp );
  add_proto_to_stream( msg, proto_timestamp );

  auto image_cont = kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( elements.at( "image" ) );
  kwiver::vital::image image_v = image_cont->get_image();
  kwiver::protobuf::timestamp proto_image;
  kasp::image::convert_protobuf( image_v, proto_image );
  add_proto_to_stream( msg, proto_image );

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::algo::data_serializer::deserialize_result_t
ts_image::
deserialize( std::shared_ptr< std::string > message )
{
  deserialize_result_t res;

  std::istringstream msg( *message );
  std::string tag;
  msg >> tag;
  msg.get();  // Eat the delimiter

  if (tag != "ts_image" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"ts_image\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    auto str_timestamp = grab_proto_from_stream( msg );
    kwiver::protobuf::timestamp proto_timestamp;
    if ( ! proto_timestamp.ParseFromString( &str_timestamp ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly. ParseFromString failed." );
    }
    kwiver::vital::timestamp ts;
    kasp::timestamp::convert_protobuf( proto_timestamp, ts );
    res[ "timestamp" ] = kwiver::vital::any( ts );

    auto str_image = grab_proto_from_stream( msg );
    kwiver::protobuf::timestamp proto_image;
    if ( ! proto_image.ParseFromString( &str_image ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly. ParseFromString failed." );
    }
    kwiver::vital::image image_v;
    kasp::timestamp::convert_protobuf( proto_image, image_v );
    kwiver::vital::image_container_sptr image = std::make_shared< kwiver::vital::image_container>( image_v );
    res[ "image" ] = kwiver::vital::any( image_v );
  }

  return res;
}

} } } }       // end namespace kwiver
