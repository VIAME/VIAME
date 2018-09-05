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

#include "detected_object.h"
#include "detected_object_type.h"
#include "bounding_box.h"

#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
detected_object::
detected_object()
{
  m_element_names.insert( DEFAULT_ELEMENT_NAME );

  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}


detected_object::
~detected_object()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object::
serialize( const data_serializer::serialize_param_t& elements )
{
  kwiver::vital::detected_object det_object =
    kwiver::vital::any_cast< kwiver::vital::detected_object > ( elements.at( DEFAULT_ELEMENT_NAME ) );

  std::ostringstream msg;
  msg << "detected_object "; // add type tag

  kwiver::protobuf::detected_object proto_det_object;
  convert_protobuf( det_object, proto_det_object );

  if ( ! proto_det_object.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::algo::data_serializer::deserialize_result_t
detected_object::
deserialize( std::shared_ptr< std::string > message )
{
  std::istringstream msg( *message );
  auto det_object_ptr = std::make_shared< kwiver::vital::detected_object >( kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "detected_object" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::detected_object proto_det_object;
    if ( ! proto_det_object.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_object from protobuf" );
    }

    convert_protobuf( proto_det_object, *det_object_ptr );
  }

  deserialize_result_t res;
  res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any( det_object_ptr );

  return res;
}

// ----------------------------------------------------------------------------
void detected_object::
convert_protobuf( const kwiver::protobuf::detected_object&  proto_det_object,
                  kwiver::vital::detected_object& det_object )
{
  det_object.set_confidence( proto_det_object.confidence() );

  kwiver::vital::bounding_box_d bbox{ 0, 0, 0, 0 };
  kwiver::protobuf::bounding_box proto_bbox = proto_det_object.bbox();
  kwiver::arrows::serialize::protobuf::bounding_box::convert_protobuf( proto_bbox, bbox );
  det_object.set_bounding_box( bbox );

  if ( proto_det_object.has_classifcations() )
  {
    auto new_dot = std::make_shared< kwiver::vital::detected_object_type >();
    kwiver::protobuf::detected_object_type proto_dot = proto_det_object.classifcations();
    kwiver::arrows::serialize::protobuf::detected_object_type::convert_protobuf( proto_dot, *new_dot );
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
}

// ----------------------------------------------------------------------------
void detected_object::
convert_protobuf( const kwiver::vital::detected_object& det_object,
                  kwiver::protobuf::detected_object&  proto_det_object )
{
  proto_det_object.set_confidence( det_object.confidence() );

  kwiver::protobuf::bounding_box *proto_bbox = new kwiver::protobuf::bounding_box();
  kwiver::arrows::serialize::protobuf::bounding_box::convert_protobuf( det_object.bounding_box(), *proto_bbox );

  proto_det_object.set_allocated_bbox(proto_bbox);  // proto_det_object takes ownership

  // We're using type() in "const" (read only) way here.  There's utility
  // in having the source object parameter be const, but type() isn't because
  // its a pointer into the det_object.  Using const_cast here is a middle ground
  // though somewhat ugly
  if ( const_cast<kwiver::vital::detected_object&>(det_object).type() != NULL )
  {
    kwiver::protobuf::detected_object_type *proto_dot = new kwiver::protobuf::detected_object_type();
    kwiver::arrows::serialize::protobuf::detected_object_type::
      convert_protobuf( * const_cast<kwiver::vital::detected_object&>(det_object).type(), *proto_dot );

    proto_det_object.set_allocated_classifcations(proto_dot); // proto_det_object takes ownership
  }

  proto_det_object.set_index( det_object.index() );

  proto_det_object.set_detector_name( det_object.detector_name() );
}

} } } } // end namespace
