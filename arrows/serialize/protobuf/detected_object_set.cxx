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

#include "detected_object_set.h"
#include "detected_object.h"

#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
detected_object_set::
detected_object_set()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}


detected_object_set::
~detected_object_set()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object_set::
serialize( const vital::any& element )
{
  kwiver::vital::detected_object_set_sptr dos_sptr =
    kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr > ( element );

  std::ostringstream msg;
  msg << "detected_object_set "; // add type tag

  kwiver::protobuf::detected_object_set proto_dos;
  convert_protobuf( *dos_sptr, proto_dos );

  if ( ! proto_dos.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object_set from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any detected_object_set::
deserialize( const std::string& message )
{
  auto dos_sptr = std::make_shared< kwiver::vital::detected_object_set >();
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();  // Eat delimiter

  if (tag != "detected_object_set" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object_set\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::detected_object_set proto_dos;
    if ( ! proto_dos.ParseFromIstream( &msg ) )
    {
      VITAL_THROW( kwiver::vital::serialization_exception,
                   "Error deserializing detected_object_set from protobuf" );
    }

    convert_protobuf( proto_dos, *dos_sptr );
  }

  return kwiver::vital::any(dos_sptr);
}

// ----------------------------------------------------------------------------
void detected_object_set::
convert_protobuf( const kwiver::protobuf::detected_object_set&  proto_dos,
                  kwiver::vital::detected_object_set& dos )
{
  const size_t count( proto_dos.detected_objects_size() );
  for (size_t i = 0; i < count; ++i )
  {
    auto det_object_sptr = std::make_shared< kwiver::vital::detected_object >(
      kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );
    auto proto_det_object = proto_dos.detected_objects( i );

    kwiver::arrows::serialize::protobuf::detected_object::convert_protobuf(proto_det_object,*det_object_sptr);

    dos.add( det_object_sptr );
  }
}

// ----------------------------------------------------------------------------
void detected_object_set::
convert_protobuf( const kwiver::vital::detected_object_set& dos,
                  kwiver::protobuf::detected_object_set&  proto_dos )
{
  // We're using type() in "const" (read only) way here.  There's utility
  // in having the source object parameter be const, but type() isn't because
  // its a pointer into the det_object.  Using const_cast here is a middle ground
  // though somewhat ugly
  for ( auto it: const_cast< kwiver::vital::detected_object_set& >( dos ) )
  {
    kwiver::protobuf::detected_object *proto_det_object_ptr = proto_dos.add_detected_objects();

    kwiver::arrows::serialize::protobuf::detected_object::convert_protobuf( *it, *proto_det_object_ptr );
  }
}

} } } } // end namespace
