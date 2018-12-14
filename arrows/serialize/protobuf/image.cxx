/*ckwg +30
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

#include "image.h"
#include "convert_protobuf.h"

#include <vital/types/image_container.h>
#include <vital/types/protobuf/image.pb.h>
#include <vital/exceptions.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// --------------------------------------------------------------------------
image::image()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}


image::~image()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
image::
serialize( const vital::any& element )
{
  kwiver::vital::image_container_sptr img_sptr =
    kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( element );

  std::ostringstream msg;
  msg << "image ";   // add type tag
  kwiver::protobuf::image proto_img;
  convert_protobuf( img_sptr, proto_img );


  if ( ! proto_img.SerializeToOstream( &msg ) )
  {
    VITAL_THROW( kwiver::vital::serialization_exception,
                 "Error serializing detected_object_set from protobuf" );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::any
image::
deserialize( const std::string& message )
{
  kwiver::vital::image_container_sptr img_container_sptr;
  std::istringstream msg( message );

  std::string tag;
  msg >> tag;
  msg.get();    // Eat delimiter

  if ( tag != "image" )
  {
    LOG_ERROR(
      logger(), "Invalid data type tag received. Expected \"image\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::image proto_img;
    if ( ! proto_img.ParseFromIstream(&msg) )
    {
      VITAL_THROW(kwiver::vital::serialization_exception,
                  "Error deserializing image_container from protobuf");
    }

    convert_protobuf(proto_img, img_container_sptr);
  }

  return kwiver::vital::any( img_container_sptr );
}

} } } }
