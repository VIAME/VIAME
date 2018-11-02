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

#include "bounding_box.h"
#include "convert_protobuf.h"

#include <vital/types/bounding_box.h>
#include <vital/types/protobuf/bounding_box.pb.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
bounding_box::
bounding_box()
{
  // Verify that the version of the library that we linked against is
  // compatible with the version of the headers we compiled against.
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}


bounding_box::
~bounding_box()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
bounding_box::
serialize( const vital::any& element )
{
  kwiver::vital::bounding_box_d bbox =
    kwiver::vital::any_cast< kwiver::vital::bounding_box_d > ( element );

  std::ostringstream msg;
  msg << "bounding_box "; // add type tag

  kwiver::protobuf::bounding_box proto_bbox;
  convert_protobuf( bbox, proto_bbox );

  if ( ! proto_bbox.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_bbox.SerializeToOStream failed" );
  }

  return std::make_shared< std::string > ( msg.str() );
}

// ----------------------------------------------------------------------------
vital::any bounding_box::
deserialize( const std::string& message )
{
  kwiver::vital::bounding_box_d bbox{ 0, 0, 0, 0 };

  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();  // Eat the delimiter

  if (tag != "bounding_box" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"bounding_box\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    // define our protobuf
    kwiver::protobuf::bounding_box proto_bbox;
    if ( ! proto_bbox.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly. ParseFromIstream failed." );
    }

    convert_protobuf( proto_bbox, bbox );
  }

  return kwiver::vital::any(bbox);
}

} } } } // end namespace
