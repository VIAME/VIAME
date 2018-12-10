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

#include "timestamp.h"
#include "convert_protobuf.h"

#include <vital/types/timestamp.h>
#include <vital/types/protobuf/timestamp.pb.h>
#include <vital/exceptions.h>

#include <cstdint>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ----------------------------------------------------------------------------
timestamp::timestamp()
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
}

timestamp::~timestamp()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
timestamp::
serialize( const vital::any& element )
{
  kwiver::vital::timestamp tstamp = kwiver::vital::any_cast< kwiver::vital::timestamp > ( element );
  std::ostringstream msg;
  msg << "timestamp ";
  kwiver::protobuf::timestamp proto_tstamp;

  convert_protobuf( tstamp, proto_tstamp );

  if ( ! proto_tstamp.SerializeToOstream( &msg ) )
  {
    LOG_ERROR( logger(), "proto_timestamp.SerializeToOStream failed" );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::any
timestamp::deserialize( const std::string& message )
{
  kwiver::vital::timestamp tstamp;
  std::istringstream msg( message );
  std::string tag;
  msg >> tag;
  msg.get();
  if ( tag != "timestamp" )
  {
    LOG_ERROR( logger(), "Invalid data type tag receiver. Expected timestamp"
               << "received " << tag << ". Message dropped." );
  }
  else
  {
    kwiver::protobuf::timestamp proto_tstamp;
    if ( ! proto_tstamp.ParseFromIstream( &msg ) )
    {
      LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly."
                 << "ParseFromIstream failed." );
    }

    convert_protobuf( proto_tstamp, tstamp );
  }

  return kwiver::vital::any( tstamp );
}

} } } }
