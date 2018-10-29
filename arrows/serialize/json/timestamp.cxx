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
#include <vital/types/timestamp.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>
#include <cstdint>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
timestamp::timestamp()
{ }

timestamp::~timestamp()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
timestamp
::serialize( const vital::any& element )
{
  kwiver::vital::timestamp tstamp =
    kwiver::vital::any_cast< kwiver::vital::timestamp > ( element );
  std::stringstream msg;
  msg << "timestamp ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, tstamp );
  }
  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::any
timestamp
::deserialize( const std::string& message )
{
  std::stringstream msg( message );
  kwiver::vital::timestamp tstamp;

  std::string tag;
  msg >> tag;
  if ( tag != "timestamp" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"timestamp\""
               << ",  received \"" << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, tstamp );
  }

  return kwiver::vital::any( tstamp );
}


// ----------------------------------------------------------------------------
void
timestamp::save( cereal::JSONOutputArchive&       archive,
                 const kwiver::vital::timestamp&  tstamp )
{
  archive( cereal::make_nvp( "time", tstamp.get_time_usec() ),
           cereal::make_nvp( "frame", tstamp.get_frame() ) );
}


// ----------------------------------------------------------------------------
void
timestamp::load( cereal::JSONInputArchive&  archive,
                 kwiver::vital::timestamp&  tstamp )
{
  int64_t time, frame;

  archive( CEREAL_NVP( time ),
           CEREAL_NVP( frame ) );
  tstamp = kwiver::vital::timestamp( static_cast< kwiver::vital::time_usec_t > ( time ),
                                     static_cast< kwiver::vital::frame_id_t > (
                                       frame ) );
}


} } } }
