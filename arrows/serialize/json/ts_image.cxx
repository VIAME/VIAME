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

// JSON converters
#include "timestamp.h"
#include "image.h"

#include <vital/types/ts_image.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

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
  auto ts = kwiver::vital::any_cast< kwiver::vital::timestamp > ( elements.at( "timestamp" ) );
  auto image_cont = kwiver::vital::any_cast< kwiver::vital::image_container_sptr > ( elements.at( "image" ) );
  kwiver::vital::image image_v = image_cont->get_image();

  std::stringstream msg;
  msg << "ts_image "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    timestamp::save( ar, ts );
    image::save( ar, image_v );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::algo::data_serializer::deserialize_result_t
ts_image::
deserialize( std::shared_ptr< std::string > message )
{
  std::stringstream msg(*message);
  kwiver::vital::timestamp ts;
  kwiver::vital::image image_v;

  std::string tag;
  msg >> tag;

  if (tag != "ts_image" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"ts_image\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    timestamp::load( ar, ts );
    image::load( ar, image_v );
  }

  deserialize_result_t res;
  res[ "timestamp" ] = kwiver::vital::any( ts );
  res[ "image" ] = kwiver::vital::any( image_v );

  return res;
}

} } } }       // end namespace kwiver
