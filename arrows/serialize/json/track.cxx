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

#include "track.h"

#include "load_save.h"
#include "load_save_track_state.h"
#include "track_item.h"

#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>
#include <vital/internal/cereal/types/utility.hpp>

#include <sstream>
#include <iostream>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {
// ----------------------------------------------------------------------------
track::
track()
{ }


track::
~track()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
track::
serialize( const vital::any& element )
{
  kwiver::vital::track_sptr trk_sptr =
    kwiver::vital::any_cast< kwiver::vital::track_sptr > ( element );
  kwiver::arrows::serialize::json::track_item trk_item(trk_sptr);
  std::stringstream msg;
  msg << "track "; // add type tag
  {
    cereal::JSONOutputArchive ar( msg );
    ar( trk_item );
  }
  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::any track::
deserialize( const std::string& message )
{
  std::stringstream msg(message);
  kwiver::arrows::serialize::json::track_item trk_item = track_item();
  std::string tag;
  msg >> tag;

  if (tag != "track" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"track\", received \""
               << tag << "\". Message dropped, returning default object." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    ar( trk_item );
  }
  return kwiver::vital::any( trk_item.get_track() );
}


} } } }       // end namespace kwiver
