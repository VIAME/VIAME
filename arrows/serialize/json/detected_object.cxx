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

#include <vital/types/detected_object.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>

namespace kasj = kwiver::arrows::serialize::json;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

// ----------------------------------------------------------------------------
detected_object::
detected_object()
{
  m_element_names.insert( DEFAULT_ELEMENT_NAME );
}


detected_object::
~detected_object()
{ }

// ----------------------------------------------------------------------------
std::shared_ptr< std::string >
detected_object::
serialize( const serialize_param_t elements )
{
  // Get native data type from any
  kwiver::vital::detected_object_sptr obj =
    kwiver::vital::any_cast< kwiver::vital::detected_object_sptr > ( elements.at( DEFAULT_ELEMENT_NAME ) );

  std::stringstream msg;
  msg << "detected_object ";
  {
    cereal::JSONOutputArchive ar( msg );
    save( ar, *obj );
  }

  return std::make_shared< std::string > ( msg.str() );
}


// ----------------------------------------------------------------------------
vital::algo::data_serializer::deserialize_result_t
detected_object::
deserialize( std::shared_ptr< std::string > message )
{
  std::stringstream msg(*message);
  auto obj = std::make_shared< kwiver::vital::detected_object >( kwiver::vital::bounding_box_d { 0, 0, 0, 0 } );

  std::string tag;
  msg >> tag;

  if (tag != "detected_object" )
  {
    LOG_ERROR( logger(), "Invalid data type tag received. Expected \"detected_object\", received \""
               << tag << "\". Message dropped." );
  }
  else
  {
    cereal::JSONInputArchive ar( msg );
    load( ar, *obj );
  }

  deserialize_result_t res;
  res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any(obj);

  return res;
}

// ----------------------------------------------------------------------------
void
detected_object::
save( cereal::JSONOutputArchive& archive, const kwiver::vital::detected_object& obj )
{
  // serialize bounding box
  kasj::bounding_box::save( archive, obj.bounding_box() );

  archive( cereal::make_nvp( "confidence", obj.confidence() ),
           cereal::make_nvp( "index", obj.index() ),
           cereal::make_nvp( "detector_name", obj.detector_name() ) );

  // This pointer may be null
  const auto dot_ptr = const_cast< kwiver::vital::detected_object& >(obj).type();
  if ( dot_ptr )
  {
    kasj::detected_object_type::save( archive, *dot_ptr );
  }
  else
  {
    kwiver::vital::detected_object_type empty_dot;
    kasj::detected_object_type::save( archive, empty_dot );
  }
  // Currently skipping the image chip and descriptor.
  //+ TBD
}

// ----------------------------------------------------------------------------
void
detected_object::
load( cereal::JSONInputArchive& archive, kwiver::vital::detected_object& obj )
{
  // deserialize bounding box
  kwiver::vital::bounding_box_d bbox { 0, 0, 0, 0 };
  kasj::bounding_box::load( archive, bbox );
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
  kasj::detected_object_type::load( archive, *new_dot );
  obj.set_type( new_dot );
}

} } } }       // end namespace kwiver
