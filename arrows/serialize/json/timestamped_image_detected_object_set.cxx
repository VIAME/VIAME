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
#include "timestamped_image_detected_object_set.h"

//JSON convertors
#include "timestamp.h"
#include "detected_object_set.h"
#include "image.h"

#include <vital/types/timestamp.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>


namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {
  timestamped_image_detected_object_set::timestamped_image_detected_object_set()
  {
    m_element_names.insert( "timestamp" );
    m_element_names.insert( "detected_object_set" );
    m_element_names.insert( "image" );
  }

  timestamped_image_detected_object_set::~timestamped_image_detected_object_set()
  { }

  std::shared_ptr< std::string> timestamped_image_detected_object_set
    ::serialize( const data_serializer::serialize_param_t& elements )
  {
    if ( ! check_element_names( elements ))
    {
      // error TBD 
    }
    
    auto tstamp = kwiver::vital::any_cast< kwiver::vital::timestamp >( elements.
                                                          at( "timestamp" ) );
    auto dos = kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr >( 
                   elements.at( "detected_object_set" ) );
    auto img_ctr_sptr = kwiver::vital::any_cast< kwiver::vital::image_container_sptr >(
                   elements.at( "image" ) );

    std::stringstream msg;
    msg << "timestamped_image_detected_object_set ";
    {
      cereal::JSONOutputArchive ar( msg );
      timestamp::save( ar, tstamp );
      detected_object_set::save( ar, *dos );
      image::save( ar, img_ctr_sptr->get_image() );
    }
    
    return std::make_shared< std::string > (msg.str());
  }

  vital::algo::data_serializer::deserialize_result_t 
  timestamped_image_detected_object_set::deserialize( 
      std::shared_ptr< std::string > message)
  {
    deserialize_result_t res;
    std::istringstream msg( *message );
    kwiver::vital::timestamp tstamp;
    kwiver::vital::detected_object_set* dos = new kwiver::vital::detected_object_set();
    kwiver::vital::image img;

    std::string tag;
    msg >> tag;
    
    if (tag != "timestamped_image_detected_object_set")
    {
      LOG_ERROR( logger(), "Invalid data type tag received." << 
         "Expected \"timestamped_image_detected_object_set\", received " << tag);
    }
    else
    {
      cereal::JSONInputArchive ar( msg );
      timestamp::load( ar, tstamp );
      detected_object_set::load( ar, *dos ); 
      image::load( ar, img );
    }

    kwiver::vital::image_container_sptr img_ctr_sptr = 
            std::make_shared< kwiver::vital::simple_image_container >( img ); 
                        
    res[ "timestamp" ] = kwiver::vital::any( tstamp );
    res[ "detected_object_set" ] = kwiver::vital::any( 
                                kwiver::vital::detected_object_set_sptr(dos) );
    res[ "image" ] = kwiver::vital::any( img_ctr_sptr );
    return res; 
  }
} 
}
}
}
