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
#include "timestamped_detected_object_set.h"
#include "protobuf_util.h"

// protobuf convertors
#include "timestamp.h"
#include "detected_object_set.h"

namespace kasp = kwiver::arrows::serialize::protobuf;

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {
  timestamped_detected_object_set::timestamped_detected_object_set()
  {
    m_element_names.insert( "timestamp" );
    m_element_names.insert( "detected_object_set" );
    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }

  timestamped_detected_object_set::~timestamped_detected_object_set()
  { }

  std::shared_ptr< std::string> timestamped_detected_object_set
    ::serialize( const data_serializer::serialize_param_t& elements )
  {
    if ( ! check_element_names( elements ))
    {
      // error TBD 
    }

    std::ostringstream msg;
    msg << "timestamped_detected_object_set" << " ";
    
    kwiver::vital::timestamp tstamp = 
                kwiver::vital::any_cast< kwiver::vital::timestamp >(elements
                                                          .at( "timestamp" ));
    kwiver::protobuf::timestamp proto_tstamp;
    kasp::timestamp::convert_protobuf( tstamp, proto_tstamp);

    add_proto_to_stream( msg, proto_tstamp );
    
    auto dos = kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr > 
                  (elements.at("detected_object_set"));
    kwiver::protobuf::detected_object_set proto_dos;
    kasp::detected_object_set::convert_protobuf( *dos, proto_dos);
    add_proto_to_stream( msg, proto_dos );
    return std::make_shared< std::string > (msg.str());
  }

  vital::algo::data_serializer::deserialize_result_t 
  timestamped_detected_object_set::deserialize( 
      std::shared_ptr< std::string > message)
  {
    deserialize_result_t res;
    std::istringstream msg( *message );
    //std::cout << "Receiving: " << kwiver::vital::hexDump(msg.str().data(), 
    //            msg.str().length()) <<std::endl; 
    std::string tag;
    msg >> tag;
    msg.get();
    if (tag != "timestamped_detected_object_set")
    {
      LOG_ERROR( logger(), "Invalid data type tag received." << 
         "Expected \"timestamped_detected_object_set\", received " << tag);
    }
    else
    {
      auto str_tstamp = grab_proto_from_stream( msg );
      kwiver::protobuf::timestamp proto_tstamp; 
      if ( ! proto_tstamp.ParseFromString( *str_tstamp) )
      {
       LOG_ERROR( logger(), "TimeStamp protobuf stream didn't parse correctly" << 
           "ParseFromString failed");
       LOG_ERROR( logger(), "message string: " << *str_tstamp);
      }
      kwiver::vital::timestamp tstamp;
      kasp::timestamp::convert_protobuf( proto_tstamp, tstamp);
      res[ "timestamp" ] = kwiver::vital::any( tstamp);
      
      auto str_detected_object_set = grab_proto_from_stream (msg);
      kwiver::protobuf::detected_object_set proto_dos;
      if ( ! proto_dos.ParseFromString( *str_detected_object_set ) )
      {
        LOG_ERROR( logger(), "Detected object set protobuf stream didn't parse correctly" <<
            "ParseFromString failed");
      } 
      auto dos_sptr = std::make_shared< kwiver::vital::detected_object_set >();
      kasp::detected_object_set::convert_protobuf(proto_dos, *dos_sptr);
      res["detected_object_set"] = kwiver::vital::any(dos_sptr);
      
    }
    return res; 
  }
} 
}
}
}
