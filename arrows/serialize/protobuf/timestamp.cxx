#include "timestamp.h"
#include <vital/exceptions.h>
#include <cstdint>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

  timestamp::timestamp()
  {
    m_element_names.insert (DEFAULT_ELEMENT_NAME);

    GOOGLE_PROTOBUF_VERIFY_VERSION;
  }

  timestamp::~timestamp()
  { }

  std::shared_ptr< std::string > timestamp::
    serialize( const data_serializer::serialize_param_t& elements)
  {
    kwiver::vital::timestamp tstamp =
      kwiver::vital::any_cast< kwiver::vital::timestamp > ( elements.at (DEFAULT_ELEMENT_NAME));
    std::ostringstream msg;
    msg << "timestamp ";
    kwiver::protobuf::timestamp proto_tstamp;
    convert_protobuf( tstamp, proto_tstamp);
    if ( !proto_tstamp.SerializeToOstream( &msg ) ){
      LOG_ERROR( logger(), "proto_timestamp.SerializeToOStream failed" );
    }    
    return std::make_shared< std::string > (msg.str() );
  }
  
  vital::algo::data_serializer::deserialize_result_t 
    timestamp::deserialize( std::shared_ptr< std::string > message)
  {
    kwiver::vital::timestamp tstamp;
    std::istringstream msg( *message );
    std::string tag;
    msg >> tag;
    msg.get();
    if (tag != "timestamp" )
    {
      LOG_ERROR( logger(), "Invalid data type tag receiver. Expected timestamp" 
          << "received " << tag << ". Message dropped.");
    } 
    else 
    {
      kwiver::protobuf::timestamp proto_tstamp;
      if ( !proto_tstamp.ParseFromIstream( &msg ) )
      {
        LOG_ERROR( logger(), "Incoming protobuf stream did not parse correctly."
            << "ParseFromIstream failed.");
      }

      convert_protobuf( proto_tstamp, tstamp);
    }

    deserialize_result_t res;
    res[ DEFAULT_ELEMENT_NAME ] = kwiver::vital::any(tstamp);
    return res;
  }

  void timestamp::convert_protobuf (const kwiver::protobuf::timestamp& proto_tstamp,
                                    kwiver::vital::timestamp& tstamp)
  {
    tstamp = kwiver::vital::timestamp( static_cast<kwiver::vital::time_us_t>(proto_tstamp
        .time()), static_cast<kwiver::vital::frame_id_t>(proto_tstamp.frame()) );
  }
  
  void timestamp::convert_protobuf (const kwiver::vital::timestamp& tstamp,
                                    kwiver::protobuf::timestamp& proto_tstamp)
  {
    proto_tstamp.set_time( static_cast<int64_t>(tstamp.get_time_usec()));
    proto_tstamp.set_frame( static_cast<int64_t>(tstamp.get_frame()) );
  }

}
}
}
}
