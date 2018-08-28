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
  timestamp::timestamp()
  {
    m_element_names.insert( DEFAULT_ELEMENT_NAME );
  }

  timestamp::~timestamp() 
  { }

  std::shared_ptr< std::string > timestamp
    ::serialize (const data_serializer::serialize_param_t& elements)
  {
    kwiver::vital::timestamp tstamp = 
      kwiver::vital::any_cast < kwiver::vital::timestamp >( elements.at( DEFAULT_ELEMENT_NAME ) );
    std::stringstream msg;
    msg << "timestamp ";
    {
      cereal::JSONOutputArchive ar( msg );
      save( ar, tstamp );
    }
    return std::make_shared< std::string > ( msg.str() );
  }

  vital::algo::data_serializer::deserialize_result_t timestamp
    ::deserialize( std::shared_ptr< std::string > message)
  {
    std::stringstream msg( *message );
    kwiver::vital::timestamp tstamp;

    std::string tag;
    msg >> tag;
    if ( tag != "timestamp" )
    {
      LOG_ERROR( logger(), "Invalid data type tag received. Expected \"timestamp\"" <<
                  ",  received \"" << tag << "\". Message dropped." );
    }
    else
    {
      cereal::JSONInputArchive ar(msg);
      load( ar, tstamp );
    }
    
    deserialize_result_t res;
    res [DEFAULT_ELEMENT_NAME] = kwiver::vital::any(tstamp);
    return res;
  }

  void timestamp::save( cereal::JSONOutputArchive& archive, 
      const kwiver::vital::timestamp& tstamp)
  {
    archive ( cereal::make_nvp( "time", tstamp.get_time_usec()),
              cereal::make_nvp( "frame", tstamp.get_frame() ) );        
  }

  void timestamp::load( cereal::JSONInputArchive& archive, 
                        kwiver::vital::timestamp& tstamp){
    int64_t time, frame;
    archive ( CEREAL_NVP ( time ),
              CEREAL_NVP ( frame ) );
    tstamp = kwiver::vital::timestamp(static_cast< kwiver::vital::time_us_t >(time),
                                  static_cast< kwiver::vital::frame_id_t >(frame));
  }
} } } }
