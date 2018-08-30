#include "timestamped_detected_object_set.h"

//JSON convertors
#include "timestamp.h"
#include "detected_object_set.h"

#include <vital/types/timestamp.h>
#include <vital/types/detected_object_set.h>
#include <vital/internal/cereal/cereal.hpp>
#include <vital/internal/cereal/archives/json.hpp>

#include <sstream>


namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {
  timestamped_detected_object_set::timestamped_detected_object_set()
  {
    m_element_names.insert( "timestamp" );
    m_element_names.insert( "detected_object_set" );
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
    
    auto tstamp = kwiver::vital::any_cast< kwiver::vital::timestamp >(elements
                                                          .at( "timestamp" ));
    auto dos = kwiver::vital::any_cast< kwiver::vital::detected_object_set_sptr > 
                  (elements.at("detected_object_set"));

    std::stringstream msg;
    msg << "timestamped_detected_object_set ";
    {
      cereal::JSONOutputArchive ar( msg );
      timestamp::save (ar, tstamp);
      detected_object_set::save (ar, *dos);
    }
    
    return std::make_shared< std::string > (msg.str());
  }

  vital::algo::data_serializer::deserialize_result_t 
  timestamped_detected_object_set::deserialize( 
      std::shared_ptr< std::string > message)
  {
    deserialize_result_t res;
    std::istringstream msg( *message );
    kwiver::vital::timestamp tstamp;
    kwiver::vital::detected_object_set* dos = new kwiver::vital::detected_object_set();

    std::string tag;
    msg >> tag;
    
    if (tag != "timestamped_detected_object_set")
    {
      LOG_ERROR( logger(), "Invalid data type tag received." << 
         "Expected \"timestamped_detected_object_set\", received " << tag);
    }
    else
    {
      cereal::JSONInputArchive ar( msg );
      timestamp::load( ar, tstamp);
      detected_object_set::load(ar, *dos); 
    }

    res[ "timestamp" ] = kwiver::vital::any( tstamp );
    res[ "detected_object_set" ] = kwiver::vital::any( 
                                kwiver::vital::detected_object_set_sptr(dos) );
    return res; 
  }
} 
}
}
}
