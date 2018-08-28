#ifndef ARROWS_SERIALIZATION_JSON_TIMESTAMP
#define ARROWS_SERIALIZATION_JSON_TIMESTAMP

#include <arrows/serialize/json/kwiver_serialize_json_export.h>
#include <vital/algo/data_serializer.h>
#include <vital/types/timestamp.h>

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;
}

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

  class KWIVER_SERIALIZE_JSON_EXPORT timestamp
    : public vital::algorithm_impl< timestamp, vital::algo::data_serializer >
  {
    public:
      static constexpr char const* name  = "kwiver:timestamp";
      static constexpr char const* description = 
        "Serializes a timestamp object using json notation";

      timestamp();
      virtual ~timestamp();

      virtual std::shared_ptr< std::string > serialize( const serialize_param_t& elements );
      virtual deserialize_result_t deserialize( std::shared_ptr< std::string > message );

      static void save( cereal::JSONOutputArchive& archive, 
          const kwiver::vital::timestamp& tstamp);

      static void load( cereal::JSONInputArchive& archive,
          kwiver::vital::timestamp& tstamp);

  };
} } } }

#endif
