#ifndef ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMPED_DETECTED_OBJECT_SET_H
#define ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMPED_DETECTED_OBJECT_SET_H

#include <arrows/serialize/json/kwiver_serialize_json_export.h>
#include <vital/algo/data_serializer.h>

namespace cereal {
  class JSONOoutputArchive;
  class JSONInputArchive;
}

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

  class KWIVER_SERIALIZE_JSON_EXPORT timestamped_detected_object_set
    : public vital::algorithm_impl< timestamped_detected_object_set, 
                                    vital::algo::data_serializer >
  {
    public:
      static constexpr char const* name= "kwiver:timestamp_detected_object_set";
      static constexpr char const* description = 
        "Serializes timestamp and detected object set using json notation. "
        " This implementation handles a timestanp, port name \"timestamp\", and "
        " a detected object set, port name \"detected_object_set\" as inputs";

      timestamped_detected_object_set();
      virtual ~timestamped_detected_object_set();

      virtual std::shared_ptr< std::string > serialize( const serialize_param_t& elements );
      virtual deserialize_result_t deserialize( std::shared_ptr< std::string > message );
  };
} } } }

#endif // ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMPED_DETECTED_OBJECT_SET_H

