#ifndef ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMP_H
#define ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMP_H

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>
#include <vital/algo/data_serializer.h>
#include <vital/types/timestamp.h>
#include <vital/types/protobuf/timestamp.pb.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {
  class KWIVER_SERIALIZE_PROTOBUF_EXPORT timestamp
      : public vital::algorithm_impl< timestamp, vital::algo::data_serializer >
  {
    public:
      static constexpr char const* name = "kwiver:timestamp";
      static constexpr char const* description = 
          "Serializes a timestamp using protobuf notation. ";

      timestamp();
      virtual ~timestamp();

      virtual std::shared_ptr< std::string > serialize( const serialize_param_t& elements );
      virtual deserialize_result_t deserialize( std::shared_ptr< std::string > message);

      static void convert_protobuf( const kwiver::protobuf::timestamp& proto_tstamp,
                                      kwiver::vital::timestamp& tstamp);
      
      static void convert_protobuf( const kwiver::vital::timestamp& tstamp,
                                      kwiver::protobuf::timestamp& proto_tstamp);
  };
} } } }

#endif // ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMP_H
