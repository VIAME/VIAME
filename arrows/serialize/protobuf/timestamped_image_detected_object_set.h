#ifndef ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMPED_IMAGE_DETECTED_OBJECT_SET_H
#define ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMPED_IMAGE_DETECTED_OBJECT_SET_H

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>
#include <vital/algo/data_serializer.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

  class KWIVER_SERIALIZE_PROTOBUF_EXPORT timestamped_image_detected_object_set
    : public vital::algorithm_impl< timestamped_image_detected_object_set, 
                                    vital::algo::data_serializer >
  {
    public:
      static constexpr char const* name= "timestamp_image_detected_object_set";
      static constexpr char const* description = 
        "Serializes timestamp, image and detected object set using protobuf notation. "
        "This implementation handles a timestanp, port name \"timestamp\", "
        "an image, port name \"image\" and a detected object set, port name "
        "\"detected_object_set\" as inputs.";

      timestamped_image_detected_object_set();
      virtual ~timestamped_image_detected_object_set();

      virtual std::shared_ptr< std::string > serialize( const serialize_param_t& elements );
      virtual deserialize_result_t deserialize( std::shared_ptr< std::string > message );
  };
} } } }

#endif // ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMPED_IMAGE_DETECTED_OBJECT_SET_H

