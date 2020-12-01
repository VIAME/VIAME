// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_ACTIVITY_TYPE
#define ARROWS_SERIALIZATION_JSON_ACTIVITY_TYPE

#include <arrows/serialize/json/kwiver_serialize_json_export.h>
#include <vital/algo/data_serializer.h>
#include "load_save.h"

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;
} // end namespace cereal

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

class KWIVER_SERIALIZE_JSON_EXPORT activity_type
  : public vital::algo::data_serializer
{
public:
  // Type name this class supports and description
  PLUGIN_INFO(
    "kwiver:activity_type",
    "Serializes an activity_type using JSON notation. "
    "This implementation only handles a single data item."
  );

  activity_type();
  virtual ~activity_type();

  std::shared_ptr< std::string >
    serialize( const kwiver::vital::any& element ) override;
  kwiver::vital::any deserialize( const std::string& message ) override;
};

} } } }

#endif
