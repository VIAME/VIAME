// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT_SET
#define ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT_SET

#include <arrows/serialize/json/kwiver_serialize_json_export.h>
#include <vital/algo/data_serializer.h>
#include <vital/types/detected_object_set.h>

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;
} // end namespace cereal

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

class KWIVER_SERIALIZE_JSON_EXPORT detected_object_set
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:detected_object_set",
               "Serializes a detected_object_set using JSON notation." );

  detected_object_set();
  virtual ~detected_object_set();

  std::shared_ptr< std::string > serialize( const vital::any& element ) override;
  vital::any deserialize( const std::string& message ) override;
};

} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT_SET
