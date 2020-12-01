// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT_TYPE
#define ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT_TYPE

#include <arrows/serialize/json/kwiver_serialize_json_export.h>
#include <vital/algo/data_serializer.h>
namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;
} // end namespace cereal
namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

class KWIVER_SERIALIZE_JSON_EXPORT detected_object_type
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:detected_object_type",
               "Serializes a detected_object_type using JSON notation. "
               "This implementation only handles a single data item." );

  detected_object_type();
  virtual ~detected_object_type();

  std::shared_ptr< std::string > serialize( const vital::any& element ) override;
  vital::any deserialize( const std::string& message ) override;
};

} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT_TYPE
