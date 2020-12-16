// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT
#define ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT

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

class KWIVER_SERIALIZE_JSON_EXPORT detected_object
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:detected_object",
               "Serializes a detected_object using JSON notation." );

  detected_object();
  virtual ~detected_object();

  std::shared_ptr< std::string > serialize( const vital::any& elements ) override;
  vital::any deserialize( const std::string& message ) override;
};

} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_JSON_DETECTED_OBJECT
