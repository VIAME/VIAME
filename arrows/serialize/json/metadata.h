// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_METADATA
#define ARROWS_SERIALIZATION_JSON_METADATA

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

class KWIVER_SERIALIZE_JSON_EXPORT metadata
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:metadata",
               "Serializes a metadata vector using json notation." );

  metadata();
  virtual ~metadata();

  std::shared_ptr< std::string > serialize( const vital::any& elements ) override;
  vital::any deserialize( const std::string& message ) override;
};

} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_JSON_METADATA
