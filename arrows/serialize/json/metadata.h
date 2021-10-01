// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_METADATA
#define ARROWS_SERIALIZATION_JSON_METADATA

#include <arrows/serialize/json/kwiver_serialize_json_export.h>

#include <vital/algo/data_serializer.h>

#include <vital/types/metadata.h>
#include <vital/types/metadata_map.h>

namespace cereal {

class JSONOutputArchive;
class JSONInputArchive;

} // namespace cereal

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
  std::shared_ptr< std::string > serialize_meta(
    vital::metadata_vector const& elements );
  std::shared_ptr< std::string > serialize_map(
    vital::metadata_map::map_metadata_t const& frame_map );
  std::shared_ptr< std::string > serialize(
    vital::any const& elements ) override;

  vital::metadata_map::map_metadata_t deserialize_map(
    std::string const& message );
  vital::any deserialize( std::string const& message ) override;
};

} // namespace json

} // namespace serialize

} // namespace arrows

} // namespace kwiver

#endif
