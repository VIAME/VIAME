// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_TRACK_SET_H
#define ARROWS_SERIALIZATION_JSON_TRACK_SET_H

#include <arrows/serialize/json/kwiver_serialize_json_export.h>
#include <vital/algo/data_serializer.h>

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;
}

namespace kwiver {
namespace arrows {
namespace serialize {
namespace json {

class KWIVER_SERIALIZE_JSON_EXPORT track_set
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:track_set",
               "Serializes a track set  using json notation. "
               "This implementation only handles a single data item." );

  track_set();
  virtual ~track_set();

  std::shared_ptr< std::string > serialize( const vital::any& element ) override;
  vital::any deserialize( const std::string& message ) override;
};

} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_JSON_TRACK_SET_H
