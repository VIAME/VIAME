// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMP_H
#define ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMP_H

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>
#include <vital/algo/data_serializer.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

class KWIVER_SERIALIZE_PROTOBUF_EXPORT timestamp
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:timestamp",
               "Serializes a timestamp using protobuf notation." );

  timestamp();
  virtual ~timestamp();

  std::shared_ptr< std::string > serialize( const vital::any& element ) override;
  vital::any deserialize( const std::string& message ) override;
};

} } } }

#endif // ARROWS_SERIALIZATION_PROTOBUF_TIMESTAMP_H
