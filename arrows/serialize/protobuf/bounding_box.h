// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_PROTOBUF_BOUNDING_BOX_H
#define ARROWS_SERIALIZATION_PROTOBUF_BOUNDING_BOX_H

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>
#include <vital/algo/data_serializer.h>

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

class KWIVER_SERIALIZE_PROTOBUF_EXPORT bounding_box
  : public vital::algo::data_serializer
{
public:
  PLUGIN_INFO( "kwiver:bounding_box",
               "Serializes a bounding_box using protobuf notation." );

  bounding_box();
  virtual ~bounding_box();

  std::shared_ptr< std::string > serialize( const vital::any& element ) override;
  vital::any deserialize( const std::string& message ) override;

};

} } } }       // end namespace kwiver

#endif // ARROWS_SERIALIZATION_PROTO_BOUNDING_BOX_H
