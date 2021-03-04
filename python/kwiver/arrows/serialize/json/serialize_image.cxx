// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include <python/kwiver/arrows/serialize/json/serialize_image.h>

#include <arrows/serialize/json/image.h>
#include <vital/types/image_container.h>
#include <vital/any.h>

namespace kwiver {
namespace arrows {
namespace python {
std::string
serialize_image_using_json( kwiver::vital::simple_image_container img )
{
  kwiver::arrows::serialize::json::image serializer_algo{};
  kwiver::vital::image_container_sptr img_sptr =
         std::make_shared< kwiver::vital::simple_image_container >( img );
  kwiver::vital::any any_img_container{ img_sptr };
  return *serializer_algo.serialize(any_img_container);
}

kwiver::vital::simple_image_container
deserialize_image_using_json( const std::string& message )
{
  kwiver::arrows::serialize::json::image serializer_algo{};
  kwiver::vital::any any_image_container{
                                      serializer_algo.deserialize( message ) };
  kwiver::vital::image_container_sptr deserialized_image =
    kwiver::vital::any_cast< kwiver::vital::image_container_sptr >(
                                                         any_image_container );
  auto deserialized_simple_image =
    std::dynamic_pointer_cast< kwiver::vital::simple_image_container >(
                                                           deserialized_image );
  return *deserialized_simple_image;
}

void serialize_image(py::module &m)
{
  m.def("serialize_image", serialize_image_using_json);
  m.def("deserialize_image", deserialize_image_using_json);
}
}
}
}
