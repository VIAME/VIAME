/*ckwg +29
 * Copyright 2020 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <python/kwiver/arrows/serialize/json/serialize_image.h>

#include <arrows/serialize/json/image.h>
#include <vital/types/image_container.h>
#include <vital/any.h>

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
