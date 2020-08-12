/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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

#ifndef ARROWS_SERIALIZATION_JSON_LOAD_SAVE_H
#define ARROWS_SERIALIZATION_JSON_LOAD_SAVE_H

#include <arrows/serialize/json/kwiver_serialize_json_export.h>

#include <vital/types/metadata.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
  class detected_object;
  class detected_object_set;
  class detected_object_type;
  class geo_point;
  class geo_polygon;
  class polygon;
  class timestamp;
} } // end namespace


namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::bounding_box_d& bbox );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::bounding_box_d& bbox );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::detected_object& obj );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::detected_object& obj );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::detected_object_set& obj );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::detected_object_set& obj );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::detected_object_type& dot );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::detected_object_type& dot );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::timestamp& tstamp );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::timestamp& tstamp );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::image_container_sptr obj );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::image_container_sptr& obj );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::metadata_vector& meta );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::metadata_vector& meta );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::metadata& meta );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::metadata& meta );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::geo_polygon& poly );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::geo_polygon& poly );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::geo_point& point );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::geo_point& point );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::polygon& poly );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::polygon& poly );

}

#endif // ARROWS_SERIALIZATION_JSON_LOAD_SAVE_H
