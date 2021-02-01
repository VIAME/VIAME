// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZATION_JSON_LOAD_SAVE_H
#define ARROWS_SERIALIZATION_JSON_LOAD_SAVE_H

#include <arrows/serialize/json/kwiver_serialize_json_export.h>

#include <vital/types/activity.h>
#include <vital/types/activity_type.h>
#include <vital/types/detected_object_type.h>
#include <vital/types/metadata.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>

namespace kwiver {
namespace vital {
  class detected_object;
  class detected_object_set;
  class geo_point;
  class geo_polygon;
  class polygon;
  class timestamp;
} } // end namespace

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::activity_type& at );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::activity_type& at );

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

KWIVER_SERIALIZE_JSON_EXPORT
void save( cereal::JSONOutputArchive& archive,
           const kwiver::vital::activity& activity );
KWIVER_SERIALIZE_JSON_EXPORT
void load( cereal::JSONInputArchive& archive,
           kwiver::vital::activity& activity );
}

#endif // ARROWS_SERIALIZATION_JSON_LOAD_SAVE_H
