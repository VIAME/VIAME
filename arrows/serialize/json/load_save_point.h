// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef ARROWS_SERIALIZE_JOAD_SAVE_POINT_H
#define ARROWS_SERIALIZE_JOAD_SAVE_POINT_H

#include <arrows/serialize/json/kwiver_serialize_json_export.h>

#include <vital/types/point.h>

namespace cereal {
  class JSONOutputArchive;
  class JSONInputArchive;

// ---- points
KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_2i& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_2i& pt );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_2d& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_2d& pt );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_2f& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_2f& pt );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_3d& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_3d& pt );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_3f& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_3f& pt );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_4d& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_4d& pt );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::point_4f& pt );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::point_4f& pt );

// ---- covariance
KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::covariance_2d& cov );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::covariance_2d& cov );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::covariance_2f& cov );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::covariance_2f& cov );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::covariance_3d& cov );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::covariance_3d& cov );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::covariance_3f& cov );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::covariance_3f& cov );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::covariance_4d& cov );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::covariance_4d& cov );

KWIVER_SERIALIZE_JSON_EXPORT
void save( ::cereal::JSONOutputArchive& archive, const ::kwiver::vital::covariance_4f& cov );
KWIVER_SERIALIZE_JSON_EXPORT
void load( ::cereal::JSONInputArchive& archive, ::kwiver::vital::covariance_4f& cov );

} // end namespace

#endif // ARROWS_SERIALIZE_JOAD_SAVE_POINT_H
