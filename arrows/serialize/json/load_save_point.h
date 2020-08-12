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
