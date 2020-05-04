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


#ifndef VITAL_PROTOBUF_CONVERT_PROTOBUF_POINT_H
#define VITAL_PROTOBUF_CONVERT_PROTOBUF_POINT_H

#include <arrows/serialize/protobuf/kwiver_serialize_protobuf_export.h>
#include <vital/types/point.h>

namespace kwiver {
namespace vital {


} } // end namespace

namespace kwiver {
namespace protobuf {

  class point_i;
  class point_d;
  class covariance;

} } // end namespace

namespace kwiver {
namespace arrows {
namespace serialize {
namespace protobuf {

// ---- 2i point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_i& proto_point,
                       ::kwiver::vital::point_2i& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_2i& point,
                        ::kwiver::protobuf::point_i& proto_point );

// ---- 2d point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_d& proto_point,
                       ::kwiver::vital::point_2d& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_2d& point,
                        ::kwiver::protobuf::point_d& proto_point );

// ---- 2f point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_d& proto_point,
                       ::kwiver::vital::point_2f& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_2f& point,
                        ::kwiver::protobuf::point_d& proto_point );

// ---- 3d point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_d& proto_point,
                       ::kwiver::vital::point_3d& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_3d& point,
                        ::kwiver::protobuf::point_d& proto_point );

// ---- 3f point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_d& proto_point,
                       ::kwiver::vital::point_3f& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_3f& point,
                        ::kwiver::protobuf::point_d& proto_point );

// ---- 4d point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_d& proto_point,
                       ::kwiver::vital::point_4d& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_4d& point,
                        ::kwiver::protobuf::point_d& proto_point );

// ---- 4f point
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::point_d& proto_point,
                       ::kwiver::vital::point_4f& point );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::point_4f& point,
                        ::kwiver::protobuf::point_d& proto_point );

// ----------------------------------------------------------------------------
// -- covariance 2d
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       ::kwiver::vital::covariance_2d& covariance );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::covariance_2d& covariance,
                        ::kwiver::protobuf::covariance& proto_covariance );

// -- covariance 2f
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       ::kwiver::vital::covariance_2f& covariance );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::covariance_2f& covariance,
                        ::kwiver::protobuf::covariance& proto_covariance );

// -- covariance 3d
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       ::kwiver::vital::covariance_3d& covariance );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::covariance_3d& covariance,
                        ::kwiver::protobuf::covariance& proto_covariance );

// -- covariance 3f
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       ::kwiver::vital::covariance_3f& covariance );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::covariance_3f& covariance,
                        ::kwiver::protobuf::covariance& proto_covariance );

// -- covariance 4d
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       ::kwiver::vital::covariance_4d& covariance );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::covariance_4d& covariance,
                        ::kwiver::protobuf::covariance& proto_covariance );

// -- covariance 4f
KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::protobuf::covariance& proto_covariance,
                       ::kwiver::vital::covariance_4f& covariance );

KWIVER_SERIALIZE_PROTOBUF_EXPORT
void convert_protobuf( const ::kwiver::vital::covariance_4f& covariance,
                        ::kwiver::protobuf::covariance& proto_covariance );

} } } } // end namespace

#endif // VITAL_PROTOBUF_CONVERT_PROTOBUF_POINT_H
