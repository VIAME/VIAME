/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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

/**
 * \file
 * \brief definition of kwiver type traits
 */

#ifndef KWIVER_VITAL_TYPE_TRAITS_H
#define KWIVER_VITAL_TYPE_TRAITS_H

#include <vital/vital_types.h>

#include <vital/types/database_query.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/descriptor_request.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/feature_set.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/geo_polygon.h>
#include <vital/types/image_container.h>
#include <vital/types/iqr_feedback.h>
#include <vital/types/matrix.h>
#include <vital/types/metadata.h>
#include <vital/types/object_track_set.h>
#include <vital/types/query_result_set.h>
#include <vital/types/track_descriptor_set.h>
#include <vital/types/uid.h>

#include "trait_utils.h"

#include <memory>
#include <string>
#include <vector>

namespace kwiver {
namespace vital {

  class timestamp;
  class f2f_homography;

  typedef std::vector< double >  double_vector;
  typedef std::shared_ptr< double_vector > double_vector_sptr;
  typedef std::vector< std::string > string_vector;
  typedef std::shared_ptr< string_vector > string_vector_sptr;
  typedef std::vector< unsigned char > uchar_vector;
  typedef std::shared_ptr< uchar_vector > uchar_vector_sptr;

} }


// ==================================================================================
//
// Create type traits for common pipeline types.
// These are types that are passed through the pipeline.
// ( type-trait-name, "canonical_type_name", concrete-type )
//
create_type_trait( bounding_box, "kwiver:bounding_box", kwiver::vital::bounding_box_d );
create_type_trait( timestamp, "kwiver:timestamp", kwiver::vital::timestamp );
create_type_trait( gsd, "kwiver:gsd", double );
create_type_trait( corner_points, "corner_points", kwiver::vital::geo_polygon );
create_type_trait( image, "kwiver:image", kwiver::vital::image_container_sptr );
create_type_trait( mask, "kwiver:mask", kwiver::vital::image_container_sptr );
create_type_trait( feature_set, "kwiver:feature_set", kwiver::vital::feature_set_sptr );
create_type_trait( database_query, "kwiver:database_query", kwiver::vital::database_query_sptr );
create_type_trait( descriptor_set, "kwiver:descriptor_set", kwiver::vital::descriptor_set_sptr );
create_type_trait( descriptor_request, "kwiver:descriptor_request", kwiver::vital::descriptor_request_sptr );
create_type_trait( query_result, "kwiver:query_result", kwiver::vital::query_result_set_sptr );
create_type_trait( iqr_feedback, "kwiver:iqr_feedback", kwiver::vital::iqr_feedback_sptr );
create_type_trait( string, "kwiver:string", kwiver::vital::string_t );
create_type_trait( string_vector, "kwiver:string_vector", kwiver::vital::string_vector_sptr );
create_type_trait( track_set, "kwiver:track_set", kwiver::vital::track_set_sptr );
create_type_trait( feature_track_set, "kwiver:feature_track_set", kwiver::vital::feature_track_set_sptr );
create_type_trait( object_track_set, "kwiver:object_track_set", kwiver::vital::object_track_set_sptr );
create_type_trait( double_vector,  "kwiver:d_vector", kwiver::vital::double_vector_sptr );
create_type_trait( uchar_vector,  "kwiver:uchar_vector", kwiver::vital::uchar_vector_sptr );
create_type_trait( detected_object_set, "kwiver:detected_object_set", kwiver::vital::detected_object_set_sptr );
create_type_trait( track_descriptor_set, "kwiver:track_descriptor_set", kwiver::vital::track_descriptor_set_sptr );
create_type_trait( matrix_d, "kwiver:matrix_d", kwiver::vital::matrix_d );

create_type_trait( homography_src_to_ref, "kwiver:s2r_homography", kwiver::vital::f2f_homography );
create_type_trait( homography_ref_to_src, "kwiver:r2s_homography", kwiver::vital::f2f_homography );
create_type_trait( file_name, "kwiver:file_name", kwiver::vital::path_t );
create_type_trait( image_file_name, "kwiver:image_file_name", kwiver::vital::path_t );
create_type_trait( video_file_name, "kwiver:video_file_name", kwiver::vital::path_t );
create_type_trait( metadata, "kwiver:metadata", kwiver::vital::metadata_vector );
create_type_trait( video_uid, "kwiver:video_uuid", kwiver::vital::uid );
create_type_trait( frame_rate, "kwiver:frame_rate", double );


// ==================================================================================
//
// Create port traits for common port types.
// ( port-name, type-trait-name, "port-description" )
//
create_port_trait( bounding_box, bounding_box, "Bounding box" );
create_port_trait( timestamp, timestamp, "Timestamp for input image." );
create_port_trait( corner_points, corner_points, "Four corner points for image in lat/lon units, ordering ul ur lr ll." );
create_port_trait( gsd, gsd, "GSD for image in meters per pixel." );
create_port_trait( image, image, "Single frame image." );
create_port_trait( left_image, image, "Single frame left image." );
create_port_trait( right_image, image, "Single frame right image." );
create_port_trait( depth_map, image, "Depth map stored in image form." );
create_port_trait( feature_set, feature_set, "Set of detected image features." );
create_port_trait( database_query, database_query, "A database query." );
create_port_trait( descriptor_set, descriptor_set, "Set of descriptors." );
create_port_trait( descriptor_request, descriptor_request, "A request to compute descriptors." );
create_port_trait( iqr_feedback, iqr_feedback, "IQR feedback." );
create_port_trait( iqr_model, uchar_vector, "IQR model." );
create_port_trait( query_result, query_result, "Set of query results." );
create_port_trait( string_vector, string_vector, "Vector of strings." );
create_port_trait( track_set, track_set, "Set of arbitrary tracks." );
create_port_trait( feature_track_set, feature_track_set, "Set of feature tracks." );
create_port_trait( object_track_set, object_track_set, "Set of object tracks." );
create_port_trait( detected_object_set, detected_object_set, "Set of detected objects." );
create_port_trait( track_descriptor_set, track_descriptor_set, "Set of track descriptors." );
create_port_trait( matrix_d, matrix_d, "2-dimensional double matrix." );

create_port_trait( homography_src_to_ref, homography_src_to_ref, "Source image to ref image homography." );
create_port_trait( file_name, file_name, "Name of an arbitrary data file." );
create_port_trait( image_file_name, image_file_name, "Name of an image file." );
create_port_trait( video_file_name, video_file_name, "Name of video file." );
create_port_trait( metadata, metadata, "Video metadata vector for a frame." );
create_port_trait( video_uid, video_uid, "Video UID value." );
create_port_trait( frame_rate, frame_rate, "Video frame rate." );

#endif // KWIVER_VITAL_TYPE_TRAITS_H
