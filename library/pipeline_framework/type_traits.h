// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief definition of kwiver type traits
 */

#ifndef KWIVER_VITAL_TYPE_TRAITS_H
#define KWIVER_VITAL_TYPE_TRAITS_H

#include <viame/core_types/vital_types.h>

#include <viame/core_types/activity.h>
#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/database_query.h>
#include <viame/core_types/descriptor_request.h>
#include <viame/core_types/descriptor_set.h>
#include <viame/core_types/detected_object_set.h>
#include <viame/core_types/feature_set.h>
#include <viame/core_types/feature_track_set.h>
#include <viame/core_types/geo_point.h>
#include <viame/core_types/geo_polygon.h>
#include <viame/core_types/homography_f2f.h>
#include <viame/core_types/image_container.h>
#include <viame/core_types/image_container_set.h>
#include <viame/core_types/iqr_feedback.h>
#include <viame/core_types/landmark_map.h>
#include <viame/core_types/local_tangent_space.h>
#include <viame/core_types/matrix.h>
#include <viame/core_types/metadata.h>
#include <viame/core_types/metadata_map.h>
#include <viame/core_types/object_track_set.h>
#include <viame/core_types/pointcloud.h>
#include <viame/core_types/query_result_set.h>
#include <viame/core_types/timestamp.h>
#include <viame/core_types/track_descriptor_set.h>
#include <viame/core_types/uid.h>
#include <viame/core_types/video_raw_image.h>
#include <viame/core_types/video_raw_metadata.h>
#include <viame/core_types/video_settings.h>
#include <viame/core_types/video_uninterpreted_data.h>

#include "viame/pipeline_framework/trait_utils.h"

#include <memory>
#include <string>
#include <vector>

namespace kwiver {

namespace vital {

typedef std::vector< double >  double_vector;
typedef std::shared_ptr< double_vector > double_vector_sptr;
typedef std::vector< std::string > string_vector;
typedef std::shared_ptr< string_vector > string_vector_sptr;
typedef std::vector< unsigned char > uchar_vector;
typedef std::shared_ptr< uchar_vector > uchar_vector_sptr;
using string_sptr = std::shared_ptr< std::string >;

} // namespace vital

} // namespace kwiver

// ==================================================================================
//
// Create type traits for common pipeline types.
// These are types that are passed through the pipeline.
// ( type-trait-name, "canonical_type_name", concrete-type )
//
create_type_trait( activity, "kwiver:activity", kwiver::vital::activity );
create_type_trait( bool, "kwiver:bool", bool );
create_type_trait(
  bounding_box, "kwiver:bounding_box",
  kwiver::vital::bounding_box_d );
create_type_trait(
  camera_perspective, "kwiver:camera_perspective",
  kwiver::vital::camera_perspective_sptr );
create_type_trait(
  corner_points, "kwiver:corner_points",
  kwiver::vital::geo_polygon );
create_type_trait(
  database_query, "kwiver:database_query",
  kwiver::vital::database_query_sptr );
create_type_trait(
  descriptor_request, "kwiver:descriptor_request",
  kwiver::vital::descriptor_request_sptr );
create_type_trait(
  descriptor_set, "kwiver:descriptor_set",
  kwiver::vital::descriptor_set_sptr );
create_type_trait(
  detected_object_set, "kwiver:detected_object_set",
  kwiver::vital::detected_object_set_sptr );
create_type_trait(
  double_vector, "kwiver:d_vector",
  kwiver::vital::double_vector_sptr );
create_type_trait(
  feature_set, "kwiver:feature_set",
  kwiver::vital::feature_set_sptr );
create_type_trait(
  feature_track_set, "kwiver:feature_track_set",
  kwiver::vital::feature_track_set_sptr );
create_type_trait( file_name, "kwiver:file_name", kwiver::vital::path_t );
create_type_trait( frame_rate, "kwiver:frame_rate", double );
create_type_trait( geo_point, "kwiver:geo_point", kwiver::vital::geo_point );
create_type_trait( gsd, "kwiver:gsd", double );
create_type_trait(
  homography, "kwiver:homography",
  kwiver::vital::homography_sptr );
create_type_trait(
  success_flag, "kwiver:success_flag",
  bool );
create_type_trait(
  homography_ref_to_src, "kwiver:r2s_homography",
  kwiver::vital::f2f_homography );
create_type_trait(
  homography_src_to_ref, "kwiver:s2r_homography",
  kwiver::vital::f2f_homography );
create_type_trait( image, "kwiver:image", kwiver::vital::image_container_sptr );
create_type_trait(
  image_set, "kwiver:image_set",
  kwiver::vital::image_container_set_sptr );
create_type_trait(
  iqr_feedback, "kwiver:iqr_feedback",
  kwiver::vital::iqr_feedback_sptr );
create_type_trait( kwiver_logical, "kwiver:logical", bool );
create_type_trait(
  landmark_map, "kwiver:landmark_map",
  kwiver::vital::landmark_map_sptr );
create_type_trait(
  local_tangent_space, "kwiver:local_tangent_space",
  kwiver::vital::local_tangent_space );
create_type_trait( mask, "kwiver:mask", kwiver::vital::image_container_sptr );
create_type_trait( matrix_d, "kwiver:matrix_d", kwiver::vital::matrix_d );
create_type_trait(
  metadata, "kwiver:metadata",
  kwiver::vital::metadata_vector );
create_type_trait(
  metadata_map, "kwiver:metadata_map",
  kwiver::vital::metadata_map_sptr );
create_type_trait(
  object_track_set, "kwiver:object_track_set",
  kwiver::vital::object_track_set_sptr );
create_type_trait(
  pointcloud, "kwiver:pointcloud",
  kwiver::vital::pointcloud_sptr );
create_type_trait(
  query_result, "kwiver:query_result",
  kwiver::vital::query_result_set_sptr );
create_type_trait(
  serialized_message, "kwiver:serialized_message",
  kwiver::vital::string_sptr );
create_type_trait( string, "kwiver:string", kwiver::vital::string_t );
create_type_trait(
  string_vector, "kwiver:string_vector",
  kwiver::vital::string_vector_sptr );
create_type_trait( timestamp, "kwiver:timestamp", kwiver::vital::timestamp );
create_type_trait(
  uchar_vector, "kwiver:uchar_vector",
  kwiver::vital::uchar_vector_sptr );
create_type_trait(
  track_descriptor_set, "kwiver:track_descriptor_set",
  kwiver::vital::track_descriptor_set_sptr );
create_type_trait(
  track_set, "kwiver:track_set",
  kwiver::vital::track_set_sptr );
create_type_trait(
  video_raw_image, "kwiver:video_raw_image",
  kwiver::vital::video_raw_image_sptr );
create_type_trait(
  video_raw_metadata, "kwiver:video_raw_metadata",
  kwiver::vital::video_raw_metadata_sptr );
create_type_trait(
  video_settings, "kwiver:video_settings",
  kwiver::vital::video_settings_sptr );
create_type_trait( video_uid, "kwiver:video_uuid", kwiver::vital::uid );
create_type_trait(
  video_uninterpreted_data, "kwiver:video_uninterpreted_data",
  kwiver::vital::video_uninterpreted_data_sptr );

// ==================================================================================
//
// Create port traits for common port types.
// ( port-name, type-trait-name, "port-description" )
//
create_port_trait( activity, activity, "Activity data." );
create_port_trait( bounding_box, bounding_box, "Bounding box" );
create_port_trait(
  camera_perspective, camera_perspective,
  "Perspective camera." );
create_port_trait(
  coordinate_system_updated, kwiver_logical,
  "Set to true if new reference frame is established." );
create_port_trait(
  corner_points, corner_points,
  "Four corner points for image in lat/lon units, ordering ul ur lr ll." );
create_port_trait( database_query, database_query, "A database query." );
create_port_trait( depth_map, image, "Depth map stored in image form." );
create_port_trait(
  detection_time, frame_rate,
  "Detection processing time in seconds." );
create_port_trait(
  descriptor_request, descriptor_request,
  "A request to compute descriptors." );
create_port_trait( descriptor_set, descriptor_set, "Set of descriptors." );
create_port_trait(
  detected_object_set, detected_object_set,
  "Set of detected objects." );
create_port_trait(
  feature_set, feature_set,
  "Set of detected image features." );
create_port_trait(
  feature_track_set, feature_track_set,
  "Set of feature tracks." );
create_port_trait( file_name, file_name, "Name of an arbitrary data file." );
create_port_trait( frame_rate, frame_rate, "Video frame rate." );
create_port_trait( geo_point, geo_point, "Geographic point." );
create_port_trait( gsd, gsd, "GSD for image in meters per pixel." );
create_port_trait(
  homography_src_to_ref, homography_src_to_ref,
  "Source image to ref image homography." );
create_port_trait(
  success_flag, success_flag,
  "Boolean success flag." );
create_port_trait( image, image, "Single frame image." );
create_port_trait( image_file_name, file_name, "Name of an image file." );
create_port_trait(
  image_set, image_set,
  "A collection of images, typically sub images." );
create_port_trait( iqr_feedback, iqr_feedback, "IQR feedback." );
create_port_trait( landmark_map, landmark_map, "Map of landmarks." );
create_port_trait( left_image, image, "Single frame left image." );
create_port_trait(
  local_tangent_space, local_tangent_space,
  "Local geographic coordinate system." );
create_port_trait( matrix_d, matrix_d, "2-dimensional double matrix." );
create_port_trait( metadata, metadata, "Video metadata vector for a frame." );
create_port_trait(
  metadata_map, metadata_map,
  "Map of metadata indexed by frame." );
create_port_trait( motion_heat_map, image, "Motion heat map." );
create_port_trait(
  object_track_set, object_track_set,
  "Set of object tracks." );
create_port_trait( pointcloud, pointcloud, "3D point cloud." );
create_port_trait( query_result, query_result, "Set of query results." );
create_port_trait( right_image, image, "Single frame right image." );
create_port_trait(
  serialized_message, serialized_message,
  "serialized data type" );
create_port_trait( string_vector, string_vector, "Vector of strings." );
create_port_trait( timestamp, timestamp, "Timestamp for input image." );
create_port_trait(
  track_descriptor_set, track_descriptor_set,
  "Set of track descriptors." );
create_port_trait( track_set, track_set, "Set of arbitrary tracks." );
create_port_trait( video_file_name, file_name, "Name of a video file." );
create_port_trait(
  video_raw_image, video_raw_image,
  "Raw video image data for efficient transcoding." );
create_port_trait(
  video_raw_metadata, video_raw_metadata,
  "Raw video metadata for efficient transcoding." );
create_port_trait( video_settings, video_settings, "Video encoding settings." );
create_port_trait( video_uid, video_uid, "Video UID value." );
create_port_trait(
  video_uninterpreted_data, video_uninterpreted_data,
  "Uninterpreted video data." );

#endif // KWIVER_VITAL_TYPE_TRAITS_H
