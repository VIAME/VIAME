/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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

/**
 * \file
 * \brief Stereo measurement utility functions implementation
 */

#include "measurement_utilities.h"

#include <vital/util/string.h>

#include <arrows/mvg/triangulate.h>

#ifdef VIAME_ENABLE_OPENCV
  #include <arrows/ocv/image_container.h>
  #include <opencv2/imgproc/imgproc.hpp>
  #include <opencv2/core/eigen.hpp>
#endif

#include <algorithm>
#include <limits>
#include <sstream>

namespace viame
{

namespace core
{

// =============================================================================
// measurement_settings implementation
// =============================================================================

// -----------------------------------------------------------------------------
measurement_settings
::measurement_settings()
  : matching_methods( "input_pairs_only,template_matching" )
  , default_depth( 5.0 )
  , template_size( 31 )
  , search_range( 128 )
  , use_distortion( true )
  , sgbm_min_disparity( 0 )
  , sgbm_num_disparities( 128 )
  , sgbm_block_size( 5 )
  , feature_search_radius( 50.0 )
  , ransac_inlier_scale( 3.0 )
  , min_ransac_inliers( 10 )
  , box_scale_factor( 1.10 )
  , record_stereo_method( true )
{
}

// -----------------------------------------------------------------------------
measurement_settings
::~measurement_settings()
{
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
measurement_settings
::get_configuration() const
{
  kv::config_block_sptr config = kv::config_block::empty_config();

  config->set_value( "matching_methods", matching_methods,
    "Comma-separated list of methods to try (in order) for finding corresponding points "
    "in right camera for left-only tracks. Methods will be tried in the order specified "
    "until one succeeds. Valid options: "
    "'input_pairs_only' (use existing keypoints from right camera if available), "
    "'depth_projection' (uses default_depth to project points), "
    "'template_matching' (rectifies images and searches along epipolar lines), "
    "'sgbm_disparity' (uses Semi-Global Block Matching to compute disparity map), "
    "'feature_descriptor' (uses vital feature detection/descriptor/matching), "
    "'ransac_feature' (feature matching with RANSAC-based fundamental matrix filtering). "
    "Example: 'input_pairs_only,template_matching,depth_projection'" );

  config->set_value( "default_depth", default_depth,
    "Default depth (in meters) to use when projecting left camera points to right camera "
    "for tracks that only exist in the left camera, when using the depth_projection option" );

  config->set_value( "template_size", template_size,
    "Template window size (in pixels) for template matching. Must be odd number." );

  config->set_value( "search_range", search_range,
    "Search range (in pixels) along epipolar line for template matching." );

  config->set_value( "use_distortion", use_distortion,
    "Whether to use distortion coefficients from the calibration during rectification. "
    "If true, distortion coefficients from the calibration file are used. "
    "If false, zero distortion is assumed." );

  config->set_value( "sgbm_min_disparity", sgbm_min_disparity,
    "Minimum possible disparity value for SGBM. Normally 0, but can be negative." );

  config->set_value( "sgbm_num_disparities", sgbm_num_disparities,
    "Maximum disparity minus minimum disparity for SGBM. Must be divisible by 16." );

  config->set_value( "sgbm_block_size", sgbm_block_size,
    "Block size for SGBM. Must be odd number >= 1. Typically 3-11." );

  config->set_value( "feature_search_radius", feature_search_radius,
    "Maximum distance (in pixels) to search for feature matches around the expected location. "
    "Used for feature_descriptor and ransac_feature methods." );

  config->set_value( "ransac_inlier_scale", ransac_inlier_scale,
    "Inlier threshold for RANSAC fundamental matrix estimation. "
    "Points with reprojection error below this threshold are considered inliers." );

  config->set_value( "min_ransac_inliers", min_ransac_inliers,
    "Minimum number of inliers required for a valid RANSAC result." );

  config->set_value( "box_scale_factor", box_scale_factor,
    "Scale factor to expand the bounding box around keypoints when creating "
    "new detections for the right image. A value of 1.10 means 10% expansion." );

  config->set_value( "record_stereo_method", record_stereo_method,
    "If true, record the stereo measurement method used as an attribute on each "
    "output detection object. The attribute will be ':stereo_method=METHOD' "
    "where METHOD is one of: input_kps_used, template_matching, sgbm_disparity, "
    "feature_descriptor, ransac_feature, or depth_projection." );

  // Add nested algorithm configurations
  kv::algo::detect_features::get_nested_algo_configuration(
    "feature_detector", config, feature_detector );
  kv::algo::extract_descriptors::get_nested_algo_configuration(
    "descriptor_extractor", config, descriptor_extractor );
  kv::algo::match_features::get_nested_algo_configuration(
    "feature_matcher", config, feature_matcher );
  kv::algo::estimate_fundamental_matrix::get_nested_algo_configuration(
    "fundamental_matrix_estimator", config, fundamental_matrix_estimator );

  return config;
}

// -----------------------------------------------------------------------------
void
measurement_settings
::set_configuration( kv::config_block_sptr config )
{
  matching_methods = config->get_value< std::string >( "matching_methods", matching_methods );
  default_depth = config->get_value< double >( "default_depth", default_depth );
  template_size = config->get_value< int >( "template_size", template_size );
  search_range = config->get_value< int >( "search_range", search_range );
  use_distortion = config->get_value< bool >( "use_distortion", use_distortion );
  sgbm_min_disparity = config->get_value< int >( "sgbm_min_disparity", sgbm_min_disparity );
  sgbm_num_disparities = config->get_value< int >( "sgbm_num_disparities", sgbm_num_disparities );
  sgbm_block_size = config->get_value< int >( "sgbm_block_size", sgbm_block_size );
  feature_search_radius = config->get_value< double >( "feature_search_radius", feature_search_radius );
  ransac_inlier_scale = config->get_value< double >( "ransac_inlier_scale", ransac_inlier_scale );
  min_ransac_inliers = config->get_value< int >( "min_ransac_inliers", min_ransac_inliers );
  box_scale_factor = config->get_value< double >( "box_scale_factor", box_scale_factor );
  record_stereo_method = config->get_value< bool >( "record_stereo_method", record_stereo_method );

  // Configure nested algorithms
  kv::algo::detect_features::set_nested_algo_configuration(
    "feature_detector", config, feature_detector );
  kv::algo::extract_descriptors::set_nested_algo_configuration(
    "descriptor_extractor", config, descriptor_extractor );
  kv::algo::match_features::set_nested_algo_configuration(
    "feature_matcher", config, feature_matcher );
  kv::algo::estimate_fundamental_matrix::set_nested_algo_configuration(
    "fundamental_matrix_estimator", config, fundamental_matrix_estimator );
}

// -----------------------------------------------------------------------------
bool
measurement_settings
::check_configuration( kv::config_block_sptr config ) const
{
  bool valid = true;

  // Check nested algorithms if present
  if( config->has_value( "feature_detector:type" ) &&
      config->get_value< std::string >( "feature_detector:type" ) != "" )
  {
    valid = kv::algo::detect_features::check_nested_algo_configuration(
      "feature_detector", config ) && valid;
  }
  if( config->has_value( "descriptor_extractor:type" ) &&
      config->get_value< std::string >( "descriptor_extractor:type" ) != "" )
  {
    valid = kv::algo::extract_descriptors::check_nested_algo_configuration(
      "descriptor_extractor", config ) && valid;
  }
  if( config->has_value( "feature_matcher:type" ) &&
      config->get_value< std::string >( "feature_matcher:type" ) != "" )
  {
    valid = kv::algo::match_features::check_nested_algo_configuration(
      "feature_matcher", config ) && valid;
  }
  if( config->has_value( "fundamental_matrix_estimator:type" ) &&
      config->get_value< std::string >( "fundamental_matrix_estimator:type" ) != "" )
  {
    valid = kv::algo::estimate_fundamental_matrix::check_nested_algo_configuration(
      "fundamental_matrix_estimator", config ) && valid;
  }

  return valid;
}

// -----------------------------------------------------------------------------
std::vector< std::string >
measurement_settings
::get_matching_methods() const
{
  return measurement_utilities::parse_matching_methods( matching_methods );
}

// -----------------------------------------------------------------------------
std::string
measurement_settings
::validate_matching_methods() const
{
  auto methods = get_matching_methods();

  if( methods.empty() )
  {
    return "No valid matching methods specified";
  }

  auto valid_methods = measurement_utilities::get_valid_methods();
  for( const auto& method : methods )
  {
    if( std::find( valid_methods.begin(), valid_methods.end(), method ) == valid_methods.end() )
    {
      return "Invalid matching method: " + method;
    }
  }

  return "";
}

// -----------------------------------------------------------------------------
bool
measurement_settings
::any_method_requires_images() const
{
  auto methods = get_matching_methods();
  for( const auto& method : methods )
  {
    if( measurement_utilities::method_requires_images( method ) )
    {
      return true;
    }
  }
  return false;
}

// -----------------------------------------------------------------------------
std::vector< std::string >
measurement_settings
::check_feature_algorithm_warnings() const
{
  std::vector< std::string > warnings;

  auto methods = get_matching_methods();
  for( const auto& method : methods )
  {
    if( method == "feature_descriptor" || method == "ransac_feature" )
    {
      if( !feature_detector )
      {
        warnings.push_back( "Feature detector not configured; " + method + " method may not work" );
      }
      if( !descriptor_extractor )
      {
        warnings.push_back( "Descriptor extractor not configured; " + method + " method may not work" );
      }
      if( !feature_matcher )
      {
        warnings.push_back( "Feature matcher not configured; " + method + " method may not work" );
      }
      if( method == "ransac_feature" && !fundamental_matrix_estimator )
      {
        warnings.push_back( "Fundamental matrix estimator not configured; ransac_feature method may not work" );
      }
      break;  // Only need to check once
    }
  }

  return warnings;
}

// =============================================================================
// measurement_utilities implementation
// =============================================================================

// -----------------------------------------------------------------------------
measurement_utilities
::measurement_utilities()
  : m_default_depth( 5.0 )
  , m_template_size( 31 )
  , m_search_range( 128 )
  , m_use_distortion( true )
  , m_sgbm_min_disparity( 0 )
  , m_sgbm_num_disparities( 128 )
  , m_sgbm_block_size( 5 )
  , m_feature_search_radius( 50.0 )
  , m_ransac_inlier_scale( 3.0 )
  , m_min_ransac_inliers( 10 )
  , m_box_scale_factor( 1.10 )
  , m_cached_frame_id( 0 )
#ifdef VIAME_ENABLE_OPENCV
  , m_rectification_computed( false )
#endif
{
}

// -----------------------------------------------------------------------------
measurement_utilities
::~measurement_utilities()
{
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_default_depth( double depth )
{
  m_default_depth = depth;
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_template_params( int template_size, int search_range )
{
  m_template_size = template_size;
  m_search_range = search_range;

  // Ensure template size is odd
  if( m_template_size % 2 == 0 )
  {
    m_template_size++;
  }
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_use_distortion( bool use_distortion )
{
  m_use_distortion = use_distortion;
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_sgbm_params( int min_disparity, int num_disparities, int block_size )
{
  m_sgbm_min_disparity = min_disparity;
  m_sgbm_num_disparities = num_disparities;
  m_sgbm_block_size = block_size;

  // Ensure num_disparities is divisible by 16
  if( m_sgbm_num_disparities % 16 != 0 )
  {
    m_sgbm_num_disparities = ( ( m_sgbm_num_disparities / 16 ) + 1 ) * 16;
  }

  // Ensure block size is odd
  if( m_sgbm_block_size % 2 == 0 )
  {
    m_sgbm_block_size++;
  }

#ifdef VIAME_ENABLE_OPENCV
  // Reset SGBM matcher so it will be recreated with new params
  m_sgbm.release();
#endif
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_feature_params( double search_radius, double ransac_inlier_scale,
                      int min_ransac_inliers )
{
  m_feature_search_radius = search_radius;
  m_ransac_inlier_scale = ransac_inlier_scale;
  m_min_ransac_inliers = min_ransac_inliers;
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_box_scale_factor( double scale_factor )
{
  m_box_scale_factor = scale_factor;
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_feature_algorithms(
  kv::algo::detect_features_sptr detector,
  kv::algo::extract_descriptors_sptr extractor,
  kv::algo::match_features_sptr matcher,
  kv::algo::estimate_fundamental_matrix_sptr fundamental_estimator )
{
  m_feature_detector = detector;
  m_descriptor_extractor = extractor;
  m_feature_matcher = matcher;
  m_fundamental_matrix_estimator = fundamental_estimator;
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::configure( const measurement_settings& settings )
{
  set_default_depth( settings.default_depth );
  set_template_params( settings.template_size, settings.search_range );
  set_use_distortion( settings.use_distortion );
  set_sgbm_params( settings.sgbm_min_disparity, settings.sgbm_num_disparities,
                   settings.sgbm_block_size );
  set_feature_params( settings.feature_search_radius, settings.ransac_inlier_scale,
                      settings.min_ransac_inliers );
  set_box_scale_factor( settings.box_scale_factor );
  set_feature_algorithms( settings.feature_detector, settings.descriptor_extractor,
                          settings.feature_matcher, settings.fundamental_matrix_estimator );
}

// -----------------------------------------------------------------------------
kv::vector_2d
measurement_utilities
::project_left_to_right(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point ) const
{
  // Unproject the left camera point to normalized image coordinates
  const auto left_intrinsics = left_cam.get_intrinsics();
  const kv::vector_2d normalized_pt = left_intrinsics->unmap( left_point );

  // Convert to homogeneous coordinates and add depth
  kv::vector_3d ray_direction( normalized_pt.x(), normalized_pt.y(), 1.0 );
  ray_direction.normalize();

  // Compute 3D point at default depth in left camera coordinates
  kv::vector_3d point_3d_left_cam = ray_direction * m_default_depth;

  // Transform to world coordinates
  const auto& left_rotation = left_cam.rotation();
  const auto& left_center = left_cam.center();
  kv::vector_3d point_3d_world = left_rotation.inverse() * point_3d_left_cam + left_center;

  // Transform to right camera coordinates
  const auto& right_rotation = right_cam.rotation();
  const auto& right_center = right_cam.center();
  kv::vector_3d point_3d_right_cam = right_rotation * ( point_3d_world - right_center );

  // Project to right camera image
  const auto right_intrinsics = right_cam.get_intrinsics();
  kv::vector_2d normalized_right( point_3d_right_cam.x() / point_3d_right_cam.z(),
                                   point_3d_right_cam.y() / point_3d_right_cam.z() );
  return right_intrinsics->map( normalized_right );
}

// -----------------------------------------------------------------------------
kv::bounding_box_d
measurement_utilities
::compute_bbox_from_keypoints(
  const kv::vector_2d& head_point,
  const kv::vector_2d& tail_point ) const
{
  // Compute bounding box around the keypoints
  double min_x = std::min( head_point.x(), tail_point.x() );
  double max_x = std::max( head_point.x(), tail_point.x() );
  double min_y = std::min( head_point.y(), tail_point.y() );
  double max_y = std::max( head_point.y(), tail_point.y() );

  // Compute center and dimensions
  double center_x = ( min_x + max_x ) / 2.0;
  double center_y = ( min_y + max_y ) / 2.0;
  double width = max_x - min_x;
  double height = max_y - min_y;

  // Apply scale factor
  double scaled_width = width * m_box_scale_factor;
  double scaled_height = height * m_box_scale_factor;

  // Compute new bounding box coordinates
  double new_min_x = center_x - scaled_width / 2.0;
  double new_max_x = center_x + scaled_width / 2.0;
  double new_min_y = center_y - scaled_height / 2.0;
  double new_max_y = center_y + scaled_height / 2.0;

  return kv::bounding_box_d( new_min_x, new_min_y, new_max_x, new_max_y );
}

// -----------------------------------------------------------------------------
kv::vector_3d
measurement_utilities
::triangulate_point(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point ) const
{
  Eigen::Matrix<float, 2, 1> left_pt( left_point.x(), left_point.y() );
  Eigen::Matrix<float, 2, 1> right_pt( right_point.x(), right_point.y() );

  auto point_3d = kwiver::arrows::mvg::triangulate_fast_two_view(
    left_cam, right_cam, left_pt, right_pt );

  return kv::vector_3d( point_3d.x(), point_3d.y(), point_3d.z() );
}

// -----------------------------------------------------------------------------
double
measurement_utilities
::compute_stereo_length(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& right_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d& right_tail ) const
{
  kv::vector_3d head_3d = triangulate_point( left_cam, right_cam, left_head, right_head );
  kv::vector_3d tail_3d = triangulate_point( left_cam, right_cam, left_tail, right_tail );

  return ( tail_3d - head_3d ).norm();
}

// -----------------------------------------------------------------------------
measurement_utilities::stereo_correspondence_result
measurement_utilities
::find_stereo_correspondence(
  const std::vector< std::string >& methods,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_head,
  const kv::vector_2d& left_tail,
  const kv::vector_2d* right_head_input,
  const kv::vector_2d* right_tail_input,
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image )
{
  stereo_correspondence_result result;
  result.success = false;
  result.left_head = left_head;
  result.left_tail = left_tail;

  bool head_found = false;
  bool tail_found = false;

#ifdef VIAME_ENABLE_OPENCV
  // Prepare stereo images if needed
  bool has_images = ( left_image && right_image );
  if( has_images )
  {
    m_cached_stereo_images = prepare_stereo_images(
      methods, left_cam, right_cam, left_image, right_image );
  }
#endif

  for( const auto& method : methods )
  {
    if( head_found && tail_found )
    {
      break;
    }

    if( method == "input_pairs_only" )
    {
      if( right_head_input && right_tail_input )
      {
        result.right_head = *right_head_input;
        result.right_tail = *right_tail_input;
        head_found = true;
        tail_found = true;
        result.method_used = "input_pairs_only";
      }
    }
    else if( method == "depth_projection" )
    {
      result.right_head = project_left_to_right( left_cam, right_cam, result.left_head );
      result.right_tail = project_left_to_right( left_cam, right_cam, result.left_tail );
      head_found = true;
      tail_found = true;
      result.method_used = "depth_projection";
    }
#ifdef VIAME_ENABLE_OPENCV
    else if( method == "template_matching" && m_cached_stereo_images.rectified_available )
    {
      kv::vector_2d left_head_rect = rectify_point( result.left_head, false );
      kv::vector_2d left_tail_rect = rectify_point( result.left_tail, false );

      kv::vector_2d right_head_rect, right_tail_rect;
      head_found = find_corresponding_point_template_matching(
        m_cached_stereo_images.left_rectified, m_cached_stereo_images.right_rectified,
        left_head_rect, right_head_rect );
      tail_found = find_corresponding_point_template_matching(
        m_cached_stereo_images.left_rectified, m_cached_stereo_images.right_rectified,
        left_tail_rect, right_tail_rect );

      if( head_found && tail_found )
      {
        result.right_head = unrectify_point( right_head_rect, true, right_cam );
        result.right_tail = unrectify_point( right_tail_rect, true, right_cam );
        result.method_used = "template_matching";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
    else if( method == "sgbm_disparity" && m_cached_stereo_images.disparity_available )
    {
      kv::vector_2d left_head_rect = rectify_point( result.left_head, false );
      kv::vector_2d left_tail_rect = rectify_point( result.left_tail, false );

      kv::vector_2d right_head_rect, right_tail_rect;
      head_found = find_corresponding_point_sgbm(
        m_cached_stereo_images.disparity_map, left_head_rect, right_head_rect );
      tail_found = find_corresponding_point_sgbm(
        m_cached_stereo_images.disparity_map, left_tail_rect, right_tail_rect );

      if( head_found && tail_found )
      {
        result.right_head = unrectify_point( right_head_rect, true, right_cam );
        result.right_tail = unrectify_point( right_tail_rect, true, right_cam );
        result.method_used = "sgbm_disparity";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
#endif
    else if( method == "feature_descriptor" && left_image && right_image )
    {
      kv::vector_2d left_head_copy = result.left_head;
      kv::vector_2d left_tail_copy = result.left_tail;

      head_found = find_corresponding_point_feature_descriptor(
        left_image, right_image, left_head_copy, result.right_head );
      tail_found = find_corresponding_point_feature_descriptor(
        left_image, right_image, left_tail_copy, result.right_tail );

      if( head_found && tail_found )
      {
        result.left_head = left_head_copy;
        result.left_tail = left_tail_copy;
        result.method_used = "feature_descriptor";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
    else if( method == "ransac_feature" && left_image && right_image )
    {
      kv::vector_2d left_head_copy = result.left_head;
      kv::vector_2d left_tail_copy = result.left_tail;

      head_found = find_corresponding_point_ransac_feature(
        left_image, right_image, left_head_copy, result.right_head );
      tail_found = find_corresponding_point_ransac_feature(
        left_image, right_image, left_tail_copy, result.right_tail );

      if( head_found && tail_found )
      {
        result.left_head = left_head_copy;
        result.left_tail = left_tail_copy;
        result.method_used = "ransac_feature";
      }
      else
      {
        head_found = false;
        tail_found = false;
      }
    }
  }

  result.success = ( head_found && tail_found );
  return result;
}

#ifdef VIAME_ENABLE_OPENCV
// -----------------------------------------------------------------------------
measurement_utilities::stereo_image_data
measurement_utilities
::prepare_stereo_images(
  const std::vector< std::string >& methods,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image )
{
  stereo_image_data data;
  data.rectified_available = false;
  data.disparity_available = false;

  if( !left_image || !right_image )
  {
    return data;
  }

  // Check which methods need rectified images
  bool needs_rectified = false;
  bool needs_sgbm = false;
  for( const auto& method : methods )
  {
    if( method == "template_matching" || method == "sgbm_disparity" )
    {
      needs_rectified = true;
    }
    if( method == "sgbm_disparity" )
    {
      needs_sgbm = true;
    }
  }

  if( !needs_rectified )
  {
    return data;
  }

  // Convert to OpenCV format
  cv::Mat left_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat right_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Convert to grayscale (handle each image independently since they may have different channel counts)
  if( left_cv.channels() == 3 )
  {
    cv::cvtColor( left_cv, left_cv, cv::COLOR_BGR2GRAY );
  }
  else if( left_cv.channels() == 4 )
  {
    cv::cvtColor( left_cv, left_cv, cv::COLOR_BGRA2GRAY );
  }
  // If already 1 channel, no conversion needed

  if( right_cv.channels() == 3 )
  {
    cv::cvtColor( right_cv, right_cv, cv::COLOR_BGR2GRAY );
  }
  else if( right_cv.channels() == 4 )
  {
    cv::cvtColor( right_cv, right_cv, cv::COLOR_BGRA2GRAY );
  }
  // If already 1 channel, no conversion needed

  // Compute rectification maps if needed
  compute_rectification_maps( left_cam, right_cam, left_cv.size() );

  // Rectify images
  data.left_rectified = rectify_image( left_cv, false );
  data.right_rectified = rectify_image( right_cv, true );
  data.rectified_available = true;

  // Compute disparity if needed
  if( needs_sgbm )
  {
    data.disparity_map = compute_sgbm_disparity( data.left_rectified, data.right_rectified );
    data.disparity_available = !data.disparity_map.empty();
  }

  return data;
}
#endif

// -----------------------------------------------------------------------------
bool
measurement_utilities
::find_corresponding_point_feature_descriptor(
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  kv::vector_2d& left_point,
  kv::vector_2d& right_point )
{
  if( !m_feature_detector || !m_descriptor_extractor || !m_feature_matcher )
  {
    return false;
  }

  // Detect features and extract descriptors if not cached for this frame
  if( !m_cached_left_features || !m_cached_right_features )
  {
    m_cached_left_features = m_feature_detector->detect( left_image );
    m_cached_right_features = m_feature_detector->detect( right_image );

    m_cached_left_descriptors = m_descriptor_extractor->extract(
      left_image, m_cached_left_features );
    m_cached_right_descriptors = m_descriptor_extractor->extract(
      right_image, m_cached_right_features );

    m_cached_matches = m_feature_matcher->match(
      m_cached_left_features, m_cached_left_descriptors,
      m_cached_right_features, m_cached_right_descriptors );
  }

  if( !m_cached_matches || m_cached_matches->size() == 0 )
  {
    return false;
  }

  // Get the feature vectors
  auto left_features = m_cached_left_features->features();
  auto right_features = m_cached_right_features->features();
  auto matches = m_cached_matches->matches();

  // Find the closest matched feature to our query point
  double best_dist = std::numeric_limits<double>::max();
  kv::vector_2d best_left_point;
  kv::vector_2d best_right_point;
  bool found = false;

  for( const auto& match : matches )
  {
    if( match.first >= left_features.size() ||
        match.second >= right_features.size() )
    {
      continue;
    }

    const auto& left_feat = left_features[match.first];
    const auto& right_feat = right_features[match.second];

    kv::vector_2d left_feat_loc = left_feat->loc();
    double dist = ( left_feat_loc - left_point ).norm();

    if( dist < m_feature_search_radius && dist < best_dist )
    {
      best_dist = dist;
      best_left_point = left_feat_loc;
      best_right_point = right_feat->loc();
      found = true;
    }
  }

  if( found )
  {
    // Update left_point to the actual feature location
    left_point = best_left_point;
    right_point = best_right_point;
  }

  return found;
}

// -----------------------------------------------------------------------------
bool
measurement_utilities
::find_corresponding_point_ransac_feature(
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  kv::vector_2d& left_point,
  kv::vector_2d& right_point )
{
  if( !m_feature_detector || !m_descriptor_extractor ||
      !m_feature_matcher || !m_fundamental_matrix_estimator )
  {
    return false;
  }

  // Detect features and extract descriptors if not cached for this frame
  if( !m_cached_left_features || !m_cached_right_features )
  {
    m_cached_left_features = m_feature_detector->detect( left_image );
    m_cached_right_features = m_feature_detector->detect( right_image );

    m_cached_left_descriptors = m_descriptor_extractor->extract(
      left_image, m_cached_left_features );
    m_cached_right_descriptors = m_descriptor_extractor->extract(
      right_image, m_cached_right_features );

    m_cached_matches = m_feature_matcher->match(
      m_cached_left_features, m_cached_left_descriptors,
      m_cached_right_features, m_cached_right_descriptors );
  }

  if( !m_cached_matches || m_cached_matches->size() == 0 )
  {
    return false;
  }

  // Get the feature vectors
  auto left_features = m_cached_left_features->features();
  auto right_features = m_cached_right_features->features();
  auto matches = m_cached_matches->matches();

  // Estimate fundamental matrix using RANSAC to filter outliers
  std::vector<bool> inliers;
  auto F = m_fundamental_matrix_estimator->estimate(
    m_cached_left_features, m_cached_right_features,
    m_cached_matches, inliers, m_ransac_inlier_scale );

  // Count inliers
  int inlier_count = 0;
  for( bool is_inlier : inliers )
  {
    if( is_inlier )
    {
      ++inlier_count;
    }
  }

  if( inlier_count < m_min_ransac_inliers )
  {
    return false;
  }

  // Find the closest inlier match to our query point
  double best_dist = std::numeric_limits<double>::max();
  kv::vector_2d best_left_point;
  kv::vector_2d best_right_point;
  bool found = false;

  for( size_t i = 0; i < matches.size(); ++i )
  {
    if( !inliers[i] )
    {
      continue;
    }

    const auto& match = matches[i];
    if( match.first >= left_features.size() ||
        match.second >= right_features.size() )
    {
      continue;
    }

    const auto& left_feat = left_features[match.first];
    const auto& right_feat = right_features[match.second];

    kv::vector_2d left_feat_loc = left_feat->loc();
    double dist = ( left_feat_loc - left_point ).norm();

    if( dist < m_feature_search_radius && dist < best_dist )
    {
      best_dist = dist;
      best_left_point = left_feat_loc;
      best_right_point = right_feat->loc();
      found = true;
    }
  }

  if( found )
  {
    // Update left_point to the actual feature location
    left_point = best_left_point;
    right_point = best_right_point;
  }

  return found;
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::clear_feature_cache()
{
  m_cached_left_features.reset();
  m_cached_right_features.reset();
  m_cached_left_descriptors.reset();
  m_cached_right_descriptors.reset();
  m_cached_matches.reset();
}

// -----------------------------------------------------------------------------
void
measurement_utilities
::set_frame_id( kv::frame_id_t frame_id )
{
  if( m_cached_frame_id != frame_id )
  {
    clear_feature_cache();
    m_cached_frame_id = frame_id;
  }
}

#ifdef VIAME_ENABLE_OPENCV

// -----------------------------------------------------------------------------
void
measurement_utilities
::compute_rectification_maps(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const cv::Size& image_size )
{
  if( m_rectification_computed )
  {
    return;
  }

  // Get camera intrinsics
  auto left_intrinsics = left_cam.get_intrinsics();
  auto right_intrinsics = right_cam.get_intrinsics();

  // Convert to OpenCV matrices
  cv::Mat K1, K2, D1, D2, R, T;

  // Camera matrices
  Eigen::Matrix3d K1_eigen = left_intrinsics->as_matrix();
  Eigen::Matrix3d K2_eigen = right_intrinsics->as_matrix();
  cv::eigen2cv( K1_eigen, K1 );
  cv::eigen2cv( K2_eigen, K2 );

  // Distortion coefficients
  D1 = cv::Mat::zeros( 5, 1, CV_64F );
  D2 = cv::Mat::zeros( 5, 1, CV_64F );

  if( m_use_distortion )
  {
    std::vector<double> left_dist = left_intrinsics->dist_coeffs();
    std::vector<double> right_dist = right_intrinsics->dist_coeffs();

    // Convert distortion coefficients to OpenCV format
    for( size_t i = 0; i < std::min( left_dist.size(), size_t(5) ); ++i )
    {
      D1.at<double>( i, 0 ) = left_dist[i];
    }

    for( size_t i = 0; i < std::min( right_dist.size(), size_t(5) ); ++i )
    {
      D2.at<double>( i, 0 ) = right_dist[i];
    }
  }

  // Compute rotation and translation between cameras
  Eigen::Matrix3d R_left = left_cam.rotation().matrix();
  Eigen::Matrix3d R_right = right_cam.rotation().matrix();
  Eigen::Matrix3d R_relative = R_right * R_left.transpose();

  Eigen::Vector3d t_relative = right_cam.center() - left_cam.center();
  t_relative = R_right * t_relative;

  cv::eigen2cv( R_relative, R );
  cv::eigen2cv( t_relative, T );

  // Compute rectification transforms
  cv::Mat Q;
  cv::stereoRectify( K1, D1, K2, D2, image_size, R, T,
                     m_R1, m_R2, m_P1, m_P2, Q,
                     cv::CALIB_ZERO_DISPARITY, 0 );

  // Store camera matrices
  m_K1 = K1.clone();
  m_K2 = K2.clone();

  // Compute rectification maps
  cv::initUndistortRectifyMap( K1, D1, m_R1, m_P1, image_size, CV_32FC1,
    m_rectification_map_left_x, m_rectification_map_left_y );
  cv::initUndistortRectifyMap( K2, D2, m_R2, m_P2, image_size, CV_32FC1,
    m_rectification_map_right_x, m_rectification_map_right_y );

  m_rectification_computed = true;
}

// -----------------------------------------------------------------------------
bool
measurement_utilities
::rectification_computed() const
{
  return m_rectification_computed;
}

// -----------------------------------------------------------------------------
kv::vector_2d
measurement_utilities
::rectify_point(
  const kv::vector_2d& original_point,
  bool is_right_camera ) const
{
  if( !m_rectification_computed )
  {
    return original_point;
  }

  const cv::Mat& map_x = is_right_camera ? m_rectification_map_right_x : m_rectification_map_left_x;
  const cv::Mat& map_y = is_right_camera ? m_rectification_map_right_y : m_rectification_map_left_y;

  int x = static_cast<int>( original_point.x() + 0.5 );
  int y = static_cast<int>( original_point.y() + 0.5 );

  if( x < 0 || x >= map_x.cols || y < 0 || y >= map_x.rows )
  {
    return original_point;
  }

  float rect_x = map_x.at<float>( y, x );
  float rect_y = map_y.at<float>( y, x );

  return kv::vector_2d( rect_x, rect_y );
}

// -----------------------------------------------------------------------------
kv::vector_2d
measurement_utilities
::unrectify_point(
  const kv::vector_2d& rectified_point,
  bool is_right_camera,
  const kv::simple_camera_perspective& camera ) const
{
  // Select appropriate matrices
  const cv::Mat& R = is_right_camera ? m_R2 : m_R1;
  const cv::Mat& P = is_right_camera ? m_P2 : m_P1;
  const cv::Mat& K = is_right_camera ? m_K2 : m_K1;

  // Convert point to homogeneous coordinates
  cv::Mat point_rect = ( cv::Mat_<double>( 3, 1 ) <<
    rectified_point.x(), rectified_point.y(), 1.0 );

  // Get the 3x3 portion of P (the rectified camera matrix)
  cv::Mat K_rect = P( cv::Rect( 0, 0, 3, 3 ) );

  // Convert to normalized rectified coordinates
  cv::Mat norm_rect = K_rect.inv() * point_rect;

  // Apply inverse rotation to get back to original camera frame
  cv::Mat norm_orig = R.t() * norm_rect;

  // Get normalized coordinates
  double x_norm = norm_orig.at<double>( 0, 0 ) / norm_orig.at<double>( 2, 0 );
  double y_norm = norm_orig.at<double>( 1, 0 ) / norm_orig.at<double>( 2, 0 );

  // Apply distortion model if enabled
  auto intrinsics = camera.get_intrinsics();

  if( m_use_distortion )
  {
    std::vector<double> dist_coeffs = intrinsics->dist_coeffs();

    if( !dist_coeffs.empty() )
    {
      double k1 = dist_coeffs.size() > 0 ? dist_coeffs[0] : 0.0;
      double k2 = dist_coeffs.size() > 1 ? dist_coeffs[1] : 0.0;
      double p1 = dist_coeffs.size() > 2 ? dist_coeffs[2] : 0.0;
      double p2 = dist_coeffs.size() > 3 ? dist_coeffs[3] : 0.0;
      double k3 = dist_coeffs.size() > 4 ? dist_coeffs[4] : 0.0;

      double r2 = x_norm * x_norm + y_norm * y_norm;
      double r4 = r2 * r2;
      double r6 = r2 * r4;

      double radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

      double x_tangential = 2.0 * p1 * x_norm * y_norm + p2 * ( r2 + 2.0 * x_norm * x_norm );
      double y_tangential = p1 * ( r2 + 2.0 * y_norm * y_norm ) + 2.0 * p2 * x_norm * y_norm;

      x_norm = x_norm * radial_distortion + x_tangential;
      y_norm = y_norm * radial_distortion + y_tangential;
    }
  }

  // Project back using original camera matrix
  Eigen::Matrix3d K_eigen = intrinsics->as_matrix();
  double fx = K_eigen( 0, 0 );
  double fy = K_eigen( 1, 1 );
  double cx = K_eigen( 0, 2 );
  double cy = K_eigen( 1, 2 );

  double x_distorted = fx * x_norm + cx;
  double y_distorted = fy * y_norm + cy;

  return kv::vector_2d( x_distorted, y_distorted );
}

// -----------------------------------------------------------------------------
cv::Mat
measurement_utilities
::rectify_image( const cv::Mat& image, bool is_right_camera ) const
{
  if( !m_rectification_computed )
  {
    return image.clone();
  }

  cv::Mat rectified;
  if( is_right_camera )
  {
    cv::remap( image, rectified, m_rectification_map_right_x,
               m_rectification_map_right_y, cv::INTER_LINEAR );
  }
  else
  {
    cv::remap( image, rectified, m_rectification_map_left_x,
               m_rectification_map_left_y, cv::INTER_LINEAR );
  }
  return rectified;
}

// -----------------------------------------------------------------------------
bool
measurement_utilities
::find_corresponding_point_template_matching(
  const cv::Mat& left_image_rect,
  const cv::Mat& right_image_rect,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect ) const
{
  int half_template = m_template_size / 2;
  int x_left = static_cast<int>( left_point_rect.x() );
  int y_left = static_cast<int>( left_point_rect.y() );

  // Check if template fits in left image
  if( x_left < half_template || x_left >= left_image_rect.cols - half_template ||
      y_left < half_template || y_left >= left_image_rect.rows - half_template )
  {
    return false;
  }

  // Extract template from left image
  cv::Rect template_rect( x_left - half_template, y_left - half_template,
                          m_template_size, m_template_size );
  cv::Mat template_img = left_image_rect( template_rect );

  // Define search region in right image
  int search_min_x = std::max( 0, x_left - m_search_range );
  int search_max_x = std::min( right_image_rect.cols - m_template_size, x_left );

  if( search_max_x <= search_min_x )
  {
    return false;
  }

  int search_y = std::max( half_template, std::min( y_left, right_image_rect.rows - half_template - 1 ) );

  cv::Rect search_rect( search_min_x, search_y - half_template,
                        search_max_x - search_min_x + m_template_size,
                        m_template_size );

  // Check search rect validity
  if( search_rect.x < 0 || search_rect.y < 0 ||
      search_rect.x + search_rect.width > right_image_rect.cols ||
      search_rect.y + search_rect.height > right_image_rect.rows )
  {
    return false;
  }

  cv::Mat search_region = right_image_rect( search_rect );

  // Perform template matching
  cv::Mat result;
  cv::matchTemplate( search_region, template_img, result, cv::TM_CCOEFF_NORMED );

  // Find best match
  double min_val, max_val;
  cv::Point min_loc, max_loc;
  cv::minMaxLoc( result, &min_val, &max_val, &min_loc, &max_loc );

  // Use a threshold for match quality
  if( max_val < 0.7 )
  {
    return false;
  }

  // Compute the matched point in the right image
  right_point_rect = kv::vector_2d(
    search_rect.x + max_loc.x + half_template,
    search_rect.y + max_loc.y + half_template );

  return true;
}

// -----------------------------------------------------------------------------
cv::Mat
measurement_utilities
::compute_sgbm_disparity(
  const cv::Mat& left_image_rect,
  const cv::Mat& right_image_rect )
{
  // Create SGBM matcher if not already created
  if( !m_sgbm )
  {
    m_sgbm = cv::StereoSGBM::create(
      m_sgbm_min_disparity,
      m_sgbm_num_disparities,
      m_sgbm_block_size,
      8 * m_sgbm_block_size * m_sgbm_block_size,   // P1
      32 * m_sgbm_block_size * m_sgbm_block_size,  // P2
      1,    // disp12MaxDiff
      0,    // preFilterCap
      10,   // uniquenessRatio
      100,  // speckleWindowSize
      32,   // speckleRange
      cv::StereoSGBM::MODE_SGBM_3WAY );
  }

  cv::Mat disparity;
  m_sgbm->compute( left_image_rect, right_image_rect, disparity );

  return disparity;
}

// -----------------------------------------------------------------------------
bool
measurement_utilities
::find_corresponding_point_sgbm(
  const cv::Mat& disparity_map,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect ) const
{
  int x = static_cast<int>( left_point_rect.x() + 0.5 );
  int y = static_cast<int>( left_point_rect.y() + 0.5 );

  // Check bounds
  if( x < 0 || x >= disparity_map.cols || y < 0 || y >= disparity_map.rows )
  {
    return false;
  }

  // Get disparity value (SGBM returns fixed-point values scaled by 16)
  short disp_raw = disparity_map.at<short>( y, x );

  // Check for invalid disparity
  if( disp_raw < 0 || disp_raw == ( m_sgbm_min_disparity - 1 ) * 16 )
  {
    return false;
  }

  // Convert to float disparity
  double disparity = static_cast<double>( disp_raw ) / 16.0;

  // Compute right point
  right_point_rect = kv::vector_2d( left_point_rect.x() - disparity, left_point_rect.y() );

  return true;
}

// -----------------------------------------------------------------------------
const cv::Mat&
measurement_utilities
::get_rectification_map_x( bool is_right_camera ) const
{
  return is_right_camera ? m_rectification_map_right_x : m_rectification_map_left_x;
}

// -----------------------------------------------------------------------------
const cv::Mat&
measurement_utilities
::get_rectification_map_y( bool is_right_camera ) const
{
  return is_right_camera ? m_rectification_map_right_y : m_rectification_map_left_y;
}

#endif // VIAME_ENABLE_OPENCV

// -----------------------------------------------------------------------------
std::vector< std::string >
measurement_utilities
::parse_matching_methods( const std::string& methods_str )
{
  std::vector< std::string > methods;
  std::stringstream ss( methods_str );
  std::string method;

  while( std::getline( ss, method, ',' ) )
  {
    // Trim whitespace
    size_t start = method.find_first_not_of( " \t" );
    size_t end = method.find_last_not_of( " \t" );

    if( start != std::string::npos && end != std::string::npos )
    {
      methods.push_back( method.substr( start, end - start + 1 ) );
    }
  }

  return methods;
}

// -----------------------------------------------------------------------------
bool
measurement_utilities
::method_requires_images( const std::string& method )
{
  return ( method == "template_matching" ||
           method == "sgbm_disparity" ||
           method == "feature_descriptor" ||
           method == "ransac_feature" );
}

// -----------------------------------------------------------------------------
std::vector< std::string >
measurement_utilities
::get_valid_methods()
{
  return {
    "input_pairs_only",
    "depth_projection",
    "template_matching",
    "sgbm_disparity",
    "feature_descriptor",
    "ransac_feature"
  };
}

} // end namespace core

} // end namespace viame
