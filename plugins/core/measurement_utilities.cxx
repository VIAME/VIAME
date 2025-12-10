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
