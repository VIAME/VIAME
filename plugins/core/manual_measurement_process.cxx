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
 * \brief Run manual measurement on input tracks
 */

#include "manual_measurement_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/image_container.h>
#include <vital/types/vector.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>
#include <vital/types/point.h>
#include <vital/types/bounding_box.h>
#include <vital/types/track.h>
#include <vital/util/string.h>
#include <vital/io/camera_rig_io.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_fundamental_matrix.h>

#include <arrows/mvg/triangulate.h>
#include <arrows/mvg/epipolar_geometry.h>

#include <sprokit/processes/kwiver_type_traits.h>

#ifdef VIAME_ENABLE_OPENCV
  #include <arrows/ocv/image_container.h>

  #include <opencv2/core/core.hpp>
  #include <opencv2/imgproc/imgproc.hpp>
  #include <opencv2/calib3d/calib3d.hpp>
  #include <opencv2/core/eigen.hpp>
#endif

#include <string>
#include <map>
#include <limits>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

create_config_trait( calibration_file, std::string, "",
  "Input filename for the calibration file to use" );

create_config_trait( matching_method, std::string, "automatic",
  "Method to use for finding corresponding points in right camera for left-only tracks. "
  "Options: 'input_pairs_only' (only measure tracks with keypoints in both cameras), "
  "'depth_projection' (uses default_depth to project points), "
  "'template_matching' (rectifies images and searches along epipolar lines using template matching), "
  "'sgbm_disparity' (uses Semi-Global Block Matching to compute disparity map), "
  "'feature_descriptor' (uses vital feature detection/descriptor/matching algorithms), "
  "'ransac_feature' (feature matching with RANSAC-based fundamental matrix filtering), "
  "'automatic' (uses matched tracks first, then tries template_matching, sgbm_disparity, depth_projection)" );

create_config_trait( template_size, int, "31",
  "Template window size (in pixels) for template matching. Must be odd number. "
  "Only used when matching_method is 'template_matching'" );

create_config_trait( search_range, int, "128",
  "Search range (in pixels) along epipolar line for template matching. "
  "Only used when matching_method is 'template_matching'" );

create_config_trait( default_depth, double, "5.0",
  "Default depth (in meters) to use when projecting left camera points to right camera "
  "for tracks that only exist in the left camera, when using the depth_projection option" );

create_config_trait( use_distortion, bool, "true",
  "Whether to use distortion coefficients from the calibration during rectification. "
  "If true, distortion coefficients from the calibration file are used. "
  "If false, zero distortion is assumed." );

create_config_trait( sgbm_min_disparity, int, "0",
  "Minimum possible disparity value for SGBM. Normally 0, but can be negative." );

create_config_trait( sgbm_num_disparities, int, "128",
  "Maximum disparity minus minimum disparity for SGBM. Must be divisible by 16." );

create_config_trait( sgbm_block_size, int, "5",
  "Block size for SGBM. Must be odd number >= 1. Typically 3-11." );

create_config_trait( feature_search_radius, double, "50.0",
  "Maximum distance (in pixels) to search for feature matches around the expected location. "
  "Used for feature_descriptor and ransac_feature methods." );

create_config_trait( ransac_inlier_scale, double, "3.0",
  "Inlier threshold for RANSAC fundamental matrix estimation. "
  "Points with reprojection error below this threshold are considered inliers." );

create_config_trait( min_ransac_inliers, int, "10",
  "Minimum number of inliers required for a valid RANSAC result." );

create_config_trait( box_scale_factor, double, "1.10",
  "Scale factor to expand the bounding box around keypoints when creating "
  "new detections for the right image. A value of 1.10 means 10% expansion." );

create_port_trait( object_track_set1, object_track_set,
  "The stereo filtered object tracks1.")
create_port_trait( object_track_set2, object_track_set,
  "The stereo filtered object tracks2.")

// =============================================================================
// Private implementation class
class manual_measurement_process::priv
{
public:
  explicit priv( manual_measurement_process* parent );
  ~priv();

  // Helper function to project a left camera point to the right camera
  // using the default depth
  kv::vector_2d project_left_to_right(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point );

#ifdef VIAME_ENABLE_OPENCV
  // Helper function to find corresponding point in right image using template matching
  // Returns true if match found, false otherwise
  bool find_corresponding_point_template_matching(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect );

  // Helper function to compute rectification maps
  void compute_rectification_maps(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const cv::Size& image_size );

  // Helper function to find corresponding point using SGBM disparity
  // Returns true if valid disparity found, false otherwise
  bool find_corresponding_point_sgbm(
    const cv::Mat& disparity_map,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect );

  // Helper function to compute SGBM disparity map
  cv::Mat compute_sgbm_disparity(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect );
#endif

  // Helper function to find corresponding point using vital feature detection/matching
  // Returns true if match found, false otherwise
  bool find_corresponding_point_feature_descriptor(
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    const kv::vector_2d& left_point,
    kv::vector_2d& right_point );

  // Helper function to find corresponding point using RANSAC feature matching
  // Returns true if match found, false otherwise
  bool find_corresponding_point_ransac_feature(
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    const kv::vector_2d& left_point,
    kv::vector_2d& right_point );

  // Helper function to compute a bounding box from keypoints with scale factor
  kv::bounding_box_d compute_bbox_from_keypoints(
    const kv::vector_2d& head_point,
    const kv::vector_2d& tail_point );

#ifdef VIAME_ENABLE_OPENCV
  // Helper function to unrectify a point from rectified space back to original
  kv::vector_2d unrectify_point(
    const kv::vector_2d& rectified_point,
    bool is_right_camera,
    const kv::simple_camera_perspective& camera );
#endif

  // Configuration settings
  std::string m_calibration_file;
  double m_default_depth;
  std::string m_matching_method;
  int m_template_size;
  int m_search_range;
  bool m_use_distortion;

  // SGBM configuration
  int m_sgbm_min_disparity;
  int m_sgbm_num_disparities;
  int m_sgbm_block_size;

  // Feature matching configuration
  double m_feature_search_radius;
  double m_ransac_inlier_scale;
  int m_min_ransac_inliers;

  // Right detection creation configuration
  double m_box_scale_factor;

  // Other variables
  kv::camera_rig_stereo_sptr m_calibration;
  unsigned m_frame_counter;
  std::set< std::string > p_port_list;
  manual_measurement_process* parent;

  // Optional vital algorithms for feature-based matching
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;
  kv::algo::estimate_fundamental_matrix_sptr m_fundamental_matrix_estimator;

  // Cached feature detection/descriptor results per frame
  kv::feature_set_sptr m_cached_left_features;
  kv::feature_set_sptr m_cached_right_features;
  kv::descriptor_set_sptr m_cached_left_descriptors;
  kv::descriptor_set_sptr m_cached_right_descriptors;
  kv::match_set_sptr m_cached_matches;
  kv::frame_id_t m_cached_frame_id;

#ifdef VIAME_ENABLE_OPENCV
  // Rectification maps (computed on first use)
  bool m_rectification_computed;
  cv::Mat m_rectification_map_left_x;
  cv::Mat m_rectification_map_left_y;
  cv::Mat m_rectification_map_right_x;
  cv::Mat m_rectification_map_right_y;

  // Rectification matrices for unrectifying points
  cv::Mat m_K1, m_K2, m_R1, m_R2, m_P1, m_P2;

  // SGBM matcher (created on first use)
  cv::Ptr<cv::StereoSGBM> m_sgbm;
#endif
};


// -----------------------------------------------------------------------------
manual_measurement_process::priv
::priv( manual_measurement_process* ptr )
  : m_calibration_file( "" )
  , m_default_depth( 5.0 )
  , m_matching_method( "depth_projection" )
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
  , m_calibration()
  , m_frame_counter( 0 )
  , parent( ptr )
  , m_cached_frame_id( 0 )
#ifdef VIAME_ENABLE_OPENCV
  , m_rectification_computed( false )
#endif
{
}


manual_measurement_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
#ifdef VIAME_ENABLE_OPENCV
void
manual_measurement_process::priv
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
    // Ensure we have at least 5 coefficients (k1, k2, p1, p2, k3)
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
kv::vector_2d
manual_measurement_process::priv
::unrectify_point(
  const kv::vector_2d& rectified_point,
  bool is_right_camera,
  const kv::simple_camera_perspective& camera )
{
  // Select appropriate matrices
  const cv::Mat& R = is_right_camera ? m_R2 : m_R1;
  const cv::Mat& P = is_right_camera ? m_P2 : m_P1;
  const cv::Mat& K = is_right_camera ? m_K2 : m_K1;

  // Convert point to homogeneous coordinates
  cv::Mat point_rect = ( cv::Mat_<double>( 3, 1 ) <<
    rectified_point.x(), rectified_point.y(), 1.0 );

  // Invert the projection matrix to get normalized coordinates
  // P = K * [R | t], so P^-1 gives us back to camera coordinates
  // For rectified stereo, t is typically [0,0,0] for left camera
  // We're only interested in the rotation part: K_rect = P[:, :3]
  cv::Mat K_rect = P( cv::Rect( 0, 0, 3, 3 ) );

  // Convert to normalized rectified coordinates
  cv::Mat norm_rect = K_rect.inv() * point_rect;

  // Apply inverse rotation to get back to original camera frame
  cv::Mat norm_orig = R.t() * norm_rect;

  // Get normalized coordinates (before applying camera matrix and distortion)
  double x_norm = norm_orig.at<double>( 0, 0 ) / norm_orig.at<double>( 2, 0 );
  double y_norm = norm_orig.at<double>( 1, 0 ) / norm_orig.at<double>( 2, 0 );

  // Apply distortion model if enabled and distortion coefficients exist
  auto intrinsics = camera.get_intrinsics();

  if( m_use_distortion )
  {
    std::vector<double> dist_coeffs = intrinsics->dist_coeffs();

    if( !dist_coeffs.empty() )
    {
      // Extract distortion coefficients (OpenCV distortion model)
      double k1 = dist_coeffs.size() > 0 ? dist_coeffs[0] : 0.0;
      double k2 = dist_coeffs.size() > 1 ? dist_coeffs[1] : 0.0;
      double p1 = dist_coeffs.size() > 2 ? dist_coeffs[2] : 0.0;
      double p2 = dist_coeffs.size() > 3 ? dist_coeffs[3] : 0.0;
      double k3 = dist_coeffs.size() > 4 ? dist_coeffs[4] : 0.0;

      // Compute r^2
      double r2 = x_norm * x_norm + y_norm * y_norm;
      double r4 = r2 * r2;
      double r6 = r2 * r4;

      // Radial distortion
      double radial_distortion = 1.0 + k1 * r2 + k2 * r4 + k3 * r6;

      // Tangential distortion
      double x_tangential = 2.0 * p1 * x_norm * y_norm + p2 * ( r2 + 2.0 * x_norm * x_norm );
      double y_tangential = p1 * ( r2 + 2.0 * y_norm * y_norm ) + 2.0 * p2 * x_norm * y_norm;

      // Apply distortion
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
bool
manual_measurement_process::priv
::find_corresponding_point_template_matching(
  const cv::Mat& left_image_rect,
  const cv::Mat& right_image_rect,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect )
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

  // Define search region in right image (along the same scanline for rectified images)
  // Search to the left of the left image point (disparity is typically negative for standard stereo)
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
manual_measurement_process::priv
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
manual_measurement_process::priv
::find_corresponding_point_sgbm(
  const cv::Mat& disparity_map,
  const kv::vector_2d& left_point_rect,
  kv::vector_2d& right_point_rect )
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

  // Compute right point (disparity is the shift from left to right)
  right_point_rect = kv::vector_2d( left_point_rect.x() - disparity, left_point_rect.y() );

  return true;
}

#endif

// -----------------------------------------------------------------------------
bool
manual_measurement_process::priv
::find_corresponding_point_feature_descriptor(
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  const kv::vector_2d& left_point,
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
      best_right_point = right_feat->loc();
      found = true;
    }
  }

  if( found )
  {
    right_point = best_right_point;
  }

  return found;
}

// -----------------------------------------------------------------------------
bool
manual_measurement_process::priv
::find_corresponding_point_ransac_feature(
  const kv::image_container_sptr& left_image,
  const kv::image_container_sptr& right_image,
  const kv::vector_2d& left_point,
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
      best_right_point = right_feat->loc();
      found = true;
    }
  }

  if( found )
  {
    right_point = best_right_point;
  }

  return found;
}

// -----------------------------------------------------------------------------
kv::vector_2d
manual_measurement_process::priv
::project_left_to_right(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point )
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
manual_measurement_process::priv
::compute_bbox_from_keypoints(
  const kv::vector_2d& head_point,
  const kv::vector_2d& tail_point )
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

// =============================================================================
manual_measurement_process
::manual_measurement_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new manual_measurement_process::priv( this ) )
{
  this->set_data_checking_level( check_none );

  make_ports();
  make_config();
}


manual_measurement_process
::~manual_measurement_process()
{
}


// -----------------------------------------------------------------------------
void
manual_measurement_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- inputs --
  declare_input_port_using_trait( timestamp, optional );

  // -- outputs --
  declare_output_port_using_trait( object_track_set1, required );
  declare_output_port_using_trait( object_track_set2, optional );
  declare_output_port_using_trait( timestamp, optional );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::make_config()
{
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( default_depth );
  declare_config_using_trait( matching_method );
  declare_config_using_trait( template_size );
  declare_config_using_trait( search_range );
  declare_config_using_trait( use_distortion );

  // SGBM configuration
  declare_config_using_trait( sgbm_min_disparity );
  declare_config_using_trait( sgbm_num_disparities );
  declare_config_using_trait( sgbm_block_size );

  // Feature matching configuration
  declare_config_using_trait( feature_search_radius );
  declare_config_using_trait( ransac_inlier_scale );
  declare_config_using_trait( min_ransac_inliers );

  // Right detection creation configuration
  declare_config_using_trait( box_scale_factor );
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_configure()
{
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_default_depth = config_value_using_trait( default_depth );
  d->m_matching_method = config_value_using_trait( matching_method );
  d->m_template_size = config_value_using_trait( template_size );
  d->m_search_range = config_value_using_trait( search_range );
  d->m_use_distortion = config_value_using_trait( use_distortion );

  if( d->m_calibration_file.empty() )
  {
    LOG_ERROR( logger(), "Calibration file not specified" );
    throw std::runtime_error( "Calibration file not specified" );
  }

  d->m_calibration = kv::read_stereo_rig( d->m_calibration_file );

  // Get camera references (needed for both matched and left-only detections)
  if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
  {
    LOG_ERROR( logger(), "Failed to load calibration file: " + d->m_calibration_file );
    throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
  }

  // SGBM configuration
  d->m_sgbm_min_disparity = config_value_using_trait( sgbm_min_disparity );
  d->m_sgbm_num_disparities = config_value_using_trait( sgbm_num_disparities );
  d->m_sgbm_block_size = config_value_using_trait( sgbm_block_size );

  // Feature matching configuration
  d->m_feature_search_radius = config_value_using_trait( feature_search_radius );
  d->m_ransac_inlier_scale = config_value_using_trait( ransac_inlier_scale );
  d->m_min_ransac_inliers = config_value_using_trait( min_ransac_inliers );

  // Right detection creation configuration
  d->m_box_scale_factor = config_value_using_trait( box_scale_factor );

  // Ensure template size is odd
  if( d->m_template_size % 2 == 0 )
  {
    d->m_template_size++;
    LOG_WARN( logger(), "Template size must be odd, adjusted to " +
                        std::to_string( d->m_template_size ) );
  }

  // Ensure SGBM num_disparities is divisible by 16
  if( d->m_sgbm_num_disparities % 16 != 0 )
  {
    d->m_sgbm_num_disparities = ( ( d->m_sgbm_num_disparities / 16 ) + 1 ) * 16;
    LOG_WARN( logger(), "SGBM num_disparities must be divisible by 16, adjusted to " +
                        std::to_string( d->m_sgbm_num_disparities ) );
  }

  // Ensure SGBM block size is odd
  if( d->m_sgbm_block_size % 2 == 0 )
  {
    d->m_sgbm_block_size++;
    LOG_WARN( logger(), "SGBM block size must be odd, adjusted to " +
                        std::to_string( d->m_sgbm_block_size ) );
  }

  // Configure optional vital algorithms for feature-based methods
  kv::config_block_sptr algo_config = get_config();

  // Try to configure feature detector (optional)
  kv::algo::detect_features::set_nested_algo_configuration(
    "feature_detector", algo_config, d->m_feature_detector );

  // Try to configure descriptor extractor (optional)
  kv::algo::extract_descriptors::set_nested_algo_configuration(
    "descriptor_extractor", algo_config, d->m_descriptor_extractor );

  // Try to configure feature matcher (optional)
  kv::algo::match_features::set_nested_algo_configuration(
    "feature_matcher", algo_config, d->m_feature_matcher );

  // Try to configure fundamental matrix estimator (optional, needed for ransac_feature)
  kv::algo::estimate_fundamental_matrix::set_nested_algo_configuration(
    "fundamental_matrix_estimator", algo_config, d->m_fundamental_matrix_estimator );

  // Check if feature-based methods are requested but algorithms are not configured
  bool needs_feature_algos = ( d->m_matching_method == "feature_descriptor" ||
                               d->m_matching_method == "ransac_feature" );

  if( needs_feature_algos )
  {
    if( !d->m_feature_detector )
    {
      LOG_WARN( logger(), "Feature detector not configured; feature_descriptor and "
                          "ransac_feature methods will not work" );
    }
    if( !d->m_descriptor_extractor )
    {
      LOG_WARN( logger(), "Descriptor extractor not configured; feature_descriptor and "
                          "ransac_feature methods will not work" );
    }
    if( !d->m_feature_matcher )
    {
      LOG_WARN( logger(), "Feature matcher not configured; feature_descriptor and "
                          "ransac_feature methods will not work" );
    }
    if( d->m_matching_method == "ransac_feature" && !d->m_fundamental_matrix_estimator )
    {
      LOG_WARN( logger(), "Fundamental matrix estimator not configured; "
                          "ransac_feature method will not work" );
    }
  }
}

// ----------------------------------------------------------------------------
void
manual_measurement_process
::_init()
{
  this->set_data_checking_level( check_valid );
}

// ----------------------------------------------------------------------------
void
manual_measurement_process
::input_port_undefined( port_t const& port_name )
{
  LOG_TRACE( logger(), "Processing undefined input port: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if( !kv::starts_with( port_name, "_" ) )
  {
    // Check for unique port name
    if( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      if( port_name.find( "image" ) != std::string::npos )
      {
        declare_input_port(
          port_name,                                 // port name
          image_port_trait::type_name,               // port type
          required,                                  // port flags
          "image container input" );
      }
      else
      {
        declare_input_port(
          port_name,                                 // port name
          object_track_set_port_trait::type_name,    // port type
          required,                                  // port flags
          "object track set input" );
      }

      d->p_port_list.insert( port_name );
    }
  }
}

// -----------------------------------------------------------------------------
void
manual_measurement_process
::_step()
{
  std::vector< kv::object_track_set_sptr > input_tracks;
  std::vector< kv::image_container_sptr > input_images;
  kv::timestamp ts;
  kv::object_track_set_sptr output;

  // Read port names allowing for an arbitrary number of inputs for multi-cam
  for( auto const& port_name : d->p_port_list )
  {
    if( port_name == "timestamp" )
    {
      ts = grab_from_port_using_trait( timestamp );
    }
    else if( port_name.find( "image" ) != std::string::npos )
    {
      input_images.push_back( 
        grab_from_port_as< kv::image_container_sptr >( port_name ) );
    }
    else
    {
      input_tracks.push_back(
        grab_from_port_as< kv::object_track_set_sptr >( port_name ) );
    }
  }

  kv::frame_id_t cur_frame_id = ( ts.has_valid_frame() ?
                                  ts.get_frame() :
                                  d->m_frame_counter );

  d->m_frame_counter++;

  if( input_tracks.size() != 2 )
  {
    const std::string err = "Currently only 2 camera inputs are supported";
    LOG_ERROR( logger(), err );
    throw std::runtime_error( err );
  }

  // Identify all input detections across all track sets on the current frame
  typedef std::vector< std::map< kv::track_id_t, kv::detected_object_sptr > > map_t;

  map_t dets( input_tracks.size() );

  for( unsigned i = 0; i < input_tracks.size(); ++i )
  {
    if( !input_tracks[i] )
    {
      continue;
    }

    for( auto& trk : input_tracks[i]->tracks() )
    {
      for( auto& state : *trk )
      {
        auto obj_state =
          std::static_pointer_cast< kwiver::vital::object_track_state >( state );

        if( state->frame() == cur_frame_id )
        {
          dets[i][trk->id()] = obj_state->detection();
        }
      }
    }
  }

  // Identify which detections are matched on the current frame
  std::vector< kv::track_id_t > common_ids;
  std::vector< kv::track_id_t > left_only_ids;

  for( auto itr : dets[0] )
  {
    bool found_match = false;

    for( unsigned i = 1; i < input_tracks.size(); ++i )
    {
      if( dets[i].find( itr.first ) != dets[i].end() )
      {
        found_match = true;
        common_ids.push_back( itr.first );
        break;
      }
    }

    if( found_match )
    {
      LOG_INFO( logger(), "Found match for track ID " + std::to_string( itr.first ) );
    }
    else
    {
      if( d->m_matching_method == "input_pairs_only" )
      {
        LOG_INFO( logger(), "No match for track ID " + std::to_string( itr.first ) +
                            ", skipping (input_pairs_only mode)" );
      }
      else
      {
        LOG_INFO( logger(), "No match for track ID " + std::to_string( itr.first ) +
                            ", will compute right camera points using " + d->m_matching_method );
        left_only_ids.push_back( itr.first );
      }
    }
  }

  // Get camera references (needed for both matched and left-only detections)
  kv::simple_camera_perspective& left_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->left())));
  kv::simple_camera_perspective& right_cam(
    dynamic_cast< kwiver::vital::simple_camera_perspective& >(
      *(d->m_calibration->right())));

  // Separate matched detections into those with full keypoints and those missing right keypoints
  std::vector< kv::track_id_t > fully_matched_ids;
  std::vector< kv::track_id_t > missing_right_kps_ids;

  for( const kv::track_id_t& id : common_ids )
  {
    const auto& det1 = dets[0][id];
    const auto& det2 = dets[1][id];

    if( !det1 || !det2 )
    {
      continue;
    }

    const auto& kp1 = det1->keypoints();
    const auto& kp2 = det2->keypoints();

    bool left_has_kp = ( kp1.find( "head" ) != kp1.end() &&
                         kp1.find( "tail" ) != kp1.end() );
    bool right_has_kp = ( kp2.find( "head" ) != kp2.end() &&
                          kp2.find( "tail" ) != kp2.end() );

    if( left_has_kp && right_has_kp )
    {
      fully_matched_ids.push_back( id );
    }
    else if( left_has_kp && !right_has_kp )
    {
      if( d->m_matching_method == "input_pairs_only" )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " right detection missing keypoints, skipping (input_pairs_only mode)" );
      }
      else
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                            " right detection missing keypoints, will compute using " + d->m_matching_method );
        missing_right_kps_ids.push_back( id );
      }
    }
  }

  // Run measurement on fully matched detections (both have keypoints)
  if( !fully_matched_ids.empty() )
  {
    for( const kv::track_id_t& id : fully_matched_ids )
    {
      const auto& det1 = dets[0][id];
      const auto& det2 = dets[1][id];

      const auto& kp1 = det1->keypoints();
      const auto& kp2 = det2->keypoints();

      // Triangulate head keypoint across both cameras
      Eigen::Matrix<float, 2, 1>
        left_head( kp1.at("head")[0], kp1.at("head")[1] ),
        right_head( kp2.at("head")[0], kp2.at("head")[1] );
      auto pp1 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_head, right_head );

      // Triangulate tail keypoint across both cameras
      Eigen::Matrix<float, 2, 1>
        left_tail( kp1.at("tail")[0], kp1.at("tail")[1] ),
        right_tail( kp2.at("tail")[0], kp2.at("tail")[1] );
      auto pp2 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_tail, right_tail );

      const double length = ( pp2 - pp1 ).norm();

      LOG_INFO( logger(), "Computed Length: " + std::to_string( length ) );

      det1->set_length( length );
      det2->set_length( length );
    }
  }

  // Combine left-only IDs and matched IDs missing right keypoints for secondary matching
  std::vector< kv::track_id_t > ids_needing_matching;
  ids_needing_matching.insert( ids_needing_matching.end(),
                               left_only_ids.begin(), left_only_ids.end() );
  ids_needing_matching.insert( ids_needing_matching.end(),
                               missing_right_kps_ids.begin(),
                               missing_right_kps_ids.end() );

  // Run measurement on detections needing secondary matching (skip if input_pairs_only mode)
  if( !ids_needing_matching.empty() && d->m_matching_method != "input_pairs_only" )
  {
    // Clear cached feature data if frame changed
    if( d->m_cached_frame_id != cur_frame_id )
    {
      d->m_cached_left_features.reset();
      d->m_cached_right_features.reset();
      d->m_cached_left_descriptors.reset();
      d->m_cached_right_descriptors.reset();
      d->m_cached_matches.reset();
      d->m_cached_frame_id = cur_frame_id;
    }

    // Determine which methods to use
    bool is_automatic = ( d->m_matching_method == "automatic" );

    bool use_template_matching =
      ( d->m_matching_method == "template_matching" || is_automatic );

    bool use_sgbm_disparity =
      ( d->m_matching_method == "sgbm_disparity" || is_automatic );

    bool use_feature_descriptor =
      ( d->m_matching_method == "feature_descriptor" );

    bool use_ransac_feature = ( d->m_matching_method == "ransac_feature" );

    bool needs_rectified_images = use_template_matching || use_sgbm_disparity;

#ifdef VIAME_ENABLE_OPENCV
    // Prepare rectified images if needed
    cv::Mat left_image_rect, right_image_rect;
    cv::Mat disparity_map;
    bool rectified_images_available = false;
    if( needs_rectified_images && input_images.size() >= 2 )
    {
      // Convert to OpenCV format and grayscale
      cv::Mat left_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
        input_images[0]->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
      cv::Mat right_cv = kwiver::arrows::ocv::image_container::vital_to_ocv(
        input_images[1]->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

      // Convert to grayscale if needed
      if( left_cv.channels() > 1 )
      {
        cv::cvtColor( left_cv, left_cv, cv::COLOR_BGR2GRAY );
        cv::cvtColor( right_cv, right_cv, cv::COLOR_BGR2GRAY );
      }

      // Compute rectification maps if needed
      d->compute_rectification_maps( left_cam, right_cam, left_cv.size() );

      // Rectify images
      cv::remap( left_cv, left_image_rect, d->m_rectification_map_left_x,
                 d->m_rectification_map_left_y, cv::INTER_LINEAR );
      cv::remap( right_cv, right_image_rect, d->m_rectification_map_right_x,
                 d->m_rectification_map_right_y, cv::INTER_LINEAR );

      rectified_images_available = true;

      // Compute SGBM disparity map for automatic or sgbm_disparity mode
      if( use_sgbm_disparity )
      {
        disparity_map = d->compute_sgbm_disparity( left_image_rect, right_image_rect );
      }
    }
    else if( needs_rectified_images )
    {
      if( is_automatic )
      {
        LOG_INFO( logger(), "Images not provided, automatic mode will use depth projection" );
      }
      else if( d->m_matching_method == "template_matching" ||
               d->m_matching_method == "sgbm_disparity" )
      {
        LOG_WARN( logger(), d->m_matching_method + " requested but images not provided, "
                            "falling back to depth projection" );
      }
    }
#endif

    for( const kv::track_id_t& id : ids_needing_matching )
    {
      const auto& det1 = dets[0][id];

      if( !det1 )
      {
        continue;
      }

      const auto& kp1 = det1->keypoints();

      if( kp1.find( "head" ) == kp1.end() ||
          kp1.find( "tail" ) == kp1.end() )
      {
        LOG_INFO( logger(), "Track ID " + std::to_string( id ) + " " +
                            "missing required keypoints (head/tail)" );
        continue;
      }

      // Check if this is a left-only track or a matched track missing right keypoints
      bool is_left_only = ( dets[1].find( id ) == dets[1].end() );

      kv::vector_2d left_head_point( kp1.at("head")[0], kp1.at("head")[1] );
      kv::vector_2d left_tail_point( kp1.at("tail")[0], kp1.at("tail")[1] );
      kv::vector_2d right_head_point, right_tail_point;
      bool head_found = false, tail_found = false;
      std::string method_used = "depth_projection";

      // Try feature_descriptor method (only if explicitly selected)
      if( use_feature_descriptor && input_images.size() >= 2 )
      {
        head_found = d->find_corresponding_point_feature_descriptor(
          input_images[0], input_images[1], left_head_point, right_head_point );
        tail_found = d->find_corresponding_point_feature_descriptor(
          input_images[0], input_images[1], left_tail_point, right_tail_point );

        if( head_found && tail_found )
        {
          method_used = "feature_descriptor";
        }
        else
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " feature_descriptor matching failed" );
        }
      }

      // Try ransac_feature method (only if explicitly selected)
      if( use_ransac_feature && input_images.size() >= 2 )
      {
        head_found = d->find_corresponding_point_ransac_feature(
          input_images[0], input_images[1], left_head_point, right_head_point );
        tail_found = d->find_corresponding_point_ransac_feature(
          input_images[0], input_images[1], left_tail_point, right_tail_point );

        if( head_found && tail_found )
        {
          method_used = "ransac_feature";
        }
        else
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " ransac_feature matching failed" );
        }
      }

#ifdef VIAME_ENABLE_OPENCV
      // For automatic mode: try template_matching first, then sgbm_disparity
      // For specific methods: only try the selected method

      // Try template_matching method (first choice for automatic mode)
      if( use_template_matching && rectified_images_available &&
          ( !head_found || !tail_found ) )
      {
        // Rectify left keypoints using remap maps
        int x_head = static_cast<int>( left_head_point.x() + 0.5 );
        int y_head = static_cast<int>( left_head_point.y() + 0.5 );
        int x_tail = static_cast<int>( left_tail_point.x() + 0.5 );
        int y_tail = static_cast<int>( left_tail_point.y() + 0.5 );

        // Check bounds
        bool keypoints_in_bounds =
          !( x_head < 0 || x_head >= d->m_rectification_map_left_x.cols ||
             y_head < 0 || y_head >= d->m_rectification_map_left_x.rows ||
             x_tail < 0 || x_tail >= d->m_rectification_map_left_x.cols ||
             y_tail < 0 || y_tail >= d->m_rectification_map_left_x.rows );

        if( keypoints_in_bounds )
        {
          float rect_x_head = d->m_rectification_map_left_x.at<float>( y_head, x_head );
          float rect_y_head = d->m_rectification_map_left_y.at<float>( y_head, x_head );
          float rect_x_tail = d->m_rectification_map_left_x.at<float>( y_tail, x_tail );
          float rect_y_tail = d->m_rectification_map_left_y.at<float>( y_tail, x_tail );

          kv::vector_2d left_head_rect( rect_x_head, rect_y_head );
          kv::vector_2d left_tail_rect( rect_x_tail, rect_y_tail );

          // Find corresponding points in right image using template matching
          kv::vector_2d right_head_rect, right_tail_rect;
          head_found = d->find_corresponding_point_template_matching(
            left_image_rect, right_image_rect, left_head_rect, right_head_rect );
          tail_found = d->find_corresponding_point_template_matching(
            left_image_rect, right_image_rect, left_tail_rect, right_tail_rect );

          if( head_found && tail_found )
          {
            // Unrectify right points back to original image coordinates
            right_head_point = d->unrectify_point( right_head_rect, true, right_cam );
            right_tail_point = d->unrectify_point( right_tail_rect, true, right_cam );
            method_used = "template_matching";
          }
          else
          {
            LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                                " template_matching failed" );
          }
        }
        else
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " keypoints out of bounds for template_matching" );
        }
      }

      // Try SGBM disparity method (second choice for automatic mode)
      if( use_sgbm_disparity && rectified_images_available && !disparity_map.empty() &&
          ( !head_found || !tail_found ) )
      {
        // Rectify left keypoints
        int x_head = static_cast<int>( left_head_point.x() + 0.5 );
        int y_head = static_cast<int>( left_head_point.y() + 0.5 );
        int x_tail = static_cast<int>( left_tail_point.x() + 0.5 );
        int y_tail = static_cast<int>( left_tail_point.y() + 0.5 );

        bool keypoints_in_bounds =
          !( x_head < 0 || x_head >= d->m_rectification_map_left_x.cols ||
             y_head < 0 || y_head >= d->m_rectification_map_left_x.rows ||
             x_tail < 0 || x_tail >= d->m_rectification_map_left_x.cols ||
             y_tail < 0 || y_tail >= d->m_rectification_map_left_x.rows );

        if( keypoints_in_bounds )
        {
          float rect_x_head = d->m_rectification_map_left_x.at<float>( y_head, x_head );
          float rect_y_head = d->m_rectification_map_left_y.at<float>( y_head, x_head );
          float rect_x_tail = d->m_rectification_map_left_x.at<float>( y_tail, x_tail );
          float rect_y_tail = d->m_rectification_map_left_y.at<float>( y_tail, x_tail );

          kv::vector_2d left_head_rect( rect_x_head, rect_y_head );
          kv::vector_2d left_tail_rect( rect_x_tail, rect_y_tail );

          kv::vector_2d right_head_rect, right_tail_rect;
          head_found = d->find_corresponding_point_sgbm(
            disparity_map, left_head_rect, right_head_rect );
          tail_found = d->find_corresponding_point_sgbm(
            disparity_map, left_tail_rect, right_tail_rect );

          if( head_found && tail_found )
          {
            // Unrectify right points back to original image coordinates
            right_head_point = d->unrectify_point( right_head_rect, true, right_cam );
            right_tail_point = d->unrectify_point( right_tail_rect, true, right_cam );
            method_used = "sgbm_disparity";
          }
          else
          {
            LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                                " sgbm_disparity lookup failed (invalid disparity)" );
          }
        }
        else
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " keypoints out of bounds for SGBM" );
        }
      }
#endif

      // Fall back to depth projection if nothing else worked (last choice for automatic mode)
      if( !head_found || !tail_found )
      {
        if( is_automatic || d->m_matching_method == "depth_projection" )
        {
          right_head_point = d->project_left_to_right( left_cam, right_cam, left_head_point );
          right_tail_point = d->project_left_to_right( left_cam, right_cam, left_tail_point );
          head_found = true;
          tail_found = true;
          method_used = "depth_projection";

          if( is_automatic )
          {
            LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                                " falling back to depth_projection" );
          }
        }
        else
        {
          LOG_INFO( logger(), "Track ID " + std::to_string( id ) +
                              " " + d->m_matching_method + " failed, skipping" );
          continue;
        }
      }

      // Triangulate head keypoint
      Eigen::Matrix<float, 2, 1>
        left_head( kp1.at("head")[0], kp1.at("head")[1] ),
        right_head( right_head_point.x(), right_head_point.y() );
      auto pp1 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_head, right_head );

      // Triangulate tail keypoint
      Eigen::Matrix<float, 2, 1>
        left_tail( kp1.at("tail")[0], kp1.at("tail")[1] ),
        right_tail( right_tail_point.x(), right_tail_point.y() );
      auto pp2 = kwiver::arrows::mvg::triangulate_fast_two_view(
        left_cam, right_cam, left_tail, right_tail );

      const double length = ( pp2 - pp1 ).norm();

      LOG_INFO( logger(), "Computed Length (" + method_used + "): " +
                          std::to_string( length ) );

      det1->set_length( length );

      if( is_left_only )
      {
        // Create a new detection for the right image with the computed keypoints
        kv::bounding_box_d right_bbox =
          d->compute_bbox_from_keypoints( right_head_point, right_tail_point );

        auto det2 = std::make_shared< kv::detected_object >( right_bbox );
        det2->add_keypoint( "head", kv::point_2d( right_head_point.x(), right_head_point.y() ) );
        det2->add_keypoint( "tail", kv::point_2d( right_tail_point.x(), right_tail_point.y() ) );
        det2->set_length( length );

        // Ensure right track set exists
        if( !input_tracks[1] )
        {
          input_tracks[1] = std::make_shared< kv::object_track_set >();
        }

        // Find or create the track with the same ID in the right track set
        kv::track_sptr right_track = input_tracks[1]->get_track( id );

        if( !right_track )
        {
          right_track = kv::track::create();
          right_track->set_id( id );
          input_tracks[1]->insert( right_track );
        }

        // Create and append a new track state with the detection
        kv::time_usec_t time_usec = ts.has_valid_time() ? ts.get_time_usec() : 0;
        auto new_state = std::make_shared< kv::object_track_state >(
          cur_frame_id, time_usec, det2 );
        right_track->append( new_state );
        input_tracks[1]->notify_new_state( new_state );

        LOG_INFO( logger(), "Created right detection for track ID " + std::to_string( id ) );
      }
      else
      {
        // Existing right detection - add keypoints to it
        auto& det2 = dets[1][id];
        det2->add_keypoint( "head", kv::point_2d( right_head_point.x(), right_head_point.y() ) );
        det2->add_keypoint( "tail", kv::point_2d( right_tail_point.x(), right_tail_point.y() ) );
        det2->set_length( length );

        LOG_INFO( logger(), "Added keypoints to existing right detection for track ID " +
                            std::to_string( id ) );
      }
    }
  }

  // Ensure output track sets exist before pushing
  if( !input_tracks[0] )
  {
    input_tracks[0] = std::make_shared< kv::object_track_set >();
  }
  if( !input_tracks[1] )
  {
    input_tracks[1] = std::make_shared< kv::object_track_set >();
  }

  // Push outputs
  push_to_port_using_trait( object_track_set1, input_tracks[0] );
  push_to_port_using_trait( object_track_set2, input_tracks[1] );
  push_to_port_using_trait( timestamp, ts );
}

} // end namespace core

} // end namespace viame
