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
 * \brief Stereo measurement utility functions
 */

#ifndef VIAME_CORE_MEASUREMENT_UTILITIES_H
#define VIAME_CORE_MEASUREMENT_UTILITIES_H

#include <plugins/core/viame_core_export.h>

#include <vital/types/vector.h>
#include <vital/types/bounding_box.h>
#include <vital/types/image_container.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>
#include <vital/config/config_block.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_fundamental_matrix.h>

#ifdef VIAME_ENABLE_OPENCV
  #include <opencv2/core/core.hpp>
  #include <opencv2/calib3d/calib3d.hpp>
#endif

#include <string>
#include <vector>
#include <memory>

namespace viame
{

namespace core
{

namespace kv = kwiver::vital;

/**
 * \brief Configuration settings for stereo measurement operations
 *
 * This class encapsulates all configuration parameters for stereo measurement
 * and provides standard kwiver get/set configuration functions.
 */
class VIAME_CORE_EXPORT measurement_settings
{
public:
  measurement_settings();
  ~measurement_settings();

  // -------------------------------------------------------------------------
  // Standard kwiver configuration functions
  // -------------------------------------------------------------------------

  /// Get the default configuration block with all parameters
  kv::config_block_sptr get_configuration() const;

  /// Set configuration from a config block
  void set_configuration( kv::config_block_sptr config );

  /// Check if configuration is valid
  bool check_configuration( kv::config_block_sptr config ) const;

  // -------------------------------------------------------------------------
  // Parameter values
  // -------------------------------------------------------------------------

  /// Comma-separated list of matching methods to try in order
  std::string matching_methods;

  /// Default depth (in meters) for depth projection method
  double default_depth;

  /// Template window size (in pixels) for template matching. Must be odd.
  int template_size;

  /// Search range (in pixels) along epipolar line for template matching
  int search_range;

  /// Whether to use distortion coefficients from calibration
  bool use_distortion;

  /// Minimum disparity value for SGBM
  int sgbm_min_disparity;

  /// Maximum disparity minus minimum disparity for SGBM. Must be divisible by 16.
  int sgbm_num_disparities;

  /// Block size for SGBM. Must be odd >= 1.
  int sgbm_block_size;

  /// Maximum distance (in pixels) to search for feature matches
  double feature_search_radius;

  /// Inlier threshold for RANSAC fundamental matrix estimation
  double ransac_inlier_scale;

  /// Minimum number of inliers required for valid RANSAC result
  int min_ransac_inliers;

  /// Scale factor to expand bounding box around keypoints
  double box_scale_factor;

  /// Whether to record stereo measurement method as detection attribute
  bool record_stereo_method;

  // -------------------------------------------------------------------------
  // Algorithm pointers (configured via nested algo configuration)
  // -------------------------------------------------------------------------

  /// Feature detector algorithm
  kv::algo::detect_features_sptr feature_detector;

  /// Descriptor extractor algorithm
  kv::algo::extract_descriptors_sptr descriptor_extractor;

  /// Feature matcher algorithm
  kv::algo::match_features_sptr feature_matcher;

  /// Fundamental matrix estimator for RANSAC filtering
  kv::algo::estimate_fundamental_matrix_sptr fundamental_matrix_estimator;

  // -------------------------------------------------------------------------
  // Utility methods
  // -------------------------------------------------------------------------

  /// Get the parsed list of matching methods
  std::vector< std::string > get_matching_methods() const;

  /// Validate matching methods and return error message (empty if valid)
  std::string validate_matching_methods() const;

  /// Check if any configured method requires images
  bool any_method_requires_images() const;

  /// Check if feature algorithms are properly configured for the specified methods
  std::vector< std::string > check_feature_algorithm_warnings() const;
};


/**
 * \brief Stereo measurement utility class
 *
 * This class provides helper functions for stereo measurement operations
 * including point projection, template matching, SGBM disparity, and
 * feature-based correspondence finding.
 */
class VIAME_CORE_EXPORT measurement_utilities
{
public:
  measurement_utilities();
  ~measurement_utilities();

  // -------------------------------------------------------------------------
  // Configuration from settings
  // -------------------------------------------------------------------------

  /// Configure all parameters from a measurement_settings object
  void configure( const measurement_settings& settings );

  // -------------------------------------------------------------------------
  // Configuration setters (for individual parameter control)
  // -------------------------------------------------------------------------

  /// Set the default depth for depth projection method
  void set_default_depth( double depth );

  /// Set template matching parameters
  void set_template_params( int template_size, int search_range );

  /// Set whether to use distortion coefficients
  void set_use_distortion( bool use_distortion );

  /// Set SGBM parameters
  void set_sgbm_params( int min_disparity, int num_disparities, int block_size );

  /// Set feature matching parameters
  void set_feature_params( double search_radius, double ransac_inlier_scale,
                           int min_ransac_inliers );

  /// Set bounding box scale factor for creating detections from keypoints
  void set_box_scale_factor( double scale_factor );

  /// Set feature detection algorithms
  void set_feature_algorithms(
    kv::algo::detect_features_sptr detector,
    kv::algo::extract_descriptors_sptr extractor,
    kv::algo::match_features_sptr matcher,
    kv::algo::estimate_fundamental_matrix_sptr fundamental_estimator = nullptr );

  // -------------------------------------------------------------------------
  // Core utility functions
  // -------------------------------------------------------------------------

  /// Project a left camera point to the right camera using the default depth
  kv::vector_2d project_left_to_right(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point ) const;

  /// Compute a bounding box from keypoints with scale factor
  kv::bounding_box_d compute_bbox_from_keypoints(
    const kv::vector_2d& head_point,
    const kv::vector_2d& tail_point ) const;

  /// Triangulate a point from stereo correspondences and compute 3D position
  kv::vector_3d triangulate_point(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point,
    const kv::vector_2d& right_point ) const;

  /// Compute length between two 3D points from stereo keypoint pairs
  double compute_stereo_length(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_head,
    const kv::vector_2d& right_head,
    const kv::vector_2d& left_tail,
    const kv::vector_2d& right_tail ) const;

  // -------------------------------------------------------------------------
  // High-level stereo correspondence functions
  // -------------------------------------------------------------------------

  /// Result structure for stereo correspondence finding
  struct stereo_correspondence_result
  {
    bool success;
    kv::vector_2d left_head;
    kv::vector_2d left_tail;
    kv::vector_2d right_head;
    kv::vector_2d right_tail;
    std::string method_used;
  };

  /// Find stereo correspondences using specified methods in order
  /// Tries each method until one succeeds for both head and tail points
  stereo_correspondence_result find_stereo_correspondence(
    const std::vector< std::string >& methods,
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_head,
    const kv::vector_2d& left_tail,
    const kv::vector_2d* right_head_input,
    const kv::vector_2d* right_tail_input,
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image );

#ifdef VIAME_ENABLE_OPENCV
  /// Prepare stereo images for matching (convert to grayscale, rectify, compute disparity)
  struct stereo_image_data
  {
    cv::Mat left_rectified;
    cv::Mat right_rectified;
    cv::Mat disparity_map;
    bool rectified_available;
    bool disparity_available;
  };

  /// Prepare stereo images for the specified methods
  stereo_image_data prepare_stereo_images(
    const std::vector< std::string >& methods,
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image );
#endif

  // -------------------------------------------------------------------------
  // Feature-based matching functions
  // -------------------------------------------------------------------------

  /// Find corresponding point using vital feature detection/matching
  /// Returns true if match found, false otherwise
  /// Updates left_point to the actual feature location if a match is found
  bool find_corresponding_point_feature_descriptor(
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    kv::vector_2d& left_point,
    kv::vector_2d& right_point );

  /// Find corresponding point using RANSAC feature matching
  /// Returns true if match found, false otherwise
  /// Updates left_point to the actual feature location if a match is found
  bool find_corresponding_point_ransac_feature(
    const kv::image_container_sptr& left_image,
    const kv::image_container_sptr& right_image,
    kv::vector_2d& left_point,
    kv::vector_2d& right_point );

  /// Clear cached feature data (call when frame changes)
  void clear_feature_cache();

  /// Set the current frame ID for caching
  void set_frame_id( kv::frame_id_t frame_id );

#ifdef VIAME_ENABLE_OPENCV
  // -------------------------------------------------------------------------
  // OpenCV-based matching functions
  // -------------------------------------------------------------------------

  /// Compute rectification maps for stereo images
  void compute_rectification_maps(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const cv::Size& image_size );

  /// Check if rectification maps have been computed
  bool rectification_computed() const;

  /// Rectify a point from original image space to rectified space
  kv::vector_2d rectify_point(
    const kv::vector_2d& original_point,
    bool is_right_camera ) const;

  /// Unrectify a point from rectified space back to original image space
  kv::vector_2d unrectify_point(
    const kv::vector_2d& rectified_point,
    bool is_right_camera,
    const kv::simple_camera_perspective& camera ) const;

  /// Rectify an image using precomputed maps
  cv::Mat rectify_image( const cv::Mat& image, bool is_right_camera ) const;

  /// Find corresponding point in right image using template matching
  /// Returns true if match found, false otherwise
  bool find_corresponding_point_template_matching(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect ) const;

  /// Compute SGBM disparity map
  cv::Mat compute_sgbm_disparity(
    const cv::Mat& left_image_rect,
    const cv::Mat& right_image_rect );

  /// Find corresponding point using SGBM disparity
  /// Returns true if valid disparity found, false otherwise
  bool find_corresponding_point_sgbm(
    const cv::Mat& disparity_map,
    const kv::vector_2d& left_point_rect,
    kv::vector_2d& right_point_rect ) const;

  /// Get rectification map for a given camera (for external use)
  const cv::Mat& get_rectification_map_x( bool is_right_camera ) const;
  const cv::Mat& get_rectification_map_y( bool is_right_camera ) const;
#endif

  // -------------------------------------------------------------------------
  // Method name parsing utilities
  // -------------------------------------------------------------------------

  /// Parse a comma-separated list of matching methods
  static std::vector< std::string > parse_matching_methods( const std::string& methods_str );

  /// Check if a method requires images
  static bool method_requires_images( const std::string& method );

  /// Get list of all valid method names
  static std::vector< std::string > get_valid_methods();

private:
  // Configuration
  double m_default_depth;
  int m_template_size;
  int m_search_range;
  bool m_use_distortion;
  int m_sgbm_min_disparity;
  int m_sgbm_num_disparities;
  int m_sgbm_block_size;
  double m_feature_search_radius;
  double m_ransac_inlier_scale;
  int m_min_ransac_inliers;
  double m_box_scale_factor;

  // Feature algorithms
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
  // Rectification maps
  bool m_rectification_computed;
  cv::Mat m_rectification_map_left_x;
  cv::Mat m_rectification_map_left_y;
  cv::Mat m_rectification_map_right_x;
  cv::Mat m_rectification_map_right_y;

  // Rectification matrices for unrectifying points
  cv::Mat m_K1, m_K2, m_R1, m_R2, m_P1, m_P2, m_D1, m_D2;

  // SGBM matcher
  cv::Ptr<cv::StereoSGBM> m_sgbm;

  // Cached stereo image data
  stereo_image_data m_cached_stereo_images;
#endif
};

} // end namespace core

} // end namespace viame

#endif // VIAME_CORE_MEASUREMENT_UTILITIES_H
