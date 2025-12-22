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
 * \brief Stereo detection pairing process implementation
 */

#include "pair_stereo_detections_process.h"
#include "measurement_utilities.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/detected_object_set.h>
#include <vital/types/object_track_set.h>
#include <vital/types/bounding_box.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/image.h>
#include <vital/types/image_container.h>
#include <vital/types/feature.h>
#include <vital/types/feature_set.h>
#include <vital/types/descriptor_set.h>
#include <vital/types/match_set.h>
#include <vital/io/camera_rig_io.h>

#include <vital/algo/detect_features.h>
#include <vital/algo/extract_descriptors.h>
#include <vital/algo/match_features.h>
#include <vital/algo/estimate_homography.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <algorithm>
#include <limits>
#include <cmath>
#include <map>
#include <set>

namespace kv = kwiver::vital;

namespace viame
{

namespace core
{

// Config traits
create_config_trait( matching_method, std::string, "iou",
  "Matching method to use. Options: 'iou' (bounding box overlap), "
  "'calibration' (uses stereo geometry and reprojection error), "
  "'feature_matching' (uses feature detection and matching within bounding boxes)" );

create_config_trait( calibration_file, std::string, "",
  "Stereo calibration file (JSON format). Required when matching_method is 'calibration'." );

create_config_trait( iou_threshold, double, "0.1",
  "Minimum IOU (Intersection over Union) threshold for matching detections. "
  "Used with 'iou' method. Pairs with IOU below this value will not be matched." );

create_config_trait( max_reprojection_error, double, "10.0",
  "Maximum reprojection error (in pixels) for valid matches. "
  "Used with 'calibration' method." );

create_config_trait( default_depth, double, "5.0",
  "Default depth (in meters) for projecting points between cameras. "
  "Used with 'calibration' method to estimate initial correspondence." );

create_config_trait( require_class_match, bool, "true",
  "If true, only detections with the same class label can be matched." );

create_config_trait( use_optimal_assignment, bool, "true",
  "If true, use optimal (greedy) assignment to maximize matching quality. "
  "If false, use simple greedy matching in order of left detections." );

create_config_trait( output_unmatched, bool, "true",
  "If true, output unmatched detections as separate tracks with unique IDs. "
  "If false, only output matched detection pairs." );

// Feature matching config traits
create_config_trait( min_feature_match_count, int, "5",
  "Minimum number of feature matches required between two detection boxes "
  "to consider them a valid match. Used with 'feature_matching' method." );

create_config_trait( min_feature_match_ratio, double, "0.1",
  "Minimum ratio of feature matches to total features detected in the left box. "
  "A ratio of 0.1 means at least 10% of left features must match. "
  "Used with 'feature_matching' method." );

create_config_trait( use_homography_filtering, bool, "true",
  "If true, estimate a homography between matched features and reject outliers. "
  "This helps filter spurious matches that don't follow a consistent geometric "
  "transformation. Used with 'feature_matching' method." );

create_config_trait( homography_inlier_threshold, double, "5.0",
  "Maximum reprojection error (in pixels) for a match to be considered an inlier "
  "when estimating the homography. Used with 'feature_matching' method." );

create_config_trait( min_homography_inlier_ratio, double, "0.5",
  "Minimum ratio of inlier matches to total matches after homography estimation. "
  "A ratio of 0.5 means at least 50% of matches must be inliers. "
  "Used with 'feature_matching' method." );

create_config_trait( box_expansion_factor, double, "1.1",
  "Factor to expand bounding boxes when extracting features. "
  "A value of 1.1 expands boxes by 10%. Used with 'feature_matching' method." );

create_config_trait( compute_head_tail_points, bool, "false",
  "If true, compute head and tail keypoints from the two furthest apart inlier "
  "feature matches and add them to the paired detections. Only applies when "
  "matching_method is 'feature_matching'. The head/tail points are useful for "
  "downstream stereo measurement algorithms." );

create_config_trait( min_inliers_for_head_tail, int, "4",
  "Minimum number of inlier feature matches required to compute head/tail points. "
  "If fewer inliers are found, no head/tail points will be added to the detection." );

// Port traits
create_port_trait( detected_object_set1, detected_object_set,
  "Detections from camera 1 (left)" );
create_port_trait( detected_object_set2, detected_object_set,
  "Detections from camera 2 (right)" );
create_port_trait( object_track_set1, object_track_set,
  "Output tracks for camera 1" );
create_port_trait( object_track_set2, object_track_set,
  "Output tracks for camera 2" );
create_port_trait( image1, image,
  "Image from camera 1 (left). Required for feature_matching method." );
create_port_trait( image2, image,
  "Image from camera 2 (right). Required for feature_matching method." );

// =============================================================================
// Private implementation class
class pair_stereo_detections_process::priv
{
public:
  explicit priv( pair_stereo_detections_process* parent );
  ~priv();

  // Compute reprojection error for a pair of points
  double compute_reprojection_error(
    const kv::simple_camera_perspective& left_cam,
    const kv::simple_camera_perspective& right_cam,
    const kv::vector_2d& left_point,
    const kv::vector_2d& right_point ) const;

  // Find matches using IOU method
  std::vector< std::pair< int, int > > find_matches_iou(
    const std::vector< kv::detected_object_sptr >& detections1,
    const std::vector< kv::detected_object_sptr >& detections2 );

  // Find matches using calibration method
  std::vector< std::pair< int, int > > find_matches_calibration(
    const std::vector< kv::detected_object_sptr >& detections1,
    const std::vector< kv::detected_object_sptr >& detections2 );

  // Find matches using feature matching method
  std::vector< std::pair< int, int > > find_matches_feature(
    const std::vector< kv::detected_object_sptr >& detections1,
    const std::vector< kv::detected_object_sptr >& detections2,
    const kv::image_container_sptr& image1,
    const kv::image_container_sptr& image2 );

  // Compute feature match score between two detection boxes
  // Returns a score where lower is better (like a cost), or infinity if no valid match
  double compute_feature_match_score(
    const kv::detected_object_sptr& det1,
    const kv::detected_object_sptr& det2,
    const kv::image_container_sptr& image1,
    const kv::image_container_sptr& image2 );

  // Extract features within a bounding box region
  void extract_box_features(
    const kv::image_container_sptr& image,
    const kv::bounding_box_d& bbox,
    kv::feature_set_sptr& features,
    kv::descriptor_set_sptr& descriptors );

  // Filter matches by homography estimation and return inlier correspondences
  std::vector< stereo_feature_correspondence > filter_matches_by_homography(
    const kv::feature_set_sptr& features1,
    const kv::feature_set_sptr& features2,
    const kv::match_set_sptr& matches );

  // Compute feature matches and return inlier correspondences for head/tail computation
  std::vector< stereo_feature_correspondence > compute_feature_correspondences(
    const kv::detected_object_sptr& det1,
    const kv::detected_object_sptr& det2,
    const kv::image_container_sptr& image1,
    const kv::image_container_sptr& image2 );

  // Configuration values
  std::string m_matching_method;
  std::string m_calibration_file;
  double m_iou_threshold;
  double m_max_reprojection_error;
  double m_default_depth;
  bool m_require_class_match;
  bool m_use_optimal_assignment;
  bool m_output_unmatched;

  // Feature matching configuration
  int m_min_feature_match_count;
  double m_min_feature_match_ratio;
  bool m_use_homography_filtering;
  double m_homography_inlier_threshold;
  double m_min_homography_inlier_ratio;
  double m_box_expansion_factor;
  bool m_compute_head_tail_points;
  int m_min_inliers_for_head_tail;

  // Calibration data
  kv::camera_rig_stereo_sptr m_calibration;

  // Feature matching algorithms
  kv::algo::detect_features_sptr m_feature_detector;
  kv::algo::extract_descriptors_sptr m_descriptor_extractor;
  kv::algo::match_features_sptr m_feature_matcher;
  kv::algo::estimate_homography_sptr m_homography_estimator;

  // State
  kv::track_id_t m_next_track_id;

  pair_stereo_detections_process* parent;
};

// -----------------------------------------------------------------------------
pair_stereo_detections_process::priv
::priv( pair_stereo_detections_process* ptr )
  : m_matching_method( "iou" )
  , m_calibration_file( "" )
  , m_iou_threshold( 0.1 )
  , m_max_reprojection_error( 10.0 )
  , m_default_depth( 5.0 )
  , m_require_class_match( true )
  , m_use_optimal_assignment( true )
  , m_output_unmatched( true )
  , m_min_feature_match_count( 5 )
  , m_min_feature_match_ratio( 0.1 )
  , m_use_homography_filtering( true )
  , m_homography_inlier_threshold( 5.0 )
  , m_min_homography_inlier_ratio( 0.5 )
  , m_box_expansion_factor( 1.1 )
  , m_compute_head_tail_points( false )
  , m_min_inliers_for_head_tail( 4 )
  , m_next_track_id( 0 )
  , parent( ptr )
{
}

// -----------------------------------------------------------------------------
pair_stereo_detections_process::priv
::~priv()
{
}

// -----------------------------------------------------------------------------
double
pair_stereo_detections_process::priv
::compute_reprojection_error(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point ) const
{
  // Triangulate the point
  kv::vector_3d point_3d = triangulate_point(
    left_cam, right_cam, left_point, right_point );

  // Check if point is in front of both cameras (positive Z in camera coordinates)
  kv::vector_3d left_cam_point = left_cam.rotation() * ( point_3d - left_cam.center() );
  kv::vector_3d right_cam_point = right_cam.rotation() * ( point_3d - right_cam.center() );

  if( left_cam_point.z() <= 0 || right_cam_point.z() <= 0 )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Reproject to both cameras
  kv::vector_2d left_reproj = left_cam.project( point_3d );
  kv::vector_2d right_reproj = right_cam.project( point_3d );

  // Compute reprojection error
  double left_err_sq = ( left_reproj - left_point ).squaredNorm();
  double right_err_sq = ( right_reproj - right_point ).squaredNorm();

  // Return RMS error
  return std::sqrt( ( left_err_sq + right_err_sq ) / 2.0 );
}

// -----------------------------------------------------------------------------
std::vector< std::pair< int, int > >
pair_stereo_detections_process::priv
::find_matches_iou(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2 )
{
  int n1 = static_cast< int >( detections1.size() );
  int n2 = static_cast< int >( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector< std::pair< int, int > >();
  }

  // Build cost matrix (1 - IOU, so lower is better)
  std::vector< std::vector< double > > cost_matrix( n1, std::vector< double >( n2, 1e10 ) );

  for( int i = 0; i < n1; ++i )
  {
    const auto& det1 = detections1[i];
    std::string class1 = get_detection_class_label( det1 );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( m_require_class_match )
      {
        std::string class2 = get_detection_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Compute IOU
      double iou = compute_iou( det1->bounding_box(), det2->bounding_box() );

      // Check threshold
      if( iou >= m_iou_threshold )
      {
        // Use 1 - IOU as cost (lower is better)
        cost_matrix[i][j] = 1.0 - iou;
      }
    }
  }

  // Find optimal assignment
  if( m_use_optimal_assignment )
  {
    return greedy_assignment( cost_matrix, n1, n2 );
  }
  else
  {
    // Simple sequential matching
    std::vector< std::pair< int, int > > matches;
    std::set< int > used_j;

    for( int i = 0; i < n1; ++i )
    {
      int best_j = -1;
      double best_cost = 1e10;

      for( int j = 0; j < n2; ++j )
      {
        if( used_j.find( j ) != used_j.end() )
        {
          continue;
        }

        if( cost_matrix[i][j] < best_cost )
        {
          best_cost = cost_matrix[i][j];
          best_j = j;
        }
      }

      if( best_j >= 0 && best_cost < 1e9 )
      {
        matches.push_back( std::make_pair( i, best_j ) );
        used_j.insert( best_j );
      }
    }

    return matches;
  }
}

// -----------------------------------------------------------------------------
std::vector< std::pair< int, int > >
pair_stereo_detections_process::priv
::find_matches_calibration(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2 )
{
  int n1 = static_cast< int >( detections1.size() );
  int n2 = static_cast< int >( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector< std::pair< int, int > >();
  }

  if( !m_calibration || !m_calibration->left() || !m_calibration->right() )
  {
    LOG_ERROR( parent->logger(), "Calibration not loaded for calibration matching method" );
    return std::vector< std::pair< int, int > >();
  }

  // Get camera references
  const kv::simple_camera_perspective& left_cam =
    dynamic_cast< const kv::simple_camera_perspective& >( *( m_calibration->left() ) );
  const kv::simple_camera_perspective& right_cam =
    dynamic_cast< const kv::simple_camera_perspective& >( *( m_calibration->right() ) );

  // Build cost matrix using reprojection error
  std::vector< std::vector< double > > cost_matrix( n1, std::vector< double >( n2, 1e10 ) );

  for( int i = 0; i < n1; ++i )
  {
    const auto& det1 = detections1[i];
    std::string class1 = get_detection_class_label( det1 );

    // Get left detection center
    const auto& bbox1 = det1->bounding_box();
    if( !bbox1.is_valid() )
    {
      continue;
    }
    kv::vector_2d left_center = bbox1.center();

    // Project left center to right camera at default depth
    kv::vector_2d expected_right = project_left_to_right(
      left_cam, right_cam, left_center, m_default_depth );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( m_require_class_match )
      {
        std::string class2 = get_detection_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Get right detection center
      const auto& bbox2 = det2->bounding_box();
      if( !bbox2.is_valid() )
      {
        continue;
      }
      kv::vector_2d right_center = bbox2.center();

      // Quick check: is the right detection center reasonably close to expected position?
      // This is a heuristic to avoid expensive triangulation for obviously bad matches
      double expected_dist = ( right_center - expected_right ).norm();
      double search_radius = std::max( bbox1.width(), bbox1.height() ) * 2.0;
      if( expected_dist > search_radius )
      {
        continue;
      }

      // Compute reprojection error by triangulating the centers
      double reproj_error = compute_reprojection_error(
        left_cam, right_cam, left_center, right_center );

      // Check threshold
      if( reproj_error < m_max_reprojection_error )
      {
        cost_matrix[i][j] = reproj_error;
      }
    }
  }

  // Find optimal assignment
  if( m_use_optimal_assignment )
  {
    return greedy_assignment( cost_matrix, n1, n2 );
  }
  else
  {
    // Simple sequential matching
    std::vector< std::pair< int, int > > matches;
    std::set< int > used_j;

    for( int i = 0; i < n1; ++i )
    {
      int best_j = -1;
      double best_cost = 1e10;

      for( int j = 0; j < n2; ++j )
      {
        if( used_j.find( j ) != used_j.end() )
        {
          continue;
        }

        if( cost_matrix[i][j] < best_cost )
        {
          best_cost = cost_matrix[i][j];
          best_j = j;
        }
      }

      if( best_j >= 0 && best_cost < 1e9 )
      {
        matches.push_back( std::make_pair( i, best_j ) );
        used_j.insert( best_j );
      }
    }

    return matches;
  }
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process::priv
::extract_box_features(
  const kv::image_container_sptr& image,
  const kv::bounding_box_d& bbox,
  kv::feature_set_sptr& features,
  kv::descriptor_set_sptr& descriptors )
{
  features = nullptr;
  descriptors = nullptr;

  if( !image || !m_feature_detector || !m_descriptor_extractor )
  {
    return;
  }

  // Expand the bounding box
  double cx = bbox.center().x();
  double cy = bbox.center().y();
  double w = bbox.width() * m_box_expansion_factor;
  double h = bbox.height() * m_box_expansion_factor;

  // Clamp to image bounds
  double img_w = static_cast< double >( image->width() );
  double img_h = static_cast< double >( image->height() );

  double x1 = std::max( 0.0, cx - w / 2.0 );
  double y1 = std::max( 0.0, cy - h / 2.0 );
  double x2 = std::min( img_w, cx + w / 2.0 );
  double y2 = std::min( img_h, cy + h / 2.0 );

  if( x2 <= x1 || y2 <= y1 )
  {
    return;
  }

  // Crop image to the bounding box region
  kv::image_of< uint8_t > cropped_img(
    static_cast< size_t >( x2 - x1 ),
    static_cast< size_t >( y2 - y1 ),
    image->depth() );

  const kv::image& src_img = image->get_image();

  for( size_t dy = 0; dy < cropped_img.height(); ++dy )
  {
    for( size_t dx = 0; dx < cropped_img.width(); ++dx )
    {
      size_t sx = static_cast< size_t >( x1 ) + dx;
      size_t sy = static_cast< size_t >( y1 ) + dy;

      for( size_t c = 0; c < cropped_img.depth(); ++c )
      {
        cropped_img( dx, dy, c ) = src_img.at< uint8_t >( sx, sy, c );
      }
    }
  }

  auto cropped_container = std::make_shared< kv::simple_image_container >( cropped_img );

  // Detect features in the cropped region
  auto local_features = m_feature_detector->detect( cropped_container );

  if( !local_features || local_features->size() == 0 )
  {
    return;
  }

  // Offset feature locations back to full image coordinates
  std::vector< kv::feature_sptr > offset_features;
  for( const auto& feat : local_features->features() )
  {
    // Get original feature properties
    kv::vector_2d loc = feat->loc();
    loc.x() += x1;
    loc.y() += y1;

    // Create new feature with offset location
    auto new_feat = std::make_shared< kv::feature_d >(
      loc, feat->magnitude(), feat->scale(), feat->angle(), feat->color() );
    offset_features.push_back( new_feat );
  }

  features = std::make_shared< kv::simple_feature_set >( offset_features );

  // Extract descriptors
  descriptors = m_descriptor_extractor->extract( cropped_container, local_features );
}

// -----------------------------------------------------------------------------
std::vector< stereo_feature_correspondence >
pair_stereo_detections_process::priv
::filter_matches_by_homography(
  const kv::feature_set_sptr& features1,
  const kv::feature_set_sptr& features2,
  const kv::match_set_sptr& matches )
{
  std::vector< stereo_feature_correspondence > inlier_correspondences;

  if( !m_homography_estimator || !matches || matches->size() < 4 )
  {
    // Not enough matches for homography estimation, return all matches as correspondences
    if( matches )
    {
      auto feat1_vec = features1->features();
      auto feat2_vec = features2->features();
      auto match_vec = matches->matches();

      for( const auto& m : match_vec )
      {
        if( m.first < feat1_vec.size() && m.second < feat2_vec.size() )
        {
          stereo_feature_correspondence corr;
          corr.left_point = feat1_vec[m.first]->loc();
          corr.right_point = feat2_vec[m.second]->loc();
          inlier_correspondences.push_back( corr );
        }
      }
    }
    return inlier_correspondences;
  }

  // Convert matches to point vectors for homography estimation
  std::vector< kv::vector_2d > pts1, pts2;
  auto feat1_vec = features1->features();
  auto feat2_vec = features2->features();
  auto match_vec = matches->matches();

  for( const auto& m : match_vec )
  {
    if( m.first < feat1_vec.size() && m.second < feat2_vec.size() )
    {
      pts1.push_back( feat1_vec[m.first]->loc() );
      pts2.push_back( feat2_vec[m.second]->loc() );
    }
  }

  if( pts1.size() < 4 )
  {
    // Return all as correspondences if not enough for homography
    for( size_t i = 0; i < pts1.size(); ++i )
    {
      stereo_feature_correspondence corr;
      corr.left_point = pts1[i];
      corr.right_point = pts2[i];
      inlier_correspondences.push_back( corr );
    }
    return inlier_correspondences;
  }

  // Estimate homography using RANSAC
  std::vector< bool > inliers;
  try
  {
    auto homography = m_homography_estimator->estimate( pts1, pts2, inliers,
                                                         m_homography_inlier_threshold );

    if( !homography )
    {
      return inlier_correspondences; // Return empty
    }

    // Collect inlier correspondences
    for( size_t i = 0; i < inliers.size() && i < pts1.size(); ++i )
    {
      if( inliers[i] )
      {
        stereo_feature_correspondence corr;
        corr.left_point = pts1[i];
        corr.right_point = pts2[i];
        inlier_correspondences.push_back( corr );
      }
    }

    return inlier_correspondences;
  }
  catch( const std::exception& e )
  {
    LOG_DEBUG( parent->logger(), "Homography estimation failed: " << e.what() );
    return inlier_correspondences; // Return empty
  }
}

// -----------------------------------------------------------------------------
std::vector< stereo_feature_correspondence >
pair_stereo_detections_process::priv
::compute_feature_correspondences(
  const kv::detected_object_sptr& det1,
  const kv::detected_object_sptr& det2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2 )
{
  std::vector< stereo_feature_correspondence > result;

  if( !det1 || !det2 || !image1 || !image2 )
  {
    return result;
  }

  const auto& bbox1 = det1->bounding_box();
  const auto& bbox2 = det2->bounding_box();

  if( !bbox1.is_valid() || !bbox2.is_valid() )
  {
    return result;
  }

  // Extract features from both detection regions
  kv::feature_set_sptr features1, features2;
  kv::descriptor_set_sptr descriptors1, descriptors2;

  extract_box_features( image1, bbox1, features1, descriptors1 );
  extract_box_features( image2, bbox2, features2, descriptors2 );

  if( !features1 || !features2 || !descriptors1 || !descriptors2 )
  {
    return result;
  }

  if( features1->size() == 0 || features2->size() == 0 )
  {
    return result;
  }

  // Match features
  if( !m_feature_matcher )
  {
    return result;
  }

  auto matches = m_feature_matcher->match( features1, descriptors1,
                                           features2, descriptors2 );

  if( !matches || matches->size() == 0 )
  {
    return result;
  }

  // Filter by homography and get inlier correspondences
  if( m_use_homography_filtering && m_homography_estimator )
  {
    result = filter_matches_by_homography( features1, features2, matches );
  }
  else
  {
    // No filtering, convert all matches to correspondences
    auto feat1_vec = features1->features();
    auto feat2_vec = features2->features();
    auto match_vec = matches->matches();

    for( const auto& m : match_vec )
    {
      if( m.first < feat1_vec.size() && m.second < feat2_vec.size() )
      {
        stereo_feature_correspondence corr;
        corr.left_point = feat1_vec[m.first]->loc();
        corr.right_point = feat2_vec[m.second]->loc();
        result.push_back( corr );
      }
    }
  }

  return result;
}

// -----------------------------------------------------------------------------
double
pair_stereo_detections_process::priv
::compute_feature_match_score(
  const kv::detected_object_sptr& det1,
  const kv::detected_object_sptr& det2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2 )
{
  if( !det1 || !det2 || !image1 || !image2 )
  {
    return std::numeric_limits< double >::infinity();
  }

  const auto& bbox1 = det1->bounding_box();
  const auto& bbox2 = det2->bounding_box();

  if( !bbox1.is_valid() || !bbox2.is_valid() )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Extract features from both detection regions
  kv::feature_set_sptr features1, features2;
  kv::descriptor_set_sptr descriptors1, descriptors2;

  extract_box_features( image1, bbox1, features1, descriptors1 );
  extract_box_features( image2, bbox2, features2, descriptors2 );

  if( !features1 || !features2 || !descriptors1 || !descriptors2 )
  {
    return std::numeric_limits< double >::infinity();
  }

  int num_features1 = static_cast< int >( features1->size() );
  int num_features2 = static_cast< int >( features2->size() );

  if( num_features1 == 0 || num_features2 == 0 )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Match features
  if( !m_feature_matcher )
  {
    return std::numeric_limits< double >::infinity();
  }

  auto matches = m_feature_matcher->match( features1, descriptors1,
                                           features2, descriptors2 );

  if( !matches || matches->size() == 0 )
  {
    return std::numeric_limits< double >::infinity();
  }

  int match_count = static_cast< int >( matches->size() );

  // Check minimum match count
  if( match_count < m_min_feature_match_count )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Check minimum match ratio
  double match_ratio = static_cast< double >( match_count ) /
                       static_cast< double >( num_features1 );

  if( match_ratio < m_min_feature_match_ratio )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Optionally filter by homography
  int inlier_count = match_count;
  if( m_use_homography_filtering && m_homography_estimator )
  {
    auto inlier_correspondences = filter_matches_by_homography( features1, features2, matches );
    inlier_count = static_cast< int >( inlier_correspondences.size() );

    // Check minimum inlier ratio
    double inlier_ratio = static_cast< double >( inlier_count ) /
                          static_cast< double >( match_count );

    if( inlier_ratio < m_min_homography_inlier_ratio )
    {
      return std::numeric_limits< double >::infinity();
    }
  }

  // Compute score: lower is better
  // Use 1 - inlier_ratio so that higher inlier ratio gives lower cost
  double inlier_ratio = static_cast< double >( inlier_count ) /
                        static_cast< double >( std::max( num_features1, num_features2 ) );

  return 1.0 - inlier_ratio;
}

// -----------------------------------------------------------------------------
std::vector< std::pair< int, int > >
pair_stereo_detections_process::priv
::find_matches_feature(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2 )
{
  int n1 = static_cast< int >( detections1.size() );
  int n2 = static_cast< int >( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector< std::pair< int, int > >();
  }

  if( !image1 || !image2 )
  {
    LOG_ERROR( parent->logger(), "Images not provided for feature matching method" );
    return std::vector< std::pair< int, int > >();
  }

  if( !m_feature_detector || !m_descriptor_extractor || !m_feature_matcher )
  {
    LOG_ERROR( parent->logger(), "Feature algorithms not configured for feature matching method" );
    return std::vector< std::pair< int, int > >();
  }

  // Build cost matrix using feature match scores
  std::vector< std::vector< double > > cost_matrix( n1, std::vector< double >( n2, 1e10 ) );

  for( int i = 0; i < n1; ++i )
  {
    const auto& det1 = detections1[i];
    std::string class1 = get_detection_class_label( det1 );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( m_require_class_match )
      {
        std::string class2 = get_detection_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Compute feature match score
      double score = compute_feature_match_score( det1, det2, image1, image2 );

      if( std::isfinite( score ) )
      {
        cost_matrix[i][j] = score;
      }
    }
  }

  // Find optimal assignment
  if( m_use_optimal_assignment )
  {
    return greedy_assignment( cost_matrix, n1, n2 );
  }
  else
  {
    // Simple sequential matching
    std::vector< std::pair< int, int > > matches;
    std::set< int > used_j;

    for( int i = 0; i < n1; ++i )
    {
      int best_j = -1;
      double best_cost = 1e10;

      for( int j = 0; j < n2; ++j )
      {
        if( used_j.find( j ) != used_j.end() )
        {
          continue;
        }

        if( cost_matrix[i][j] < best_cost )
        {
          best_cost = cost_matrix[i][j];
          best_j = j;
        }
      }

      if( best_j >= 0 && best_cost < 1e9 )
      {
        matches.push_back( std::make_pair( i, best_j ) );
        used_j.insert( best_j );
      }
    }

    return matches;
  }
}

// =============================================================================
pair_stereo_detections_process
::pair_stereo_detections_process( kv::config_block_sptr const& config )
  : process( config ),
    d( new pair_stereo_detections_process::priv( this ) )
{
  make_ports();
  make_config();
}

pair_stereo_detections_process
::~pair_stereo_detections_process()
{
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::make_ports()
{
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // Input ports - either detections or tracks can be connected
  declare_input_port_using_trait( detected_object_set1, optional );
  declare_input_port_using_trait( detected_object_set2, optional );
  declare_input_port_using_trait( object_track_set1, optional );
  declare_input_port_using_trait( object_track_set2, optional );
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image1, optional );
  declare_input_port_using_trait( image2, optional );

  // Output ports
  declare_output_port_using_trait( object_track_set1, optional );
  declare_output_port_using_trait( object_track_set2, optional );
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::make_config()
{
  declare_config_using_trait( matching_method );
  declare_config_using_trait( calibration_file );
  declare_config_using_trait( iou_threshold );
  declare_config_using_trait( max_reprojection_error );
  declare_config_using_trait( default_depth );
  declare_config_using_trait( require_class_match );
  declare_config_using_trait( use_optimal_assignment );
  declare_config_using_trait( output_unmatched );

  // Feature matching configuration
  declare_config_using_trait( min_feature_match_count );
  declare_config_using_trait( min_feature_match_ratio );
  declare_config_using_trait( use_homography_filtering );
  declare_config_using_trait( homography_inlier_threshold );
  declare_config_using_trait( min_homography_inlier_ratio );
  declare_config_using_trait( box_expansion_factor );
  declare_config_using_trait( compute_head_tail_points );
  declare_config_using_trait( min_inliers_for_head_tail );

  // Algorithm configuration (nested algorithms for feature matching)
  kv::algo::detect_features::get_nested_algo_configuration(
    "feature_detector", get_config(), d->m_feature_detector );

  kv::algo::extract_descriptors::get_nested_algo_configuration(
    "descriptor_extractor", get_config(), d->m_descriptor_extractor );

  kv::algo::match_features::get_nested_algo_configuration(
    "feature_matcher", get_config(), d->m_feature_matcher );

  kv::algo::estimate_homography::get_nested_algo_configuration(
    "homography_estimator", get_config(), d->m_homography_estimator );
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::_configure()
{
  d->m_matching_method = config_value_using_trait( matching_method );
  d->m_calibration_file = config_value_using_trait( calibration_file );
  d->m_iou_threshold = config_value_using_trait( iou_threshold );
  d->m_max_reprojection_error = config_value_using_trait( max_reprojection_error );
  d->m_default_depth = config_value_using_trait( default_depth );
  d->m_require_class_match = config_value_using_trait( require_class_match );
  d->m_use_optimal_assignment = config_value_using_trait( use_optimal_assignment );
  d->m_output_unmatched = config_value_using_trait( output_unmatched );

  // Feature matching configuration
  d->m_min_feature_match_count = config_value_using_trait( min_feature_match_count );
  d->m_min_feature_match_ratio = config_value_using_trait( min_feature_match_ratio );
  d->m_use_homography_filtering = config_value_using_trait( use_homography_filtering );
  d->m_homography_inlier_threshold = config_value_using_trait( homography_inlier_threshold );
  d->m_min_homography_inlier_ratio = config_value_using_trait( min_homography_inlier_ratio );
  d->m_box_expansion_factor = config_value_using_trait( box_expansion_factor );
  d->m_compute_head_tail_points = config_value_using_trait( compute_head_tail_points );
  d->m_min_inliers_for_head_tail = config_value_using_trait( min_inliers_for_head_tail );

  // Validate matching method
  if( d->m_matching_method != "iou" &&
      d->m_matching_method != "calibration" &&
      d->m_matching_method != "feature_matching" )
  {
    throw std::runtime_error( "Invalid matching_method: '" + d->m_matching_method +
                              "'. Must be 'iou', 'calibration', or 'feature_matching'." );
  }

  // Load calibration if using calibration method
  if( d->m_matching_method == "calibration" )
  {
    if( d->m_calibration_file.empty() )
    {
      throw std::runtime_error( "calibration_file is required when matching_method is 'calibration'" );
    }

    d->m_calibration = kv::read_stereo_rig( d->m_calibration_file );

    if( !d->m_calibration || !d->m_calibration->left() || !d->m_calibration->right() )
    {
      throw std::runtime_error( "Failed to load calibration file: " + d->m_calibration_file );
    }

    LOG_INFO( logger(), "Loaded stereo calibration from: " << d->m_calibration_file );
  }

  // Configure feature matching algorithms if using feature_matching method or compute_head_tail_points
  bool need_feature_algorithms = ( d->m_matching_method == "feature_matching" ) ||
                                  d->m_compute_head_tail_points;

  if( need_feature_algorithms )
  {
    // Get nested algorithm configuration
    kv::config_block_sptr config = get_config();

    kv::algo::detect_features::set_nested_algo_configuration(
      "feature_detector", config, d->m_feature_detector );

    kv::algo::extract_descriptors::set_nested_algo_configuration(
      "descriptor_extractor", config, d->m_descriptor_extractor );

    kv::algo::match_features::set_nested_algo_configuration(
      "feature_matcher", config, d->m_feature_matcher );

    if( d->m_use_homography_filtering )
    {
      kv::algo::estimate_homography::set_nested_algo_configuration(
        "homography_estimator", config, d->m_homography_estimator );
    }

    // Validate that required algorithms are configured
    std::string context = d->m_matching_method == "feature_matching"
      ? "matching_method is 'feature_matching'"
      : "compute_head_tail_points is enabled";

    if( !d->m_feature_detector )
    {
      throw std::runtime_error(
        "feature_detector algorithm is required when " + context + ". "
        "Configure it using 'feature_detector:type = <algorithm_name>'" );
    }

    if( !d->m_descriptor_extractor )
    {
      throw std::runtime_error(
        "descriptor_extractor algorithm is required when " + context + ". "
        "Configure it using 'descriptor_extractor:type = <algorithm_name>'" );
    }

    if( !d->m_feature_matcher )
    {
      throw std::runtime_error(
        "feature_matcher algorithm is required when " + context + ". "
        "Configure it using 'feature_matcher:type = <algorithm_name>'" );
    }

    if( d->m_use_homography_filtering && !d->m_homography_estimator )
    {
      throw std::runtime_error(
        "homography_estimator algorithm is required when use_homography_filtering is true. "
        "Configure it using 'homography_estimator:type = <algorithm_name>'" );
    }

    LOG_INFO( logger(), "Feature matching algorithms configured" );
  }

  LOG_INFO( logger(), "Stereo detection pairing configured:" );
  LOG_INFO( logger(), "  Matching method: " << d->m_matching_method );
  if( d->m_matching_method == "iou" )
  {
    LOG_INFO( logger(), "  IOU threshold: " << d->m_iou_threshold );
  }
  else if( d->m_matching_method == "calibration" )
  {
    LOG_INFO( logger(), "  Max reprojection error: " << d->m_max_reprojection_error );
    LOG_INFO( logger(), "  Default depth: " << d->m_default_depth );
  }
  else if( d->m_matching_method == "feature_matching" )
  {
    LOG_INFO( logger(), "  Min feature match count: " << d->m_min_feature_match_count );
    LOG_INFO( logger(), "  Min feature match ratio: " << d->m_min_feature_match_ratio );
    LOG_INFO( logger(), "  Use homography filtering: " << ( d->m_use_homography_filtering ? "true" : "false" ) );
    if( d->m_use_homography_filtering )
    {
      LOG_INFO( logger(), "  Homography inlier threshold: " << d->m_homography_inlier_threshold );
      LOG_INFO( logger(), "  Min homography inlier ratio: " << d->m_min_homography_inlier_ratio );
    }
    LOG_INFO( logger(), "  Box expansion factor: " << d->m_box_expansion_factor );
  }
  LOG_INFO( logger(), "  Require class match: " << ( d->m_require_class_match ? "true" : "false" ) );
  LOG_INFO( logger(), "  Use optimal assignment: " << ( d->m_use_optimal_assignment ? "true" : "false" ) );
  LOG_INFO( logger(), "  Output unmatched: " << ( d->m_output_unmatched ? "true" : "false" ) );
  LOG_INFO( logger(), "  Compute head/tail points: " << ( d->m_compute_head_tail_points ? "true" : "false" ) );
  if( d->m_compute_head_tail_points )
  {
    LOG_INFO( logger(), "  Min inliers for head/tail: " << d->m_min_inliers_for_head_tail );
  }
}

// -----------------------------------------------------------------------------
void
pair_stereo_detections_process
::_step()
{
  // Grab timestamp (always required)
  auto timestamp = grab_from_port_using_trait( timestamp );

  // Determine input source and grab detections
  std::vector< kv::detected_object_sptr > detections1, detections2;

  bool use_detections1 = has_input_port_edge_using_trait( detected_object_set1 );
  bool use_detections2 = has_input_port_edge_using_trait( detected_object_set2 );
  bool use_tracks1 = has_input_port_edge_using_trait( object_track_set1 );
  bool use_tracks2 = has_input_port_edge_using_trait( object_track_set2 );

  // Validate input configuration
  if( !use_detections1 && !use_tracks1 )
  {
    throw std::runtime_error( "No input connected for camera 1. "
      "Connect either detected_object_set1 or object_track_set1." );
  }
  if( !use_detections2 && !use_tracks2 )
  {
    throw std::runtime_error( "No input connected for camera 2. "
      "Connect either detected_object_set2 or object_track_set2." );
  }

  // Grab camera 1 input
  if( use_detections1 )
  {
    auto detection_set1 = grab_from_port_using_trait( detected_object_set1 );
    for( const auto& det : *detection_set1 )
    {
      detections1.push_back( det );
    }
  }
  else if( use_tracks1 )
  {
    auto track_set1 = grab_from_port_using_trait( object_track_set1 );
    for( const auto& track : track_set1->tracks() )
    {
      // Get the state for the current frame
      auto it = track->find( timestamp.get_frame() );
      if( it != track->end() )
      {
        auto state = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( state && state->detection() )
        {
          detections1.push_back( state->detection() );
        }
      }
    }
  }

  // Grab camera 2 input
  if( use_detections2 )
  {
    auto detection_set2 = grab_from_port_using_trait( detected_object_set2 );
    for( const auto& det : *detection_set2 )
    {
      detections2.push_back( det );
    }
  }
  else if( use_tracks2 )
  {
    auto track_set2 = grab_from_port_using_trait( object_track_set2 );
    for( const auto& track : track_set2->tracks() )
    {
      // Get the state for the current frame
      auto it = track->find( timestamp.get_frame() );
      if( it != track->end() )
      {
        auto state = std::dynamic_pointer_cast< kv::object_track_state >( *it );
        if( state && state->detection() )
        {
          detections2.push_back( state->detection() );
        }
      }
    }
  }

  // Grab images if needed for feature matching or head/tail point computation
  kv::image_container_sptr image1, image2;
  bool need_images = ( d->m_matching_method == "feature_matching" ) ||
                     d->m_compute_head_tail_points;

  if( need_images )
  {
    bool has_image1 = has_input_port_edge_using_trait( image1 );
    bool has_image2 = has_input_port_edge_using_trait( image2 );

    if( !has_image1 || !has_image2 )
    {
      if( d->m_matching_method == "feature_matching" )
      {
        throw std::runtime_error( "Images are required for feature_matching method. "
          "Connect image1 and image2 ports." );
      }
      else if( d->m_compute_head_tail_points )
      {
        LOG_WARN( logger(), "Images not connected but compute_head_tail_points is enabled. "
          "Head/tail points will not be computed." );
        // Reset flag since we can't compute without images
        need_images = false;
      }
    }
    else
    {
      image1 = grab_from_port_using_trait( image1 );
      image2 = grab_from_port_using_trait( image2 );
    }
  }

  // Find matches using configured method
  std::vector< std::pair< int, int > > matches;
  if( d->m_matching_method == "iou" )
  {
    matches = d->find_matches_iou( detections1, detections2 );
  }
  else if( d->m_matching_method == "calibration" )
  {
    matches = d->find_matches_calibration( detections1, detections2 );
  }
  else // feature_matching
  {
    matches = d->find_matches_feature( detections1, detections2, image1, image2 );
  }

  LOG_DEBUG( logger(), "Frame " << timestamp.get_frame()
             << ": Found " << matches.size() << " matches out of "
             << detections1.size() << " left, " << detections2.size() << " right detections" );

  // Track which detections have matches
  std::vector< bool > has_match1( detections1.size(), false );
  std::vector< bool > has_match2( detections2.size(), false );

  // Create tracks for matched pairs
  std::vector< kv::track_sptr > output_trks1, output_trks2;

  for( const auto& match : matches )
  {
    int i1 = match.first;
    int i2 = match.second;

    has_match1[i1] = true;
    has_match2[i2] = true;

    // Compute head/tail keypoints if enabled and images are available
    if( d->m_compute_head_tail_points && image1 && image2 )
    {
      // Get feature correspondences with outlier rejection
      auto correspondences = d->compute_feature_correspondences(
        detections1[i1], detections2[i2], image1, image2 );

      if( static_cast< int >( correspondences.size() ) >= d->m_min_inliers_for_head_tail )
      {
        kv::vector_2d left_head, left_tail, right_head, right_tail;

        if( find_furthest_apart_points( correspondences,
                                         left_head, left_tail,
                                         right_head, right_tail ) )
        {
          // Add head/tail keypoints to both detections
          detections1[i1]->add_keypoint( "head", kv::point_2d( left_head.x(), left_head.y() ) );
          detections1[i1]->add_keypoint( "tail", kv::point_2d( left_tail.x(), left_tail.y() ) );
          detections2[i2]->add_keypoint( "head", kv::point_2d( right_head.x(), right_head.y() ) );
          detections2[i2]->add_keypoint( "tail", kv::point_2d( right_tail.x(), right_tail.y() ) );

          LOG_DEBUG( logger(), "Added head/tail keypoints for matched pair with "
                     << correspondences.size() << " inlier correspondences" );
        }
      }
      else
      {
        LOG_DEBUG( logger(), "Not enough inliers (" << correspondences.size()
                   << " < " << d->m_min_inliers_for_head_tail
                   << ") to compute head/tail keypoints" );
      }
    }

    // Create tracks with same ID for matched pairs
    auto state1 = std::make_shared< kv::object_track_state >( timestamp, detections1[i1] );
    auto state2 = std::make_shared< kv::object_track_state >( timestamp, detections2[i2] );

    auto track1 = kv::track::create();
    track1->set_id( d->m_next_track_id );
    track1->append( state1 );

    auto track2 = kv::track::create();
    track2->set_id( d->m_next_track_id );
    track2->append( state2 );

    output_trks1.push_back( track1 );
    output_trks2.push_back( track2 );

    d->m_next_track_id++;
  }

  // Add unmatched detections as separate tracks if configured
  if( d->m_output_unmatched )
  {
    for( size_t i = 0; i < detections1.size(); ++i )
    {
      if( !has_match1[i] )
      {
        auto state = std::make_shared< kv::object_track_state >( timestamp, detections1[i] );
        auto track = kv::track::create();
        track->set_id( d->m_next_track_id );
        track->append( state );
        output_trks1.push_back( track );
        d->m_next_track_id++;
      }
    }

    for( size_t i = 0; i < detections2.size(); ++i )
    {
      if( !has_match2[i] )
      {
        auto state = std::make_shared< kv::object_track_state >( timestamp, detections2[i] );
        auto track = kv::track::create();
        track->set_id( d->m_next_track_id );
        track->append( state );
        output_trks2.push_back( track );
        d->m_next_track_id++;
      }
    }
  }

  // Create output sets
  auto output_track_set1 = std::make_shared< kv::object_track_set >( output_trks1 );
  auto output_track_set2 = std::make_shared< kv::object_track_set >( output_trks2 );

  // Push outputs
  push_to_port_using_trait( object_track_set1, output_track_set1 );
  push_to_port_using_trait( object_track_set2, output_track_set2 );
}

} // end namespace core
} // end namespace viame
