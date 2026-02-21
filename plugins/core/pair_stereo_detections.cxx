/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo detection pairing utility implementations
 */

#include "pair_stereo_detections.h"
#include "measurement_utilities.h"

#include <vital/types/image.h>
#include <vital/types/feature.h>

#include <algorithm>
#include <limits>
#include <cmath>
#include <set>

namespace viame
{

namespace core
{

// =============================================================================
// Utility function implementations
// =============================================================================

// Compute the distance from a 2D point to the epipolar line of a left image
// point. The epipolar line is determined by projecting the left point at two
// widely spaced depths and fitting a line through the resulting right image
// points. Returns perpendicular distance in pixels.
static double
epipolar_line_distance(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point )
{
  // Project at two widely separated depths to define the epipolar line.
  // The specific depth values don't matter (and are unit-independent for
  // any reasonable calibration) as long as they are far enough apart.
  kv::vector_2d p1 = project_left_to_right( left_cam, right_cam, left_point, 1.0 );
  kv::vector_2d p2 = project_left_to_right( left_cam, right_cam, left_point, 1.0e8 );

  // Line coefficients: a*x + b*y + c = 0
  double a = p1.y() - p2.y();
  double b = p2.x() - p1.x();
  double c = p1.x() * p2.y() - p2.x() * p1.y();

  double norm = std::sqrt( a * a + b * b );
  if( norm < 1e-12 )
  {
    return 1e10; // Degenerate epipolar line
  }

  return std::abs( a * right_point.x() + b * right_point.y() + c ) / norm;
}

// -----------------------------------------------------------------------------
double
compute_stereo_reprojection_error(
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const kv::vector_2d& left_point,
  const kv::vector_2d& right_point )
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
void
extract_detection_box_features(
  const kv::image_container_sptr& image,
  const kv::bounding_box_d& bbox,
  double box_expansion_factor,
  const kv::algo::detect_features_sptr& feature_detector,
  const kv::algo::extract_descriptors_sptr& descriptor_extractor,
  kv::feature_set_sptr& features,
  kv::descriptor_set_sptr& descriptors )
{
  features = nullptr;
  descriptors = nullptr;

  if( !image || !feature_detector || !descriptor_extractor )
  {
    return;
  }

  // Expand the bounding box
  double cx = bbox.center().x();
  double cy = bbox.center().y();
  double w = bbox.width() * box_expansion_factor;
  double h = bbox.height() * box_expansion_factor;

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
  auto local_features = feature_detector->detect( cropped_container );

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
  descriptors = descriptor_extractor->extract( cropped_container, local_features );
}

// -----------------------------------------------------------------------------
std::vector< stereo_feature_correspondence >
filter_matches_by_homography(
  const kv::feature_set_sptr& features1,
  const kv::feature_set_sptr& features2,
  const kv::match_set_sptr& matches,
  const kv::algo::estimate_homography_sptr& homography_estimator,
  double inlier_threshold,
  kv::logger_handle_t logger )
{
  std::vector< stereo_feature_correspondence > inlier_correspondences;

  if( !homography_estimator || !matches || matches->size() < 4 )
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
    auto homography = homography_estimator->estimate( pts1, pts2, inliers,
                                                       inlier_threshold );

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
    if( logger )
    {
      LOG_DEBUG( logger, "Homography estimation failed: " << e.what() );
    }
    return inlier_correspondences; // Return empty
  }
}

// -----------------------------------------------------------------------------
std::vector< stereo_feature_correspondence >
compute_detection_feature_correspondences(
  const kv::detected_object_sptr& det1,
  const kv::detected_object_sptr& det2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms& algorithms,
  const feature_matching_options& options,
  kv::logger_handle_t logger )
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

  extract_detection_box_features( image1, bbox1, options.box_expansion_factor,
    algorithms.feature_detector, algorithms.descriptor_extractor,
    features1, descriptors1 );

  extract_detection_box_features( image2, bbox2, options.box_expansion_factor,
    algorithms.feature_detector, algorithms.descriptor_extractor,
    features2, descriptors2 );

  if( !features1 || !features2 || !descriptors1 || !descriptors2 )
  {
    return result;
  }

  if( features1->size() == 0 || features2->size() == 0 )
  {
    return result;
  }

  // Match features
  if( !algorithms.feature_matcher )
  {
    return result;
  }

  auto matches = algorithms.feature_matcher->match( features1, descriptors1,
                                                     features2, descriptors2 );

  if( !matches || matches->size() == 0 )
  {
    return result;
  }

  // Filter by homography and get inlier correspondences
  if( options.use_homography_filtering && algorithms.homography_estimator )
  {
    result = filter_matches_by_homography( features1, features2, matches,
      algorithms.homography_estimator, options.homography_inlier_threshold, logger );
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
compute_detection_feature_match_score(
  const kv::detected_object_sptr& det1,
  const kv::detected_object_sptr& det2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms& algorithms,
  const feature_matching_options& options )
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

  extract_detection_box_features( image1, bbox1, options.box_expansion_factor,
    algorithms.feature_detector, algorithms.descriptor_extractor,
    features1, descriptors1 );

  extract_detection_box_features( image2, bbox2, options.box_expansion_factor,
    algorithms.feature_detector, algorithms.descriptor_extractor,
    features2, descriptors2 );

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
  if( !algorithms.feature_matcher )
  {
    return std::numeric_limits< double >::infinity();
  }

  auto matches = algorithms.feature_matcher->match( features1, descriptors1,
                                                     features2, descriptors2 );

  if( !matches || matches->size() == 0 )
  {
    return std::numeric_limits< double >::infinity();
  }

  int match_count = static_cast< int >( matches->size() );

  // Check minimum match count
  if( match_count < options.min_feature_match_count )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Check minimum match ratio
  double match_ratio = static_cast< double >( match_count ) /
                       static_cast< double >( num_features1 );

  if( match_ratio < options.min_feature_match_ratio )
  {
    return std::numeric_limits< double >::infinity();
  }

  // Optionally filter by homography
  int inlier_count = match_count;
  if( options.use_homography_filtering && algorithms.homography_estimator )
  {
    auto inlier_correspondences = filter_matches_by_homography(
      features1, features2, matches,
      algorithms.homography_estimator, options.homography_inlier_threshold, nullptr );
    inlier_count = static_cast< int >( inlier_correspondences.size() );

    // Check minimum inlier ratio
    double inlier_ratio = static_cast< double >( inlier_count ) /
                          static_cast< double >( match_count );

    if( inlier_ratio < options.min_homography_inlier_ratio )
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
find_stereo_matches_iou(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const iou_matching_options& options )
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
      if( options.require_class_match )
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
      if( iou >= options.iou_threshold )
      {
        // Use 1 - IOU as cost (lower is better)
        cost_matrix[i][j] = 1.0 - iou;
      }
    }
  }

  // Find optimal assignment
  if( options.use_optimal_assignment )
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
find_stereo_matches_calibration(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const calibration_matching_options& options,
  kv::logger_handle_t logger )
{
  int n1 = static_cast< int >( detections1.size() );
  int n2 = static_cast< int >( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector< std::pair< int, int > >();
  }

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
      left_cam, right_cam, left_center, options.default_depth );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( options.require_class_match )
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
      double reproj_error = compute_stereo_reprojection_error(
        left_cam, right_cam, left_center, right_center );

      // Check threshold
      if( reproj_error < options.max_reprojection_error )
      {
        cost_matrix[i][j] = reproj_error;
      }
    }
  }

  // Find optimal assignment
  if( options.use_optimal_assignment )
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
find_stereo_matches_feature(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms& algorithms,
  const feature_matching_options& options,
  kv::logger_handle_t logger )
{
  int n1 = static_cast< int >( detections1.size() );
  int n2 = static_cast< int >( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector< std::pair< int, int > >();
  }

  if( !image1 || !image2 )
  {
    if( logger )
    {
      LOG_ERROR( logger, "Images not provided for feature matching method" );
    }
    return std::vector< std::pair< int, int > >();
  }

  if( !algorithms.is_valid() )
  {
    if( logger )
    {
      LOG_ERROR( logger, "Feature algorithms not configured for feature matching method" );
    }
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
      if( options.require_class_match )
      {
        std::string class2 = get_detection_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Compute feature match score
      double score = compute_detection_feature_match_score(
        det1, det2, image1, image2, algorithms, options );

      if( std::isfinite( score ) )
      {
        cost_matrix[i][j] = score;
      }
    }
  }

  // Find optimal assignment
  if( options.use_optimal_assignment )
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
find_stereo_matches_epipolar_iou(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const epipolar_iou_matching_options& options,
  kv::logger_handle_t logger )
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

    const auto& bbox1 = det1->bounding_box();
    if( !bbox1.is_valid() )
    {
      continue;
    }

    // Project left bbox center to right image at default depth
    kv::vector_2d left_center = bbox1.center();
    kv::vector_2d projected_center = project_left_to_right(
      left_cam, right_cam, left_center, options.default_depth );

    // Build projected bbox in right image (same width/height as left, centered at projected point)
    double w = bbox1.width();
    double h = bbox1.height();
    kv::bounding_box_d projected_bbox(
      projected_center.x() - w / 2.0,
      projected_center.y() - h / 2.0,
      projected_center.x() + w / 2.0,
      projected_center.y() + h / 2.0 );

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( options.require_class_match )
      {
        std::string class2 = get_detection_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      const auto& bbox2 = det2->bounding_box();
      if( !bbox2.is_valid() )
      {
        continue;
      }

      // Compute IOU between projected left bbox and actual right bbox
      double iou = compute_iou( projected_bbox, bbox2 );

      // Check threshold
      if( iou >= options.iou_threshold )
      {
        cost_matrix[i][j] = 1.0 - iou;
      }
    }
  }

  // Find optimal assignment
  if( options.use_optimal_assignment )
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
find_stereo_matches_keypoint_projection(
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective& left_cam,
  const kv::simple_camera_perspective& right_cam,
  const keypoint_projection_matching_options& options,
  kv::logger_handle_t logger )
{
  int n1 = static_cast< int >( detections1.size() );
  int n2 = static_cast< int >( detections2.size() );

  if( n1 == 0 || n2 == 0 )
  {
    return std::vector< std::pair< int, int > >();
  }

  // Build cost matrix using average keypoint distance
  std::vector< std::vector< double > > cost_matrix( n1, std::vector< double >( n2, 1e10 ) );

  for( int i = 0; i < n1; ++i )
  {
    const auto& det1 = detections1[i];
    std::string class1 = get_detection_class_label( det1 );

    // Check that left detection has head and tail keypoints
    const auto& kp1 = det1->keypoints();
    if( kp1.find( "head" ) == kp1.end() || kp1.find( "tail" ) == kp1.end() )
    {
      continue;
    }

    kv::vector_2d left_head( kp1.at( "head" )[0], kp1.at( "head" )[1] );
    kv::vector_2d left_tail( kp1.at( "tail" )[0], kp1.at( "tail" )[1] );

    // When default_depth > 0, project keypoints at that depth and compare
    // positions directly. When default_depth <= 0, use epipolar line distance
    // which requires no depth prior (unit-independent).
    kv::vector_2d proj_head, proj_tail;
    bool use_epipolar = ( options.default_depth <= 0 );

    if( !use_epipolar )
    {
      proj_head = project_left_to_right(
        left_cam, right_cam, left_head, options.default_depth );
      proj_tail = project_left_to_right(
        left_cam, right_cam, left_tail, options.default_depth );
    }

    for( int j = 0; j < n2; ++j )
    {
      const auto& det2 = detections2[j];

      // Check class match if required
      if( options.require_class_match )
      {
        std::string class2 = get_detection_class_label( det2 );
        if( class1 != class2 )
        {
          continue;
        }
      }

      // Check that right detection has head and tail keypoints
      const auto& kp2 = det2->keypoints();
      if( kp2.find( "head" ) == kp2.end() || kp2.find( "tail" ) == kp2.end() )
      {
        continue;
      }

      kv::vector_2d right_head( kp2.at( "head" )[0], kp2.at( "head" )[1] );
      kv::vector_2d right_tail( kp2.at( "tail" )[0], kp2.at( "tail" )[1] );

      double avg_dist;
      if( use_epipolar )
      {
        // Distance from right keypoints to epipolar lines of left keypoints
        double head_dist = epipolar_line_distance(
          left_cam, right_cam, left_head, right_head );
        double tail_dist = epipolar_line_distance(
          left_cam, right_cam, left_tail, right_tail );
        avg_dist = ( head_dist + tail_dist ) / 2.0;
      }
      else
      {
        // Distance between projected and actual keypoints
        double head_dist = ( proj_head - right_head ).norm();
        double tail_dist = ( proj_tail - right_tail ).norm();
        avg_dist = ( head_dist + tail_dist ) / 2.0;
      }

      // Check threshold
      if( avg_dist <= options.max_keypoint_distance )
      {
        cost_matrix[i][j] = avg_dist;
      }
    }
  }

  // Find optimal assignment
  if( options.use_optimal_assignment )
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
// Unified detection pairing dispatch
// =============================================================================

std::vector< std::pair< int, int > >
find_stereo_detection_matches(
  const detection_pairing_params& params,
  const std::vector< kv::detected_object_sptr >& detections1,
  const std::vector< kv::detected_object_sptr >& detections2,
  const kv::simple_camera_perspective* left_cam,
  const kv::simple_camera_perspective* right_cam,
  const kv::image_container_sptr& image1,
  const kv::image_container_sptr& image2,
  const feature_matching_algorithms* feature_algos,
  const feature_matching_options* feature_opts,
  kv::logger_handle_t logger )
{
  if( params.method == "iou" )
  {
    iou_matching_options opts;
    opts.iou_threshold = params.threshold;
    opts.require_class_match = params.require_class_match;
    opts.use_optimal_assignment = params.use_optimal_assignment;
    return find_stereo_matches_iou( detections1, detections2, opts );
  }
  else if( params.method == "calibration" )
  {
    if( !left_cam || !right_cam )
    {
      if( logger )
        LOG_ERROR( logger, "Cameras required for 'calibration' pairing method" );
      return {};
    }
    calibration_matching_options opts;
    opts.max_reprojection_error = params.threshold;
    opts.default_depth = params.default_depth;
    opts.require_class_match = params.require_class_match;
    opts.use_optimal_assignment = params.use_optimal_assignment;
    return find_stereo_matches_calibration(
      detections1, detections2, *left_cam, *right_cam, opts, logger );
  }
  else if( params.method == "feature_matching" )
  {
    if( !image1 || !image2 )
    {
      if( logger )
        LOG_ERROR( logger, "Images required for 'feature_matching' pairing method" );
      return {};
    }
    if( !feature_algos || !feature_algos->is_valid() )
    {
      if( logger )
        LOG_ERROR( logger, "Feature algorithms required for 'feature_matching' pairing method" );
      return {};
    }
    feature_matching_options opts;
    if( feature_opts )
    {
      opts = *feature_opts;
    }
    opts.require_class_match = params.require_class_match;
    opts.use_optimal_assignment = params.use_optimal_assignment;
    return find_stereo_matches_feature(
      detections1, detections2, image1, image2, *feature_algos, opts, logger );
  }
  else if( params.method == "epipolar_iou" )
  {
    if( !left_cam || !right_cam )
    {
      if( logger )
        LOG_ERROR( logger, "Cameras required for 'epipolar_iou' pairing method" );
      return {};
    }
    epipolar_iou_matching_options opts;
    opts.iou_threshold = params.threshold;
    opts.default_depth = params.default_depth;
    opts.require_class_match = params.require_class_match;
    opts.use_optimal_assignment = params.use_optimal_assignment;
    return find_stereo_matches_epipolar_iou(
      detections1, detections2, *left_cam, *right_cam, opts, logger );
  }
  else if( params.method == "keypoint_projection" )
  {
    if( !left_cam || !right_cam )
    {
      if( logger )
        LOG_ERROR( logger, "Cameras required for 'keypoint_projection' pairing method" );
      return {};
    }
    keypoint_projection_matching_options opts;
    opts.max_keypoint_distance = params.threshold;
    opts.default_depth = params.default_depth;
    opts.require_class_match = params.require_class_match;
    opts.use_optimal_assignment = params.use_optimal_assignment;
    return find_stereo_matches_keypoint_projection(
      detections1, detections2, *left_cam, *right_cam, opts, logger );
  }
  else
  {
    if( logger )
      LOG_ERROR( logger, "Unknown detection pairing method: " << params.method );
    return {};
  }
}

} // end namespace core

} // end namespace viame
