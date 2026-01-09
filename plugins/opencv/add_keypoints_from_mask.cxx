/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Algorithm and utility functions for adding keypoints to detections from masks
 */

#include "add_keypoints_from_mask.h"

#include <vital/types/point.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/imgproc/imgproc.hpp>

#include <cmath>
#include <limits>

namespace kv = kwiver::vital;

namespace viame
{

// =============================================================================
// Utility function implementations
// =============================================================================

// -----------------------------------------------------------------------------
std::vector< cv::Point >
get_mask_points( kv::detected_object_sptr det )
{
  std::vector< cv::Point > points;
  kv::bounding_box_d bbox = det->bounding_box();
  auto mask_container = det->mask();

  if( !mask_container )
  {
    return points;
  }

  cv::Mat mask = kwiver::arrows::ocv::image_container::vital_to_ocv(
    mask_container->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  cv::Mat mask_squeezed = mask;
  if( mask.dims > 2 && mask.size[2] == 1 )
  {
    mask_squeezed = mask.reshape( 1, mask.rows );
  }

  for( int y = 0; y < mask_squeezed.rows; ++y )
  {
    for( int x = 0; x < mask_squeezed.cols; ++x )
    {
      if( mask_squeezed.at< uchar >( y, x ) > 0 )
      {
        // Convert to image coordinates
        points.push_back( cv::Point(
          static_cast< int >( x + bbox.min_x() ),
          static_cast< int >( y + bbox.min_y() ) ) );
      }
    }
  }

  return points;
}

// -----------------------------------------------------------------------------
std::vector< cv::Point2d >
compute_box_points( kv::detected_object_sptr det )
{
  std::vector< cv::Point2d > box_points;
  kv::bounding_box_d bbox = det->bounding_box();
  auto mask_container = det->mask();

  if( !mask_container )
  {
    // Use axis-aligned bbox corners
    box_points.push_back( cv::Point2d( bbox.min_x(), bbox.min_y() ) );
    box_points.push_back( cv::Point2d( bbox.max_x(), bbox.min_y() ) );
    box_points.push_back( cv::Point2d( bbox.max_x(), bbox.max_y() ) );
    box_points.push_back( cv::Point2d( bbox.min_x(), bbox.max_y() ) );
  }
  else
  {
    // Convert mask to OpenCV format
    cv::Mat mask = kwiver::arrows::ocv::image_container::vital_to_ocv(
      mask_container->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

    // Find convex hull from mask
    std::vector< cv::Point > points;
    cv::Mat mask_squeezed = mask;
    if( mask.dims > 2 && mask.size[2] == 1 )
    {
      mask_squeezed = mask.reshape( 1, mask.rows );
    }

    for( int y = 0; y < mask_squeezed.rows; ++y )
    {
      for( int x = 0; x < mask_squeezed.cols; ++x )
      {
        if( mask_squeezed.at< uchar >( y, x ) > 0 )
        {
          points.push_back( cv::Point( x, y ) );
        }
      }
    }

    if( points.empty() )
    {
      // Fallback to bbox
      box_points.push_back( cv::Point2d( bbox.min_x(), bbox.min_y() ) );
      box_points.push_back( cv::Point2d( bbox.max_x(), bbox.min_y() ) );
      box_points.push_back( cv::Point2d( bbox.max_x(), bbox.max_y() ) );
      box_points.push_back( cv::Point2d( bbox.min_x(), bbox.max_y() ) );
      return box_points;
    }

    std::vector< cv::Point > hull;
    cv::convexHull( points, hull );

    // Find minimum area rotated rectangle
    cv::RotatedRect rotated_rect = cv::minAreaRect( hull );
    cv::Point2f vertices[4];
    rotated_rect.points( vertices );

    // Transform from mask coordinates to image coordinates
    for( int i = 0; i < 4; ++i )
    {
      double x = vertices[i].x + bbox.min_x();
      double y = vertices[i].y + bbox.min_y();
      box_points.push_back( cv::Point2d( x, y ) );
    }
  }

  return box_points;
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
center_keypoints( const std::vector< cv::Point2d >& box_points )
{
  if( box_points.size() < 4 )
  {
    return std::make_pair( cv::Point2d( 0, 0 ), cv::Point2d( 0, 0 ) );
  }

  // Compute edge midpoints
  std::vector< cv::Point2d > centers;
  centers.push_back( ( box_points[0] + box_points[1] ) * 0.5 );
  centers.push_back( ( box_points[1] + box_points[2] ) * 0.5 );
  centers.push_back( ( box_points[2] + box_points[3] ) * 0.5 );
  centers.push_back( ( box_points[3] + box_points[0] ) * 0.5 );

  // Find min/max x points (head/tail)
  cv::Point2d min_pt = centers[0];
  cv::Point2d max_pt = centers[0];
  double min_x = centers[0].x;
  double max_x = centers[0].x;

  for( const auto& pt : centers )
  {
    if( pt.x < min_x )
    {
      min_x = pt.x;
      min_pt = pt;
    }
    if( pt.x > max_x )
    {
      max_x = pt.x;
      max_pt = pt;
    }
  }

  return std::make_pair( max_pt, min_pt );  // head (max_x), tail (min_x)
}

// -----------------------------------------------------------------------------
bool
add_keypoints_from_box( kv::detected_object_sptr det )
{
  if( !det )
  {
    return false;
  }

  auto box_pts = compute_box_points( det );
  if( box_pts.size() < 4 )
  {
    return false;
  }

  auto kp = center_keypoints( box_pts );

  det->add_keypoint( "head", kv::point_2d( kp.first.x, kp.first.y ) );
  det->add_keypoint( "tail", kv::point_2d( kp.second.x, kp.second.y ) );

  return true;
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_oriented_bbox( kv::detected_object_sptr det )
{
  auto box_pts = compute_box_points( det );
  if( box_pts.size() < 4 )
  {
    return std::make_pair( cv::Point2d( 0, 0 ), cv::Point2d( 0, 0 ) );
  }
  return center_keypoints( box_pts );
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_pca( kv::detected_object_sptr det )
{
  auto points = get_mask_points( det );

  if( points.size() < 2 )
  {
    // Fallback to oriented bbox
    return compute_keypoints_oriented_bbox( det );
  }

  // Convert to Mat for PCA
  cv::Mat data( static_cast< int >( points.size() ), 2, CV_64F );
  for( size_t i = 0; i < points.size(); ++i )
  {
    data.at< double >( static_cast< int >( i ), 0 ) = points[i].x;
    data.at< double >( static_cast< int >( i ), 1 ) = points[i].y;
  }

  // Perform PCA
  cv::PCA pca( data, cv::Mat(), cv::PCA::DATA_AS_ROW );

  // Get the center (mean)
  cv::Point2d center( pca.mean.at< double >( 0, 0 ), pca.mean.at< double >( 0, 1 ) );

  // Get the principal eigenvector (first row of eigenvectors)
  cv::Point2d eigenvector( pca.eigenvectors.at< double >( 0, 0 ),
                           pca.eigenvectors.at< double >( 0, 1 ) );

  // Project all points onto principal axis and find extremes
  double min_proj = std::numeric_limits< double >::max();
  double max_proj = std::numeric_limits< double >::lowest();
  cv::Point2d min_pt, max_pt;

  for( const auto& pt : points )
  {
    cv::Point2d p( pt.x, pt.y );
    cv::Point2d diff = p - center;
    double proj = diff.x * eigenvector.x + diff.y * eigenvector.y;

    if( proj < min_proj )
    {
      min_proj = proj;
      min_pt = p;
    }
    if( proj > max_proj )
    {
      max_proj = proj;
      max_pt = p;
    }
  }

  // Return head (max x) first, tail (min x) second
  if( max_pt.x >= min_pt.x )
  {
    return std::make_pair( max_pt, min_pt );
  }
  else
  {
    return std::make_pair( min_pt, max_pt );
  }
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_farthest( kv::detected_object_sptr det )
{
  auto points = get_mask_points( det );

  if( points.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  // Get convex hull to reduce search space
  std::vector< cv::Point > hull;
  cv::convexHull( points, hull );

  if( hull.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  // Find the two farthest points on the hull (rotating calipers would be optimal,
  // but for typical polygon sizes, brute force is acceptable)
  double max_dist_sq = 0;
  cv::Point2d pt1, pt2;

  for( size_t i = 0; i < hull.size(); ++i )
  {
    for( size_t j = i + 1; j < hull.size(); ++j )
    {
      double dx = hull[j].x - hull[i].x;
      double dy = hull[j].y - hull[i].y;
      double dist_sq = dx * dx + dy * dy;

      if( dist_sq > max_dist_sq )
      {
        max_dist_sq = dist_sq;
        pt1 = cv::Point2d( hull[i].x, hull[i].y );
        pt2 = cv::Point2d( hull[j].x, hull[j].y );
      }
    }
  }

  // Return head (max x) first, tail (min x) second
  if( pt1.x >= pt2.x )
  {
    return std::make_pair( pt1, pt2 );
  }
  else
  {
    return std::make_pair( pt2, pt1 );
  }
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_hull_extremes( kv::detected_object_sptr det )
{
  auto points = get_mask_points( det );

  if( points.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  // Get convex hull
  std::vector< cv::Point > hull;
  cv::convexHull( points, hull );

  if( hull.size() < 2 )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  // Find minimum area rotated rectangle of the hull
  cv::RotatedRect rotated_rect = cv::minAreaRect( hull );
  cv::Point2f vertices[4];
  rotated_rect.points( vertices );

  // Determine which edges are the long edges
  double edge01 = cv::norm( cv::Point2f( vertices[1].x - vertices[0].x,
                                          vertices[1].y - vertices[0].y ) );
  double edge12 = cv::norm( cv::Point2f( vertices[2].x - vertices[1].x,
                                          vertices[2].y - vertices[1].y ) );

  cv::Point2d center1, center2;

  if( edge01 > edge12 )
  {
    // Long edges are 0-1 and 2-3, short edges are 1-2 and 3-0
    center1 = cv::Point2d( ( vertices[1].x + vertices[2].x ) * 0.5,
                           ( vertices[1].y + vertices[2].y ) * 0.5 );
    center2 = cv::Point2d( ( vertices[3].x + vertices[0].x ) * 0.5,
                           ( vertices[3].y + vertices[0].y ) * 0.5 );
  }
  else
  {
    // Long edges are 1-2 and 3-0, short edges are 0-1 and 2-3
    center1 = cv::Point2d( ( vertices[0].x + vertices[1].x ) * 0.5,
                           ( vertices[0].y + vertices[1].y ) * 0.5 );
    center2 = cv::Point2d( ( vertices[2].x + vertices[3].x ) * 0.5,
                           ( vertices[2].y + vertices[3].y ) * 0.5 );
  }

  // Return head (max x) first, tail (min x) second
  if( center1.x >= center2.x )
  {
    return std::make_pair( center1, center2 );
  }
  else
  {
    return std::make_pair( center2, center1 );
  }
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints_skeleton( kv::detected_object_sptr det )
{
  kv::bounding_box_d bbox = det->bounding_box();
  auto mask_container = det->mask();

  if( !mask_container )
  {
    return compute_keypoints_oriented_bbox( det );
  }

  cv::Mat mask = kwiver::arrows::ocv::image_container::vital_to_ocv(
    mask_container->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Ensure binary mask
  cv::Mat binary;
  if( mask.channels() > 1 )
  {
    cv::cvtColor( mask, binary, cv::COLOR_BGR2GRAY );
  }
  else
  {
    binary = mask.clone();
  }
  cv::threshold( binary, binary, 0, 255, cv::THRESH_BINARY );

  // Compute skeleton using morphological thinning
  cv::Mat skeleton = cv::Mat::zeros( binary.size(), CV_8UC1 );
  cv::Mat temp, eroded;
  cv::Mat element = cv::getStructuringElement( cv::MORPH_CROSS, cv::Size( 3, 3 ) );

  bool done = false;
  int max_iterations = 1000;
  int iter = 0;

  cv::Mat img = binary.clone();

  while( !done && iter < max_iterations )
  {
    cv::erode( img, eroded, element );
    cv::dilate( eroded, temp, element );
    cv::subtract( img, temp, temp );
    cv::bitwise_or( skeleton, temp, skeleton );
    eroded.copyTo( img );

    done = ( cv::countNonZero( img ) == 0 );
    iter++;
  }

  // Find skeleton endpoints (pixels with only one neighbor)
  std::vector< cv::Point > endpoints;
  std::vector< cv::Point > skeleton_points;

  for( int y = 1; y < skeleton.rows - 1; ++y )
  {
    for( int x = 1; x < skeleton.cols - 1; ++x )
    {
      if( skeleton.at< uchar >( y, x ) > 0 )
      {
        skeleton_points.push_back( cv::Point( x, y ) );

        // Count 8-connected neighbors
        int neighbors = 0;
        for( int dy = -1; dy <= 1; ++dy )
        {
          for( int dx = -1; dx <= 1; ++dx )
          {
            if( dx == 0 && dy == 0 ) continue;
            if( skeleton.at< uchar >( y + dy, x + dx ) > 0 )
            {
              neighbors++;
            }
          }
        }

        if( neighbors == 1 )
        {
          endpoints.push_back( cv::Point( x, y ) );
        }
      }
    }
  }

  cv::Point2d pt1, pt2;

  if( endpoints.size() >= 2 )
  {
    // Find the two endpoints that are farthest apart
    double max_dist_sq = 0;
    size_t idx1 = 0, idx2 = 1;

    for( size_t i = 0; i < endpoints.size(); ++i )
    {
      for( size_t j = i + 1; j < endpoints.size(); ++j )
      {
        double dx = endpoints[j].x - endpoints[i].x;
        double dy = endpoints[j].y - endpoints[i].y;
        double dist_sq = dx * dx + dy * dy;

        if( dist_sq > max_dist_sq )
        {
          max_dist_sq = dist_sq;
          idx1 = i;
          idx2 = j;
        }
      }
    }

    pt1 = cv::Point2d( endpoints[idx1].x + bbox.min_x(),
                       endpoints[idx1].y + bbox.min_y() );
    pt2 = cv::Point2d( endpoints[idx2].x + bbox.min_x(),
                       endpoints[idx2].y + bbox.min_y() );
  }
  else if( skeleton_points.size() >= 2 )
  {
    // No clear endpoints, find farthest skeleton points
    double max_dist_sq = 0;

    for( size_t i = 0; i < skeleton_points.size(); ++i )
    {
      for( size_t j = i + 1; j < skeleton_points.size(); ++j )
      {
        double dx = skeleton_points[j].x - skeleton_points[i].x;
        double dy = skeleton_points[j].y - skeleton_points[i].y;
        double dist_sq = dx * dx + dy * dy;

        if( dist_sq > max_dist_sq )
        {
          max_dist_sq = dist_sq;
          pt1 = cv::Point2d( skeleton_points[i].x + bbox.min_x(),
                             skeleton_points[i].y + bbox.min_y() );
          pt2 = cv::Point2d( skeleton_points[j].x + bbox.min_x(),
                             skeleton_points[j].y + bbox.min_y() );
        }
      }
    }
  }
  else
  {
    // Fallback
    return compute_keypoints_oriented_bbox( det );
  }

  // Return head (max x) first, tail (min x) second
  if( pt1.x >= pt2.x )
  {
    return std::make_pair( pt1, pt2 );
  }
  else
  {
    return std::make_pair( pt2, pt1 );
  }
}

// -----------------------------------------------------------------------------
std::pair< cv::Point2d, cv::Point2d >
compute_keypoints( kv::detected_object_sptr det, const std::string& method )
{
  if( method == "pca" )
  {
    return compute_keypoints_pca( det );
  }
  else if( method == "farthest" )
  {
    return compute_keypoints_farthest( det );
  }
  else if( method == "hull_extremes" )
  {
    return compute_keypoints_hull_extremes( det );
  }
  else if( method == "skeleton" )
  {
    return compute_keypoints_skeleton( det );
  }
  else // oriented_bbox (default)
  {
    return compute_keypoints_oriented_bbox( det );
  }
}

// -----------------------------------------------------------------------------
bool
is_valid_keypoint_method( const std::string& method )
{
  return method == "oriented_bbox" ||
         method == "pca" ||
         method == "farthest" ||
         method == "hull_extremes" ||
         method == "skeleton";
}

// -----------------------------------------------------------------------------
std::string
keypoint_method_description()
{
  return "Method for computing keypoints from polygon/mask. Options:\n"
         "  oriented_bbox - Use midpoints of short edges of oriented bounding box (default)\n"
         "  pca - Use Principal Component Analysis to find major axis extremes\n"
         "  farthest - Find the two farthest points on the polygon\n"
         "  hull_extremes - Use midpoints of short edges of convex hull's oriented bbox\n"
         "  skeleton - Use endpoints of the medial axis/skeleton";
}

// =============================================================================
// Algorithm class implementation
// =============================================================================

// Private implementation class
class add_keypoints_from_mask::priv
{
public:
  priv()
    : method( "oriented_bbox" )
  {
  }

  ~priv()
  {
  }

  // Configuration
  std::string method;
};

// =============================================================================
add_keypoints_from_mask
::add_keypoints_from_mask()
  : d( new priv() )
{
}

add_keypoints_from_mask
::~add_keypoints_from_mask()
{
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
add_keypoints_from_mask
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::refine_detections::get_configuration();

  config->set_value( "method", d->method, keypoint_method_description() );

  return config;
}

// -----------------------------------------------------------------------------
void
add_keypoints_from_mask
::set_configuration( kv::config_block_sptr config )
{
  d->method = config->get_value< std::string >( "method", d->method );
}

// -----------------------------------------------------------------------------
bool
add_keypoints_from_mask
::check_configuration( kv::config_block_sptr config ) const
{
  std::string method = config->get_value< std::string >( "method", "oriented_bbox" );

  if( !is_valid_keypoint_method( method ) )
  {
    LOG_ERROR( logger(), "Invalid method: " << method );
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
add_keypoints_from_mask
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  auto output = std::make_shared< kv::detected_object_set >();

  for( auto det : *detections )
  {
    if( det->mask() )
    {
      auto keypoints = compute_keypoints( det, d->method );

      det->add_keypoint( "head", kv::point_2d( keypoints.first.x, keypoints.first.y ) );
      det->add_keypoint( "tail", kv::point_2d( keypoints.second.x, keypoints.second.y ) );
    }

    output->add( det );
  }

  return output;
}

} // end namespace viame
