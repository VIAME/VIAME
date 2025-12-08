/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
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
 * \brief Utility functions for stereo processing and keypoint computation
 */

#include "ocv_stereo_utils.h"

#include <vital/types/point.h>

#include <arrows/ocv/image_container.h>

#include <opencv2/imgproc/imgproc.hpp>

namespace viame
{

// -----------------------------------------------------------------------------
std::vector<cv::Point2d>
compute_box_points( kwiver::vital::detected_object_sptr det )
{
  namespace kv = kwiver::vital;

  std::vector<cv::Point2d> box_points;
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
    std::vector<cv::Point> points;
    cv::Mat mask_squeezed = mask;
    if( mask.dims > 2 && mask.size[2] == 1 )
    {
      mask_squeezed = mask.reshape( 1, mask.rows );
    }

    for( int y = 0; y < mask_squeezed.rows; ++y )
    {
      for( int x = 0; x < mask_squeezed.cols; ++x )
      {
        if( mask_squeezed.at<uchar>( y, x ) > 0 )
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

    std::vector<cv::Point> hull;
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
std::pair<cv::Point2d, cv::Point2d>
center_keypoints( const std::vector<cv::Point2d>& box_points )
{
  if( box_points.size() < 4 )
  {
    return std::make_pair( cv::Point2d(0, 0), cv::Point2d(0, 0) );
  }

  // Compute edge midpoints
  std::vector<cv::Point2d> centers;
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
add_keypoints_from_box( kwiver::vital::detected_object_sptr det )
{
  namespace kv = kwiver::vital;

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

} // namespace viame
