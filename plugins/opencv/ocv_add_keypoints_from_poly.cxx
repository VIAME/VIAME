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
 * \brief Algorithm to add keypoints to detections from oriented bounding box
 */

#include "ocv_add_keypoints_from_poly.h"
#include "ocv_keypoints_from_mask.h"

#include <vital/types/point.h>

namespace kv = kwiver::vital;

namespace viame
{

// =============================================================================
// Private implementation class
class ocv_add_keypoints_from_poly::priv
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
ocv_add_keypoints_from_poly
::ocv_add_keypoints_from_poly()
  : d( new priv() )
{
}

ocv_add_keypoints_from_poly
::~ocv_add_keypoints_from_poly()
{
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
ocv_add_keypoints_from_poly
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::refine_detections::get_configuration();

  config->set_value( "method", d->method,
    "Method for computing keypoints from polygon/mask. Options:\n"
    "  oriented_bbox - Use midpoints of short edges of oriented bounding box (default)\n"
    "  pca - Use Principal Component Analysis to find major axis extremes\n"
    "  farthest - Find the two farthest points on the polygon\n"
    "  hull_extremes - Use midpoints of short edges of convex hull's oriented bbox\n"
    "  skeleton - Use endpoints of the medial axis/skeleton" );

  return config;
}

// -----------------------------------------------------------------------------
void
ocv_add_keypoints_from_poly
::set_configuration( kv::config_block_sptr config )
{
  d->method = config->get_value<std::string>( "method", d->method );
}

// -----------------------------------------------------------------------------
bool
ocv_add_keypoints_from_poly
::check_configuration( kv::config_block_sptr config ) const
{
  std::string method = config->get_value<std::string>( "method", "oriented_bbox" );

  if( method != "oriented_bbox" &&
      method != "pca" &&
      method != "farthest" &&
      method != "hull_extremes" &&
      method != "skeleton" )
  {
    LOG_ERROR( logger(), "Invalid method: " << method <<
      ". Must be one of: oriented_bbox, pca, farthest, hull_extremes, skeleton" );
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
ocv_add_keypoints_from_poly
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  auto output = std::make_shared<kv::detected_object_set>();

  for( auto det : *detections )
  {
    if( det->mask() )
    {
      std::pair<cv::Point2d, cv::Point2d> keypoints;

      if( d->method == "pca" )
      {
        keypoints = compute_keypoints_pca( det );
      }
      else if( d->method == "farthest" )
      {
        keypoints = compute_keypoints_farthest( det );
      }
      else if( d->method == "hull_extremes" )
      {
        keypoints = compute_keypoints_hull_extremes( det );
      }
      else if( d->method == "skeleton" )
      {
        keypoints = compute_keypoints_skeleton( det );
      }
      else // oriented_bbox (default)
      {
        keypoints = compute_keypoints_oriented_bbox( det );
      }

      det->add_keypoint( "head", kv::point_2d( keypoints.first.x, keypoints.first.y ) );
      det->add_keypoint( "tail", kv::point_2d( keypoints.second.x, keypoints.second.y ) );
    }

    output->add( det );
  }

  return output;
}

} // end namespace viame
