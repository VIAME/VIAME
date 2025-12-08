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

#ifndef VIAME_OCV_STEREO_UTILS_H
#define VIAME_OCV_STEREO_UTILS_H

#include <vital/types/detected_object.h>
#include <vital/types/bounding_box.h>

#include <opencv2/core/core.hpp>

#include <vector>
#include <utility>

namespace viame
{

/// Compute oriented bounding box corner points from a detection's mask or bounding box.
///
/// If the detection has a mask, computes the minimum area rotated rectangle
/// from the mask's convex hull. Otherwise, returns the axis-aligned bounding
/// box corners.
///
/// \param det The detection to compute box points for
/// \return Vector of 4 corner points in image coordinates
std::vector<cv::Point2d>
compute_box_points( kwiver::vital::detected_object_sptr det );

/// Compute head and tail keypoints from oriented bounding box points.
///
/// Given 4 corner points of an oriented bounding box, computes the midpoints
/// of the 4 edges and returns the two with the maximum and minimum x coordinates
/// as head and tail respectively.
///
/// \param box_points Vector of 4 corner points
/// \return Pair of (head, tail) points where head has max x and tail has min x
std::pair<cv::Point2d, cv::Point2d>
center_keypoints( const std::vector<cv::Point2d>& box_points );

/// Add head and tail keypoints to a detection based on its mask or bounding box.
///
/// Convenience function that computes box points and center keypoints,
/// then adds them to the detection.
///
/// \param det The detection to add keypoints to (modified in place)
/// \return true if keypoints were added, false if detection has no valid geometry
bool
add_keypoints_from_box( kwiver::vital::detected_object_sptr det );

} // namespace viame

#endif // VIAME_OCV_STEREO_UTILS_H
