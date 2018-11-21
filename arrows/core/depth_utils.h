/*ckwg +29
* Copyright 2018 by Kitware, Inc.
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
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
* \brief Header for depth estimation utility functions.
*/


#ifndef DEPTH_UTILS_H_
#define DEPTH_UTILS_H_

#include <arrows/core/kwiver_algo_core_export.h>
#include <vector>
#include <vital/types/landmark.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/bounding_box.h>


using namespace kwiver::vital;

namespace kwiver {
namespace arrows {


/// Compute the range of depths of landmarks from a camera
/// A bounding box is used to define a crop or the full image
/// \param landmarks a vector of landmarks
/// \param camera the perspective camera depth is measured from
/// \param roi region of interest in the image (or the full dimensions of the image)
/// \retval minimum depth from camera
/// \retval maximum depth from camera
/// \param the direction the depth is sliced in world coordinates
KWIVER_ALGO_CORE_EXPORT
void
compute_depth_range_from_landmarks(const std::vector<landmark_sptr> &landmarks, camera_perspective_sptr camera, 
                                   const bounding_box<double> &roi, double &depth_min, double &depth_max,
                                   const vector_3d &world_nomal = vector_3d(0.0, 0.0, 1.0));


/// Return the axis aligned 2D box of a 3D box projected into an image
/// \param minpt is one of the points defining the 3D region
/// \param maxpt is the other point defining the 3D region
/// \param cam is the perspective camera
/// \param imgwidth width of the image
/// \param imgheight height of the image
/// \param world_normal the direction the depth is sliced in world coordinates
KWIVER_ALGO_CORE_EXPORT
vital::bounding_box<double>
project_3d_bounds(kwiver::vital::vector_3d &minpt, const kwiver::vital::vector_3d &maxpt,
                  const camera_perspective_sptr &cam, int imgwidth, int imgheight,
                  const vector_3d &world_nomal = vector_3d(0.0, 0.0, 1.0));


///Return the depth range of a 3d region along a normal
/// \param minpt is one of the points defining the 3D region
/// \param maxpt is the other point defining the 3D region
/// \retval depth_min min of depth range
/// \retval depth_max max of depth range
/// \param world_normal the direction the depth is sliced in world coordinates
KWIVER_ALGO_CORE_EXPORT
void
depth_range_from_3d_bounds(kwiver::vital::vector_3d &minpt, const kwiver::vital::vector_3d &maxpt,
                           double &depth_min, double &depth_max,
                           const vector_3d &world_nomal = vector_3d(0.0, 0.0, 1.0));


/// Return a subset of landmark points that project into the given region of interest
/// \param camera is the camera used to project the points
/// \param roi region of interest within image (or entire image)
/// \param landmarks is the set of 3D landmark points to project
/// \return the subset of \p landmarks that project into the ROI
std::vector<vector_3d>
filter_visible_landmarks(camera_perspective_sptr camera,
                         const bounding_box<double> &roi,
                         const std::vector<vital::landmark_sptr> &landmarks);


/// Robustly compute the bounding planes of the landmarks in a given direction
/// \param  landmarks is the set of 3D landmark points
/// \param  normal is the normal vector of the plane
/// \retval min_offset is the minimum plane offset
/// \retval max_offset is the maximum plane offset
/// \param  outlier_thresh is the threshold for fraction of outlier offsets to
///         reject at both the top and bottom
/// \param  safety_margin_factor is the fraction of total offset range to pad
///         both top and bottom to account for insufficient landmark samples
void
compute_offset_range(const std::vector<vector_3d> &landmarks,
                     const vector_3d &normal,
                     double &min_offset, double &max_offset,
                     const double outlier_thresh = 0.1,
                     const double safety_margin_factor = 0.5);


} //end namespace arrows
} //end namespace kwiver

#endif