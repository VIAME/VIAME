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
namespace core {

/// Compute the range of depths of landmarks from a camera
/**
* A bounding box is used to define a crop or the full image
* \param landmarks a vector of landmarks
* \param camera the perspective camera depth is measured from
* \param roi region of interest in the image (or the full dimensions of the image)
* \param minimum depth from camera
* \param maximum depth from camera
*/
KWIVER_ALGO_CORE_EXPORT
void
compute_depth_range_from_landmarks(std::vector<landmark_sptr> const& landmarks,
                                   camera_perspective const& cam,
                                   bounding_box<int> const& roi,
                                   double &depth_min, double &depth_max);

/// Compute the range of heights of landmarks seen by camera along a normal direction
/**
* A bounding box is used to define a crop or the full image
* \param landmarks a vector of landmarks
* \param cam the camera used to find visible landmarks
* \param roi region of interest in the image (or the full dimensions of the image)
* \param minimum height along normal
* \param maximum height along normal
* \param the direction the depth is sliced in world coordinates
*/
KWIVER_ALGO_CORE_EXPORT
void
compute_height_range_from_landmarks(std::vector<landmark_sptr> const& landmarks,
                                    camera const& cam,  bounding_box<int> const& roi,
                                    double &height_min, double &height_max,
                                    vector_3d const& world_normal = vector_3d(0.0, 0.0, 1.0));

/// Compute a robust 3D bounding box for a set of landmarks
/**
* \param landmarks a vector of landmarks
* \param bounds is the output 3D bounds
* \param percentile outlier percentile for x and y dimensions
* \param zmax_percentile outlier percentile for z dimension
* \param margin widening factor applied to resulting bounds
*/
KWIVER_ALGO_CORE_EXPORT
bool
compute_robust_ROI(std::vector<landmark_sptr> const& landmarks,
                   double bounds[6],
                   double percentile = 0.1,
                   double zmax_percentile = 0.01,
                   double margin = 0.5);

/// Return the axis aligned 2D box of a 3D box projected into an image
/**
* \param minpt is one of the points defining the 3D region
* \param maxpt is the other point defining the 3D region
* \param cam is the camera
* \param imgwidth width of the image
* \param imgheight height of the image
* \param world_normal the direction the depth is sliced in world coordinates
* \returns bounding box in 2d
*/
KWIVER_ALGO_CORE_EXPORT
vital::bounding_box<int>
project_3d_bounds(kwiver::vital::vector_3d const& minpt,
                  kwiver::vital::vector_3d const& maxpt,
                  camera const& cam, int imgwidth, int imgheight);


///Return the height range of a 3d region along a normal
/**
* \param minpt is one of the points defining the 3D region
* \param maxpt is the other point defining the 3D region
* \param height_min min of depth range
* \param depth_max max of depth range
* \param world_normal the direction the depth is sliced in world coordinates
*/
KWIVER_ALGO_CORE_EXPORT
void
height_range_from_3d_bounds(kwiver::vital::vector_3d const& minpt,
                            kwiver::vital::vector_3d const& maxpt,
                            double &height_min, double &height_max,
                            vector_3d const& world_normal = vector_3d(0.0, 0.0, 1.0));

///Return the depth range of a 3d region from a camera
/**
* \param minpt is one of the points defining the 3D region
* \param maxpt is the other point defining the 3D region
* \param cam is the camera the depth is measured from
* \param depth_min min of depth range
* \param depth_max max of depth range
*/
KWIVER_ALGO_CORE_EXPORT
void
depth_range_from_3d_bounds(kwiver::vital::vector_3d const& minpt,
                           kwiver::vital::vector_3d const& maxpt,
                           camera_perspective const& cam,
                           double &depth_min, double &depth_max);


/// Return a subset of landmark points that project into the given region of interest
/**
* \param cam is the camera used to project the points
* \param roi region of interest within image (or entire image)
* \param landmarks is the set of 3D landmark points to project
* \returns the subset of landmarks that project into the ROI
*/
std::vector<vector_3d>
filter_visible_landmarks(camera const& cam,
                         bounding_box<int> const& roi,
                         std::vector<vital::landmark_sptr> const& landmarks);


/// Robustly compute the bounding planes of the landmarks in a given direction
/**
* \param  landmarks is the set of 3D landmark points
* \param  normal is the normal vector of the plane
* \param  min_offset is the minimum plane offset
* \param  max_offset is the maximum plane offset
* \param  outlier_thresh is the threshold for fraction of outlier offsets to
*         reject at both the top and bottom
* \param  safety_margin_factor is the fraction of total offset range to pad
*         both top and bottom to account for insufficient landmark samples
*/
void
compute_offset_range(std::vector<vector_3d> const& landmarks,
                     vector_3d const& normal,
                     double &min_offset, double &max_offset,
                     const double outlier_thresh = 0.1,
                     const double safety_margin_factor = 0.5);

/// Robustly compute the bounding planes of the landmarks along a camera's view axis
/**
* \param  landmarks is the set of 3D landmark points
* \param  cam is the perspective camera to compute the range from
* \param  depth_min is the minimum of the depth range
* \param  depth_max is the maximum of the depth range
* \param  outlier_thresh is the threshold for fraction of outlier offsets to
*         reject at both the top and bottom
* \param  safety_margin_factor is the fraction of total offset range to pad
*         both top and bottom to account for insufficient landmark samples
*/
void
compute_depth_range(std::vector<vector_3d> const& landmarks,
                    camera_perspective const& cam,
                    double &depth_min, double &depth_max,
                    const double outlier_thresh = 0.1,
                    const double safety_margin_factor = 0.5);

/// Estimate the pixel to world scale over a set of cameras
/**
* \param  minpt Minimum point of 3d box
  \param  maxpt Maximum point of 3d box
* \param  cameras Vector of perspective cameras to compute the average scale from
*/
KWIVER_ALGO_CORE_EXPORT
double
compute_pixel_to_world_scale(kwiver::vital::vector_3d const& minpt,
                             kwiver::vital::vector_3d const& maxpt,
                             std::vector<camera_perspective_sptr> const& cameras);
} //end namespace core
} //end namespace arrows
} //end namespace kwiver

#endif