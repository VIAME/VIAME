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
* \brief depth estimation utility functions.
*/

#include <arrows/core/depth_utils.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {


/// Compute the range of depths of landmarks from a camera
void
compute_depth_range_from_landmarks(std::vector<landmark_sptr> const& landmarks,
                                   camera_perspective const& cam,
                                   bounding_box<int> const& roi,
                                   double &depth_min, double &depth_max)
{
  std::vector<vector_3d> visible_landmarks = filter_visible_landmarks(cam, roi, landmarks);
  compute_depth_range(visible_landmarks, cam, depth_min, depth_max);
}

//*****************************************************************************

/// Compute the range of heights of landmarks seen by camera along a normal direction
void
compute_height_range_from_landmarks(std::vector<landmark_sptr> const& landmarks,
                                    camera const& cam, bounding_box<int> const& roi,
                                    double &height_min, double &height_max,
                                    vector_3d const& world_normal)
{
  std::vector<vector_3d> visible_landmarks = filter_visible_landmarks(cam, roi, landmarks);
  compute_offset_range(visible_landmarks, world_normal, height_min, height_max);
}

//*****************************************************************************

std::vector<vector_3d>
points_of_box(kwiver::vital::vector_3d const& minpt,
              kwiver::vital::vector_3d const& maxpt)
{
  std::vector<vector_3d> points(8);
  points[0] = vector_3d(minpt[0], minpt[1], minpt[2]);
  points[1] = vector_3d(maxpt[0], minpt[1], minpt[2]);
  points[2] = vector_3d(minpt[0], maxpt[1], minpt[2]);
  points[3] = vector_3d(maxpt[0], maxpt[1], minpt[2]);
  points[4] = vector_3d(minpt[0], minpt[1], maxpt[2]);
  points[5] = vector_3d(maxpt[0], minpt[1], maxpt[2]);
  points[6] = vector_3d(minpt[0], maxpt[1], maxpt[2]);
  points[7] = vector_3d(maxpt[0], maxpt[1], maxpt[2]);
  return points;
}

//*****************************************************************************

/// Return the axis aligned 2D box of a 3D box projected into an image
vital::bounding_box<int>
project_3d_bounds(kwiver::vital::vector_3d const& minpt,
                  kwiver::vital::vector_3d const& maxpt,
                  camera const& cam, int imgwidth, int imgheight)
{
  std::vector<vector_3d> points = points_of_box(minpt, maxpt);

  int i0, j0, i1, j1;
  vector_2d pp = cam.project(points[0]);
  i0 = i1 = static_cast<int>(pp[0]);
  j0 = j1 = static_cast<int>(pp[1]);

  for (vector_3d const& p : points)
  {
    vector_2d pp = cam.project(p);
    int ui = static_cast<int>(pp[0]), vi = static_cast<int>(pp[1]);
    i0 = std::min(i0, ui);
    j0 = std::min(j0, vi);
    i1 = std::max(i1, ui);
    j1 = std::max(j1, vi);
  }

  vital::bounding_box<int> roi(i0, j0, i1, j1);
  vital::bounding_box<int> img_bounds(0, 0, imgwidth, imgheight);

  return intersection<int>(roi, img_bounds);
}

//*****************************************************************************

///Return the depth range of a 3d region along a normal
void
height_range_from_3d_bounds(kwiver::vital::vector_3d const& minpt,
                            kwiver::vital::vector_3d const& maxpt,
                            double &height_min, double &height_max,
                            vector_3d const& world_plane_normal)
{
  std::vector<vector_3d> points = points_of_box(minpt, maxpt);

  height_min = height_max = world_plane_normal.dot(points[0]);
  for (vector_3d const& p : points)
  {
    double h = world_plane_normal.dot(p);
    height_min = std::min(height_min, h);
    height_max = std::max(height_max, h);
  }
}

//*****************************************************************************

///Return the depth range of a 3d region from a camera
void
depth_range_from_3d_bounds(kwiver::vital::vector_3d const& minpt,
                           kwiver::vital::vector_3d const& maxpt,
                           camera_perspective const& cam,
                           double &depth_min, double &depth_max)
{
  std::vector<vector_3d> points = points_of_box(minpt, maxpt);
  std::vector<double> depths;

  depth_min = std::numeric_limits<double>::infinity();
  depth_max = -std::numeric_limits<double>::infinity();
  for (vector_3d const& p : points)
  {
    double d = cam.depth(p);
    depth_min = std::min(depth_min, d);
    depth_max = std::max(depth_max, d);
  }
}

//*****************************************************************************

/// Return a subset of landmark points that project into the given region of interest
std::vector<vector_3d>
filter_visible_landmarks(camera const& cam,
                         bounding_box<int> const& roi,
                         std::vector<vital::landmark_sptr> const& landmarks)
{
  std::vector<vector_3d> visible_landmarks;

  for (unsigned int i = 0; i < landmarks.size(); i++)
  {
    vector_3d p = landmarks[i]->loc();
    vector_2d pp = cam.project(p);
    if (roi.contains(pp.cast<int>()))
    {
      visible_landmarks.push_back(p);
    }
  }

  return visible_landmarks;
}

//*****************************************************************************

/// Robustly compute the bounding planes of the landmarks in a given direction
void
compute_offset_range(std::vector<vector_3d> const& landmarks,
                     vector_3d const& normal,
                     double &min_offset, double &max_offset,
                     const double outlier_thresh,
                     const double safety_margin_factor)
{
  min_offset = std::numeric_limits<double>::infinity();
  max_offset = -std::numeric_limits<double>::infinity();

  std::vector<double> offsets;

  for (unsigned int i = 0; i < landmarks.size(); i++)
  {
    offsets.push_back(normal.dot(landmarks[i]));
  }
  std::sort(offsets.begin(), offsets.end());

  const unsigned int min_index =
    static_cast<unsigned int>((offsets.size() - 1) * outlier_thresh);
  const unsigned int max_index = offsets.size() - 1 - min_index;
  min_offset = offsets[min_index];
  max_offset = offsets[max_index];

  const double safety_margin = safety_margin_factor * (max_offset - min_offset);
  max_offset += safety_margin;
  min_offset -= safety_margin;
}

//*****************************************************************************

/// Robustly compute the bounding planes of the landmarks along a camera's view axis
void
compute_depth_range(std::vector<vector_3d> const& landmarks,
                    camera_perspective const& cam,
                    double &depth_min, double &depth_max,
                    const double outlier_thresh,
                    const double safety_margin_factor)
{
  depth_min = std::numeric_limits<double>::infinity();
  depth_max = -std::numeric_limits<double>::infinity();

  std::vector<double> depths;

  for (unsigned int i = 0; i < landmarks.size(); i++)
  {
    depths.push_back(cam.depth(landmarks[i]));
  }
  std::sort(depths.begin(), depths.end());

  const unsigned int min_index =
    static_cast<unsigned int>((depths.size() - 1) * outlier_thresh);
  const unsigned int max_index = depths.size() - 1 - min_index;
  depth_min = depths[min_index];
  depth_max = depths[max_index];

  const double safety_margin = safety_margin_factor * (depth_max - depth_min);
  depth_max += safety_margin;
  depth_min -= safety_margin;
}

} //end namespace core
} //end namespace arrows
} //end namespace kwiver
