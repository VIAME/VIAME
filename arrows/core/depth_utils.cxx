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
compute_depth_range_from_landmarks(const std::vector<landmark_sptr> &landmarks, const camera_perspective_sptr &cam,
                                   const bounding_box<double> &roi, double &depth_min, double &depth_max)
{
  std::vector<vector_3d> visible_landmarks = filter_visible_landmarks(cam, roi, landmarks);
  compute_depth_range(visible_landmarks, cam, depth_min, depth_max);
}

//*****************************************************************************

/// Compute the range of heights of landmarks seen by camera along a normal direction
void
compute_height_range_from_landmarks(const std::vector<landmark_sptr> &landmarks, const camera_sptr &cam,
                                   const bounding_box<double> &roi, double &height_min, double &height_max,
                                   const vector_3d &world_plane_normal)
{
  std::vector<vector_3d> visible_landmarks = filter_visible_landmarks(cam, roi, landmarks);
  compute_offset_range(visible_landmarks, world_plane_normal, height_min, height_max);
}

//*****************************************************************************

std::vector<vector_3d>
points_of_box(kwiver::vital::vector_3d &minpt, const kwiver::vital::vector_3d &maxpt)
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
vital::bounding_box<double>
project_3d_bounds(kwiver::vital::vector_3d &minpt, const kwiver::vital::vector_3d &maxpt,
                  const camera_sptr &cam, int imgwidth, int imgheight)
{
  std::vector<vector_3d> points = points_of_box(minpt, maxpt);

  int i0, j0, i1, j1;
  vector_2d pp = cam->project(points[0]);
  i0 = i1 = (int)pp[0];
  j0 = j1 = (int)pp[1];


  for (int i = 1; i < 8; i++)
  {
    vector_2d pp = cam->project(points[i]);
    int ui = (int)pp[0], vi = (int)pp[1];
    if (ui < i0)
      i0 = ui;
    if (vi < j0)
      j0 = vi;
    if (ui > i1)
      i1 = ui;
    if (vi > j1)
      j1 = vi;
  }

  vital::bounding_box<double> roi(i0, j0, i1, j1);
  vital::bounding_box<double> img_bounds(0, imgwidth, 0, imgheight);

  return intersection<double>(roi, img_bounds);
}

//*****************************************************************************

///Return the depth range of a 3d region along a normal
void
height_range_from_3d_bounds(kwiver::vital::vector_3d &minpt, const kwiver::vital::vector_3d &maxpt,
                            double &height_min, double &height_max,
                            const vector_3d &world_plane_normal)
{
  std::vector<vector_3d> points = points_of_box(minpt, maxpt);

  height_min = height_max = world_plane_normal.dot(points[0]);
  for (int i = 1; i < 8; i++)
  {
    double d = world_plane_normal.dot(points[i]);
    if (height_min > d)
      height_min = d;
    if (height_max < d)
      height_max = d;
  }
}

//*****************************************************************************

///Return the depth range of a 3d region from a camera
void
depth_range_from_3d_bounds(kwiver::vital::vector_3d &minpt, const kwiver::vital::vector_3d &maxpt,
                           const camera_perspective_sptr &cam,
                           double &depth_min, double &depth_max)
{
  std::vector<vector_3d> points = points_of_box(minpt, maxpt);
  std::vector<double> depths;

  depth_min = std::numeric_limits<double>::infinity();
  depth_max = -std::numeric_limits<double>::infinity();
  for (unsigned int i = 0; i < 8; i++)
  {
    const vector_3d &p = points[i];
    vector_4d pt(p[0], p[1], p[2], 1.0);
    vector_3d res = cam->as_matrix() * pt;
    double d = res[2];
    if (depth_min > d)
      depth_min = d;
    if (depth_max < d)
      depth_max = d;
  }
}

//*****************************************************************************

/// Return a subset of landmark points that project into the given region of interest
std::vector<vector_3d>
filter_visible_landmarks(const camera_sptr &cam,
                         const bounding_box<double> &roi,
                         const std::vector<vital::landmark_sptr> &landmarks)
{
  std::vector<vector_3d> visible_landmarks;

  for (unsigned int i = 0; i < landmarks.size(); i++)
  {
    vector_3d p = landmarks[i]->loc();
    vector_2d pp = cam->project(p);
    if (roi.contains(pp))
    {
      visible_landmarks.push_back(p);
    }
  }

  return visible_landmarks;
}

//*****************************************************************************

/// Robustly compute the bounding planes of the landmarks in a given direction
void
compute_offset_range(const std::vector<vector_3d> &landmarks,
                     const vector_3d &normal,
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
compute_depth_range(const std::vector<vector_3d> &landmarks,
                    const camera_perspective_sptr &cam,
                    double &depth_min, double &depth_max,
                    const double outlier_thresh,
                    const double safety_margin_factor)
{
  depth_min = std::numeric_limits<double>::infinity();
  depth_max = -std::numeric_limits<double>::infinity();

  std::vector<double> depths;

  for (unsigned int i = 0; i < landmarks.size(); i++)
  {
    const vector_3d &p = landmarks[i];
    vector_4d pt(p[0], p[1], p[2], 1.0);
    vector_3d res = cam->as_matrix() * pt;
    depths.push_back(res[2]);
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
