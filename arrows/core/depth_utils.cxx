// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief depth estimation utility functions.
*/

#include <arrows/core/depth_utils.h>
#include <vital/math_constants.h>
#include <iostream>
#include <cmath>

using namespace kwiver::vital;
using namespace kwiver::vital::algo;

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

/// Compute a robust 3D bounding box for a set of landmarks
bool
compute_robust_ROI(std::vector<landmark_sptr> const& landmarks,
                   double bounds[6],
                   double percentile,
                   double zmax_percentile,
                   double margin)
{
  unsigned int num_pts = static_cast<unsigned int>(landmarks.size());
  if (num_pts < 2)
  {
    return false;
  }

  std::vector<double> x, y, z;
  x.reserve(num_pts);
  y.reserve(num_pts);
  z.reserve(num_pts);

  for (unsigned int i = 0; i < num_pts; ++i)
  {
    vector_3d pt = landmarks[i]->loc();
    x.push_back(pt[0]);
    y.push_back(pt[1]);
    z.push_back(pt[2]);
  }

  std::sort(x.begin(), x.end());
  std::sort(y.begin(), y.end());
  std::sort(z.begin(), z.end());

  unsigned int min_index = static_cast<unsigned int>(percentile * (num_pts - 1));
  unsigned int max_index = static_cast<unsigned int>(num_pts - 1 - min_index);
  unsigned int zmax_index = static_cast<unsigned int>((num_pts - 1) * (1.0 - zmax_percentile));

  bounds[0] = x[min_index];
  bounds[1] = x[max_index];
  bounds[2] = y[min_index];
  bounds[3] = y[max_index];
  bounds[4] = z[min_index];
  bounds[5] = z[zmax_index];

  for (unsigned i = 0; i < 3; ++i)
  {
    unsigned i_min = 2 * i;
    unsigned i_max = i_min + 1;
    double offset = (bounds[i_max] - bounds[i_min]) * margin;
    bounds[i_min] -= offset;
    bounds[i_max] += offset;
  }

  return true;
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
  {
    vector_2d pp = cam.project(points[0]);
    i0 = i1 = static_cast<int>(pp[0]);
    j0 = j1 = static_cast<int>(pp[1]);
  }

  for (vector_3d const& p : points)
  {
    const vector_2d pp = cam.project(p);
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

  const size_t min_index =
    static_cast<size_t>((offsets.size() - 1) * outlier_thresh);
  const size_t max_index = offsets.size() - 1 - min_index;
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

  const size_t min_index =
    static_cast<size_t>((depths.size() - 1) * outlier_thresh);
  const size_t max_index = depths.size() - 1 - min_index;
  depth_min = depths[min_index];
  depth_max = depths[max_index];

  const double safety_margin = safety_margin_factor * (depth_max - depth_min);
  depth_max += safety_margin;
  depth_min -= safety_margin;
}

//*****************************************************************************

/// Estimate the pixel to world scale over a set of cameras
double
compute_pixel_to_world_scale(kwiver::vital::vector_3d const& minpt,
                             kwiver::vital::vector_3d const& maxpt,
                             std::vector<camera_perspective_sptr> const& cameras)
{
  std::vector<vector_3d> pts = points_of_box(minpt, maxpt);

  double scale = 0.0;
  unsigned int count = 0;
  for (unsigned int c = 0; c < cameras.size(); c++)
  {
    //iterate thru different cameras
    camera_perspective const& cam = *cameras[c];
    matrix_3x4d P = cam.as_matrix();
    vector_3d cam_axis(P(2, 0), P(2, 1), P(2, 2));
    cam_axis.normalize();

    for (unsigned int i = 0; i < pts.size(); i++)
    {
      for (unsigned int j = i + 1; j < pts.size(); j++)
      {
        vector_3d const& pt1 = pts[i];
        vector_3d const& pt2 = pts[j];

        vector_3d vec = pt2 - pt1;
        vector_3d proj = vec.dot(cam_axis) * cam_axis;
        vector_3d pt2p = pt2 - proj;

        double world_dist = (pt2p - pt1).norm();

        vector_2d ppt1 = cam.project(pt1);
        vector_2d ppt2p = cam.project(pt2p);

        double pixel_dist = (ppt2p - ppt1).norm();

        double val = world_dist / pixel_dist;

        if (std::isfinite(val))
        {
          scale += val;
          count++;
        }
      }
    }
  }

  return scale / count;
}

//*****************************************************************************

/// Gather images and masks corresponding to cameras from video sources
int gather_depth_frames(
  camera_perspective_map const& cameras,
  video_input_sptr video,
  video_input_sptr masks,
  frame_id_t ref_frame,
  std::vector<camera_perspective_sptr>& cameras_out,
  std::vector<image_container_sptr>& frames_out,
  std::vector<image_container_sptr>& masks_out,
  gather_callback_t cb)
{
  auto logger = kwiver::vital::get_logger(
    "kwiver.arrows.core.depth_utils.gather_depth_frames");
  cameras_out.clear();
  frames_out.clear();
  masks_out.clear();
  if (!video)
  {
    return 0;
  }
  int ref_index = -1;
  kwiver::vital::timestamp currentTimestamp;
  for (auto const& item : cameras.T_cameras())
  {
    if (cb && !cb(static_cast<unsigned int>(frames_out.size()),
                  static_cast<unsigned int>(cameras.size())))
    {
      break;
    }
    // seek to the frame
    video->seek_frame(currentTimestamp, item.first);
    if (currentTimestamp.get_frame() != item.first)
    {
      LOG_WARN(logger, "Could not find video frame " << item.first);
      continue;
    }
    if (masks)
    {
      masks->seek_frame(currentTimestamp, item.first);
    }
    if (currentTimestamp.get_frame() != item.first)
    {
      LOG_WARN(logger, "Could not find mask frame " << item.first);
      continue;
    }
    auto cam = item.second;
    auto const image = video->frame_image();
    if (!image)
    {
      LOG_WARN(logger, "No image available on frame " << item.first);
      continue;
    }
    auto const mdv = video->frame_metadata();
    if (!mdv.empty())
    {
      image->set_metadata(mdv[0]);
    }
    frames_out.push_back(image);
    cameras_out.push_back(item.second);
    if (masks)
    {
      masks_out.push_back(masks->frame_image());
    }
    if (item.first == ref_frame)
    {
      ref_index = static_cast<int>(frames_out.size()-1);
    }
  }
  return ref_index;
}

//*****************************************************************************

/// Find a subset of cameras within an angular span of a target camera
camera_perspective_map_sptr
find_similar_cameras_angles(camera_perspective const& ref_camera,
                            camera_perspective_map const& cameras,
                            double max_angle,
                            unsigned max_count)
{
  const vector_3d z(0.0, 0.0, 1.0);
  const double cos_max_angle = std::cos(max_angle * deg_to_rad);

  camera_perspective_map::frame_to_T_sptr_map selected_cameras;

  // reference camera principal axis
  vector_3d ref_pa = ref_camera.rotation().inverse() * z;
  for (auto const& item : cameras.T_cameras())
  {
    // current camera principal axis
    vector_3d curr_pa = item.second->rotation().inverse() * z;
    if (ref_pa.dot(curr_pa) < cos_max_angle)
    {
      continue;
    }
    selected_cameras.insert(item);
  }
  // downsample the set of camera to return no more than max_count
  if (max_count > 0  && selected_cameras.size() > max_count)
  {
    camera_perspective_map::frame_to_T_sptr_map sub_selected_cameras;
    std::vector<frame_id_t> frames;
    for (auto const& item : selected_cameras)
    {
      frames.push_back(item.first);
    }
    for (unsigned i = 0; i < max_count; ++i)
    {
      size_t j = (i * selected_cameras.size()) / max_count;
      sub_selected_cameras.insert(*selected_cameras.find(frames[j]));
    }
    selected_cameras = sub_selected_cameras;
  }
  return std::make_shared<camera_perspective_map>(selected_cameras);
}

} //end namespace core
} //end namespace arrows
} //end namespace kwiver
