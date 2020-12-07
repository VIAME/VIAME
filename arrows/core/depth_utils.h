// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
* \file
* \brief Header for depth estimation utility functions.
*/

#ifndef DEPTH_UTILS_H_
#define DEPTH_UTILS_H_

#include <arrows/core/kwiver_algo_core_export.h>

#include <vector>
#include <functional>

#include <vital/algo/video_input.h>
#include <vital/types/landmark.h>
#include <vital/types/camera_perspective_map.h>
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

/// The callback function signature used in gather_depth_frames
/**
 * This function is called with two values.  The first is the current loop
 * iteration.  The second is the number of total iterations.  The function
 * must return true or the loop will terminate early.
 */
using gather_callback_t = std::function<bool(unsigned int, unsigned int)>;

/// Gather images and masks corresponding to cameras from video sources
/**
 * This function is used to create parallel vectors of cameras, frames, and
 * masks which are used as input to the compute_depth algorithm.  This function
 * seeks through the videos finding frames corresponding to frames in the
 * camera map.  If a camera, video frame, and mask are all found they are
 * pushed into the corresponding output vectors.  The provided \a ref_frame
 * is the frame number of a frame to be used as a reference.  It should
 * also be included in \cameras.  The return value of this function is the
 * index into the output vectors of this reference frame.
 *
 * The masks video input is optional and can be set to nullptr to skip mask
 * collection.  In this case, \a mask_out will also be empty.
 *
 * The optional callback \a cb is called at the start of each loop
 * iteration to report progress as number of cameras processed out of the
 * total camera map size.  if the callback returns false the loop will
 * terminate early.
 *
 * \param  cameras The map of frame numbers to cameras
 * \param  video The video from which to extract frames matching the cameras
 * \param  masks The video of masks from which to extract mask images
 * \param  ref_frame The video frame which will be the reference for depth
 * \param  cameras_out The output vector containing cameras
 * \param  frames_out The output vector containing frames
 * \param  masks_out The output vector containing masks
 * \param  cb The optional callback to report progress
 * \returns the index into the output vectors of the ref_frame data
 */
KWIVER_ALGO_CORE_EXPORT
int gather_depth_frames(
  kwiver::vital::camera_perspective_map const& cameras,
  kwiver::vital::algo::video_input_sptr video,
  kwiver::vital::algo::video_input_sptr masks,
  kwiver::vital::frame_id_t ref_frame,
  std::vector<kwiver::vital::camera_perspective_sptr>& cameras_out,
  std::vector<kwiver::vital::image_container_sptr>& frames_out,
  std::vector<kwiver::vital::image_container_sptr>& masks_out,
  gather_callback_t cb = nullptr);

/// Find a subset of cameras within an angular span of a target camera
/**
 * Return the subset of all cameras within the specified angle similarity bounds
 * to the reference camera.  If \p max_count is greater than zero then uniformly
 * subsample this many cameras from the selected set.
 *
 * \param  ref_camera The reference camera, find similar cameras similar to this
 * \param  cameras The pool of cameras to select from
 * \param  max_angle The maximum angle in degrees between camera principal rays
 * \param  max_count The maximum number of cameras to return (if 0, return all)
 */
KWIVER_ALGO_CORE_EXPORT
kwiver::vital::camera_perspective_map_sptr
find_similar_cameras_angles(camera_perspective const& ref_camera,
                            camera_perspective_map const& cameras,
                            double max_angle,
                            unsigned max_count = 0);
} //end namespace core
} //end namespace arrows
} //end namespace kwiver

#endif
