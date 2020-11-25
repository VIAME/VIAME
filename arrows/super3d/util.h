// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

 /**
 * \file
 * \brief Header file for util, various helper functions for depth estimation
 */

#ifndef KWIVER_ARROWS_SUPER3D_UTIL_H_
#define KWIVER_ARROWS_SUPER3D_UTIL_H_

#include <vpgl/vpgl_perspective_camera.h>
#include <vil/vil_image_view.h>

namespace kwiver {
namespace arrows {
namespace super3d {

/// Produce the camera corresponding to a downsampled image
/// \param camera The input camera
/// \param scale The scale of the downsampled image (default is 0.5)
/// \return A camera corresponding to the downsampled image
vpgl_perspective_camera<double>
scale_camera(const vpgl_perspective_camera<double>& camera,
             double scale);

/// Produce the camera corresponding to a cropped image
/// \param camera The input camera
/// \param left the left coordinate of the cropping
/// \param top the left coordinate of the cropping
/// \return A camera corresponding to the cropped image
vpgl_perspective_camera<double>
crop_camera(const vpgl_perspective_camera<double>& camera,
      double left,
      double top);

/// Convert a depth map into a height map
/// \param camera the camera corresponding to the depth map
/// \param depth_map input depth map
/// \param height_map output height map
void depth_map_to_height_map(const vpgl_perspective_camera<double>& camera,
                             const vil_image_view<double>& depth_map,
                             vil_image_view<double>& height_map);

/// Convert a height map into a depth map
/// \param camera the camera corresponding to the height map
/// \param height_map input height map
/// \param depth_map output depth map
void height_map_to_depth_map(const vpgl_perspective_camera<double>& camera,
                             const vil_image_view<double>& height_map,
                             vil_image_view<double>& depth_map);

/// Convert a height map into a depth map and scale uncertainty
/**
 * \param [in]  camera         the camera corresponding to the height map
 * \param [in]  height_map     input height map
 * \param [out] depth_map      output depth map
 * \param [in,out] uncertainty uncertainty map to scale in place
 */
void height_map_to_depth_map(vpgl_perspective_camera<double> const& camera,
                             vil_image_view<double> const& height_map,
                             vil_image_view<double>& depth_map,
                             vil_image_view<double>& uncertainty);

} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver

#endif
