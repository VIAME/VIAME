/*ckwg +29
 * Copyright 2012-2019 by Kitware, Inc.
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


} // end namespace super3d
} // end namespace arrows
} // end namespace kwiver

#endif
