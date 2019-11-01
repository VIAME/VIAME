/*ckwg +29
 * Copyright 2014-2016, 2019 by Kitware, Inc.
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
 * \brief Header for Necker reverse functions
 */

#ifndef KWIVER_ARROWS_CORE_NECKER_REVERSE_H_
#define KWIVER_ARROWS_CORE_NECKER_REVERSE_H_


#include <arrows/core/kwiver_algo_core_export.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>


namespace kwiver {
namespace arrows {
namespace core {


/// Compute a plane passing through the landmarks
/**
 * Returns the parameters of the best fit plan passing through
 * the centroid of the landmarks.  The plane is [nx, ny, nz, d] where
 * [nx, ny, nz] is a unit normal and d is the offset.
 */
KWIVER_ALGO_CORE_EXPORT
vital::vector_4d
landmark_plane(const vital::landmark_map::map_landmark_t& landmarks);


/// Mirror landmarks about the specified plane
KWIVER_ALGO_CORE_EXPORT
vital::landmark_map_sptr
mirror_landmarks(vital::landmark_map const& landmarks,
                 vital::vector_4d const& plane);


/// Compute the Necker reversal of a camera in place
/**
 * Using the specified landmark plane, apply a Necker reversal
 * transformation to the camera in place.  The camera rotates 180 degrees
 * about the normal vector of the plane at the point where the principal
 * ray intersects the plane.  The camera also rotates 180 degrees about its
 * principal.  The resulting camera provides a similar projection of points
 * near the specified plane, espcially for very long focal lengths.
 */
KWIVER_ALGO_CORE_EXPORT
void
necker_reverse_inplace(vital::simple_camera_perspective& camera,
                       vital::vector_4d const& plane);


/// Compute the Necker reversal of the cameras
/**
* Using the specified landmark plane, apply a Necker reversal
* transformation to each camera.  The camera rotates 180 degrees
* about the normal vector of the plane at the point where the principal
* ray intersects the plane.  The camera also rotates 180 degrees about its
* principal.  The resulting camera provides a similar projection of points
* near the specified plane, espcially for very long focal lengths.
*/
KWIVER_ALGO_CORE_EXPORT
vital::camera_map_sptr
necker_reverse(vital::camera_map const& cameras,
               vital::vector_4d const& plane);


/// Compute an approximate Necker reversal of cameras and landmarks
/**
 * This operation help restart bundle adjustment after falling into
 * a common local minima that with depth reversal that is illustrated
 * by the Necker cube phenomena.
 *
 * The functions finds the axis, A, connecting the mean of the camera centers
 * and mean of the landmark locations.  It then rotates each camera 180 degrees
 * about this axis and also 180 degrees about each cameras own principal axis.
 * The landmarks are mirrored about the plane passing through the
 * mean landmark location and with a normal aligning with axis A.
 * Setting reverse_landmarks to false, prevents this mirroring of the landmarks,
 * leaving them where they originally were.  Only the cameras are modified
 * in this case.
 *
 */
KWIVER_ALGO_CORE_EXPORT
void
necker_reverse(vital::camera_map_sptr& cameras,
               vital::landmark_map_sptr& landmarks,
               bool reverse_landmarks = true);

} // end namespace core
} // end namespace arrows
} // end namespace kwiver

#endif
