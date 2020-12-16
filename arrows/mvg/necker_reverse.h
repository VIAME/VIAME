// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for Necker reverse functions
 */

#ifndef KWIVER_ARROWS_CORE_NECKER_REVERSE_H_
#define KWIVER_ARROWS_CORE_NECKER_REVERSE_H_

#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// Compute a plane passing through the landmarks
/**
 * Returns the parameters of the best fit plan passing through
 * the centroid of the landmarks.  The plane is [nx, ny, nz, d] where
 * [nx, ny, nz] is a unit normal and d is the offset.
 */
KWIVER_ALGO_MVG_EXPORT
vital::vector_4d
landmark_plane(const vital::landmark_map::map_landmark_t& landmarks);

/// Mirror landmarks about the specified plane
KWIVER_ALGO_MVG_EXPORT
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
KWIVER_ALGO_MVG_EXPORT
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
KWIVER_ALGO_MVG_EXPORT
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
KWIVER_ALGO_MVG_EXPORT
void
necker_reverse(vital::camera_map_sptr& cameras,
               vital::landmark_map_sptr& landmarks,
               bool reverse_landmarks = true);

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
