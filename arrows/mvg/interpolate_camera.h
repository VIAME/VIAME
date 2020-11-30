// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for camera interpolation functions
 */

#ifndef KWIVER_ARROWS_MVG_INTERPOLATE_CAMERA_H_
#define KWIVER_ARROWS_MVG_INTERPOLATE_CAMERA_H_

#include <vital/vital_config.h>
#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vector>
#include <vital/types/camera_perspective.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// Generate an interpolated camera between \c A and \c B by a given fraction \c f
/**
 * \c f should be 0 < \c f < 1. A value outside this range is valid, but \c f
 * must not be 0.
 *
 * \param A Camera to interpolate from.
 * \param B Camera to interpolate to.
 * \param f Decimal fraction in between A and B for the returned camera to represent.
 */
KWIVER_ALGO_MVG_EXPORT
vital::simple_camera_perspective
interpolate_camera(vital::simple_camera_perspective const& A,
                   vital::simple_camera_perspective const& B, double f);

/// Genreate an interpolated camera from sptrs
/**
 * \relatesalso interpolate_camera
 *
 */
KWIVER_ALGO_MVG_EXPORT
vital::camera_perspective_sptr
interpolate_camera(vital::camera_perspective_sptr A,
                   vital::camera_perspective_sptr B, double f);

/// Generate N evenly interpolated cameras in between \c A and \c B
/**
 * \c n must be >= 1.
 */
KWIVER_ALGO_MVG_EXPORT
void interpolated_cameras(vital::simple_camera_perspective const& A,
                          vital::simple_camera_perspective const& B,
                          size_t n,
                          std::vector< vital::simple_camera_perspective > & interp_cams);

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif // ALGORITHMS_INTERPOLATE_CAMERA_H_
