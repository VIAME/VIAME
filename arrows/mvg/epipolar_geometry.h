// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for epipolar geometry functions.
 */

#ifndef KWIVER_ARROWS_MVG_EPIPOLAR_GEOMETRY_H_
#define KWIVER_ARROWS_MVG_EPIPOLAR_GEOMETRY_H_

#include <vital/vital_config.h>
#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/essential_matrix.h>
#include <vital/types/fundamental_matrix.h>
#include <vector>

namespace kwiver {
namespace arrows {
namespace mvg {

/// Test corresponding points against a fundamental matrix and mark inliers
/**
 * \param [in]  fm   the fundamental matrix
 * \param [in]  pts1 the vector or corresponding points from the first image
 * \param [in]  pts2 the vector of corresponding points from the second image
 * \param [in]  inlier_scale error distance tolerated for matches to be inliers
 * \returns     a vector of booleans, one for each point pair, the value is
 *                true if this pair is an inlier to the fundamental matrix
 */
KWIVER_ALGO_MVG_EXPORT
std::vector<bool>
mark_fm_inliers(vital::fundamental_matrix const& fm,
                std::vector<vital::vector_2d> const& pts1,
                std::vector<vital::vector_2d> const& pts2,
                double inlier_scale = 1.0);

/// Compute a valid left camera from an essential matrix
/**
 * There are four valid left camera possibilities for any essential
 * matrix (assuming the right camera is the identity camera).
 * This function selects the left camera such that a corresponding
 * pair of points (in normalized coordinates) triangulates
 * in front of both cameras.
 *
 * \param [in]  e        the essential matrix
 * \param [in]  left_pt  a point in normalized coordinates of the left image
 * \param [in]  right_pt a point in normalized coordinates of the right image
 *                       that corresponds with \p left_pt
 * \returns     a camera containing the rotation and unit translation of the
 *                left camera assuming the right camera is the identity
 */
KWIVER_ALGO_MVG_EXPORT
kwiver::vital::simple_camera_perspective
extract_valid_left_camera(const kwiver::vital::essential_matrix_d& e,
                          const kwiver::vital::vector_2d& left_pt,
                          const kwiver::vital::vector_2d& right_pt);

/// Compute the fundamental matrix from a pair of cameras
KWIVER_ALGO_MVG_EXPORT
kwiver::vital::fundamental_matrix_sptr
fundamental_matrix_from_cameras(kwiver::vital::camera_perspective const& right_cam,
                                kwiver::vital::camera_perspective const& left_cam);

/// Compute the essential matrix from a pair of cameras
KWIVER_ALGO_MVG_EXPORT
kwiver::vital::essential_matrix_sptr
essential_matrix_from_cameras(kwiver::vital::camera_perspective const& right_cam,
                              kwiver::vital::camera_perspective const& left_cam);

/// Convert an essential matrix to a fundamental matrix
KWIVER_ALGO_MVG_EXPORT
kwiver::vital::fundamental_matrix_sptr
essential_matrix_to_fundamental(kwiver::vital::essential_matrix const& E,
                                kwiver::vital::camera_intrinsics const& right_cal,
                                kwiver::vital::camera_intrinsics const& left_cal);

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
