/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/// \file
/// \brief Pinhole projection, lens distortion and stereo rectification
///
/// The calib3d that the stereo measurement chain needs, without OpenCV:
/// `cv::projectPoints`, `cv::undistortPoints`, `cv::stereoRectify` and
/// `cv::initUndistortRectifyMap`. P7-T06's plan asks for exactly this --
/// "`measure_using_stereo`/`compute_measurements` keep C++ but use
/// `core_types/math` projection" -- and the rectification came with them
/// because `measurement_utilities` grew a rectified matching path after that
/// plan was written.
///
/// The arithmetic is OpenCV's, step for step, because everything downstream
/// is held to a recording of what OpenCV produced. Where this deliberately
/// differs it is said so at the function: `stereo_rectify` samples its image
/// border in **double** where OpenCV uses float, which moves the rectified
/// focal length by about one part in ten million.
///
/// Distortion coefficients are in OpenCV's order and may be 4, 5, 8, 12 or
/// 14 long: `k1 k2 p1 p2 [k3 [k4 k5 k6 [s1 s2 s3 s4 [taux tauy]]]]`. The
/// thin-prism and tilt terms are **not** implemented -- nothing in VIAME
/// writes them and a calibration file that carries them would be silently
/// mis-modelled, so they are refused rather than ignored.

#ifndef VIAME_MEASUREMENT_PROJECTION_H
#define VIAME_MEASUREMENT_PROJECTION_H

#include "viame_measurement_export.h"

#include <viame/core_types/image.h>
#include <viame/core_types/matrix.h>
#include <viame/core_types/vector.h>

#include <vector>

namespace viame {

namespace measurement {

/// Brown-Conrady coefficients in OpenCV's order, or empty for none.
using distortion_t = std::vector< double >;

// ----------------------------------------------------------------------------
/// Where a point in camera coordinates lands in the image.
///
/// `cv::projectPoints` with no rotation or translation of its own: divide by
/// z, distort, then apply the intrinsic matrix. The two callers that used it
/// both passed a zero rvec and tvec.
VIAME_MEASUREMENT_EXPORT
kwiver::vital::vector_2d
project_point(
  kwiver::vital::vector_3d const& point,
  kwiver::vital::matrix_3x3d const& intrinsics,
  distortion_t const& coefficients );

// ----------------------------------------------------------------------------
/// The same, for a point already rotated into another frame first.
///
/// `cv::projectPoints` with an `rvec` given as a matrix. `stereo_rectify`
/// needs this and nothing else does.
VIAME_MEASUREMENT_EXPORT
kwiver::vital::vector_2d
project_point(
  kwiver::vital::vector_3d const& point,
  kwiver::vital::matrix_3x3d const& rotation,
  kwiver::vital::matrix_3x3d const& intrinsics,
  distortion_t const& coefficients );

// ----------------------------------------------------------------------------
/// `cv::projectPoints` with a full pose: rotate, translate, then project.
VIAME_MEASUREMENT_EXPORT
kwiver::vital::vector_2d
project_point(
  kwiver::vital::vector_3d const& point,
  kwiver::vital::matrix_3x3d const& rotation,
  kwiver::vital::vector_3d const& translation,
  kwiver::vital::matrix_3x3d const& intrinsics,
  distortion_t const& coefficients );

// ----------------------------------------------------------------------------
/// A disparity map as a field of 3D points, which is
/// `cv::reprojectImageTo3D` with `handleMissingValues` false.
///
/// \p disparity is one plane and the result is three: x, y and z in the left
/// rectified camera's frame. Note what OpenCV documents and what every
/// caller in VIAME forgets: a **16-bit signed** disparity is taken to have
/// no fractional bits, and SGBM's has four.
VIAME_MEASUREMENT_EXPORT
kwiver::vital::image_of< float >
reproject_to_3d(
  kwiver::vital::image const& disparity,
  kwiver::vital::matrix_4x4d const& disparity_to_depth );

// ----------------------------------------------------------------------------
/// A rotation matrix from an axis-angle vector, and the inverse.
///
/// `cv::Rodrigues`, both directions. Exported because the stereo pairing
/// needs the vector form to hand to `project_point`.
VIAME_MEASUREMENT_EXPORT
kwiver::vital::matrix_3x3d
rodrigues( kwiver::vital::vector_3d const& vector );

VIAME_MEASUREMENT_EXPORT
kwiver::vital::vector_3d
inverse_rodrigues( kwiver::vital::matrix_3x3d const& matrix );

// ----------------------------------------------------------------------------
/// Undo the lens, rotate, and project through a new matrix.
///
/// `cv::undistortPoints( src, dst, intrinsics, coefficients, rotation,
/// projection )`. With an empty \p coefficients this is exact; with any it is
/// OpenCV's fixed-point iteration, five passes and no convergence test,
/// which is what `cv::undistortPoints` does when given no term criteria.
///
/// \p rotation is `R` -- identity when there is no rectification -- and
/// \p projection is `P`, a 3 by 4 whose **fourth column is ignored**, as
/// `cv::undistortPoints` ignores it: what is mapped here is a direction, so
/// the baseline term of a rectified `P2` has no part in it. Pass a 3 by 3
/// intrinsic matrix widened with a zero column for the un-rectified case, or
/// the identity to get normalised coordinates back, which is what OpenCV's
/// null `P` means.
VIAME_MEASUREMENT_EXPORT
kwiver::vital::vector_2d
undistort_point(
  kwiver::vital::vector_2d const& point,
  kwiver::vital::matrix_3x3d const& intrinsics,
  distortion_t const& coefficients,
  kwiver::vital::matrix_3x3d const& rotation,
  kwiver::vital::matrix_3x4d const& projection );

// ----------------------------------------------------------------------------
/// What a stereo rectification produces.
struct VIAME_MEASUREMENT_EXPORT rectification
{
  /// The rotation that takes each camera into the rectified frame.
  kwiver::vital::matrix_3x3d left_rotation;
  kwiver::vital::matrix_3x3d right_rotation;

  /// The rectified projection matrices. `right_projection( 0, 3 )` is the
  /// baseline times the focal length, negated, for a horizontal rig.
  kwiver::vital::matrix_3x4d left_projection;
  kwiver::vital::matrix_3x4d right_projection;

  /// Disparity to depth, for `reprojectImageTo3D`.
  kwiver::vital::matrix_4x4d disparity_to_depth;
};

// ----------------------------------------------------------------------------
/// `cv::stereoRectify` with `CALIB_ZERO_DISPARITY` and `alpha = 0`.
///
/// Those are the only settings VIAME asks for, and both are load-bearing:
/// zero disparity puts the two principal points at the same place, so a
/// correspondence is a pure horizontal shift, and an alpha of zero zooms in
/// until the rectified image has no invalid border.
///
/// \p rotation and \p translation take a point from the left camera's frame
/// into the right's, which is what a calibration file holds.
VIAME_MEASUREMENT_EXPORT
rectification
stereo_rectify(
  kwiver::vital::matrix_3x3d const& left_intrinsics,
  distortion_t const& left_distortion,
  kwiver::vital::matrix_3x3d const& right_intrinsics,
  distortion_t const& right_distortion,
  size_t width, size_t height,
  kwiver::vital::matrix_3x3d const& rotation,
  kwiver::vital::vector_3d const& translation );

// ----------------------------------------------------------------------------
/// The two sampling maps a rectification needs, which is
/// `cv::initUndistortRectifyMap` into a pair of `CV_32FC1`.
///
/// Each entry is where in the source image an output pixel comes from, so
/// `image_ops::remap` turns them into a rectified frame.
VIAME_MEASUREMENT_EXPORT
void
rectification_maps(
  kwiver::vital::matrix_3x3d const& intrinsics,
  distortion_t const& coefficients,
  kwiver::vital::matrix_3x3d const& rotation,
  kwiver::vital::matrix_3x4d const& projection,
  size_t width, size_t height,
  kwiver::vital::image_of< float >& map_x,
  kwiver::vital::image_of< float >& map_y );

} // namespace measurement

} // namespace viame

#endif // VIAME_MEASUREMENT_PROJECTION_H
