// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief resection_camera algorithm definition

#ifndef VITAL_ALGO_RESECTION_CAMERA_H_
#define VITAL_ALGO_RESECTION_CAMERA_H_

#include <vital/algo/algorithm.h>

#include <vital/types/camera_perspective.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>

#include <unordered_set>
#include <vector>

namespace kwiver {

namespace vital {

namespace algo {

/// An abstract base class to resection a camera using 3D feature and point
/// projection pairs.
class VITAL_ALGO_EXPORT resection_camera
  : public kwiver::vital::algorithm_def< resection_camera >
{
public:
  /// \return name of this algorithm
  static std::string
  static_type_name() { return "resection_camera"; }

  /// Estimate camera parameters from 3D points and their corresponding
  /// projections.
  ///
  /// \param [in] image_points
  ///   the 2D image space locations which are projections of \p world_points
  /// \param [in] world_points
  ///   locations in 3D world space corresponding to the \p image_points
  /// \param [in] initial_calibration
  ///   initial guess on intrinsic parameters of the camera
  /// \param [out] inliers estimated inlier status for the point pairs
  /// \return estimated camera parameters
  virtual
  kwiver::vital::camera_perspective_sptr
  resection(
    std::vector< kwiver::vital::vector_2d > const& image_points,
    std::vector< kwiver::vital::vector_3d > const& world_points,
    kwiver::vital::camera_intrinsics_sptr initial_calibration,
    std::vector< bool >* inliers = nullptr ) const = 0;

  /// Estimate camera parameters for a frame from landmarks and tracks.
  ///
  /// This is a convenience function for resectioning a camera for a particular
  /// frame number in a collection of tracks with corresponding landmarks.
  /// This function extracts corresponding image and worlds points from the
  /// \p tracks and \p landmarks and then calls resection on those.
  /// The image \p width and \p height are used to construct an initial
  /// guess of camera intrinsics.
  ///
  /// \param [in] frame_id frame number for which to estimate a camera
  /// \param [in] landmarks 3D landmark locations to constrain camera
  /// \param [in] tracks 2D feature tracks in image coordinates
  /// \param [in] width image size in the x dimension in pixels
  /// \param [in] height image size in the y dimension in pixels
  /// \param [out] inliers landmark identifiers of inliers
  /// \return estimated camera parameters
  virtual
  kwiver::vital::camera_perspective_sptr
  resection(
    kwiver::vital::frame_id_t frame_id,
    kwiver::vital::landmark_map_sptr landmarks,
    kwiver::vital::feature_track_set_sptr tracks,
    unsigned width, unsigned height,
    std::unordered_set< landmark_id_t >* inliers = nullptr ) const;

  /// Estimate camera parameters for a frame from landmarks and tracks.
  ///
  /// This is a convenience function for resectioning a camera for a particular
  /// frame number in a collection of tracks with corresponding landmarks.
  /// This function extracts corresponding image and worlds points from the
  /// \p tracks and \p landmarks and then calls resection on those.
  ///
  /// \param [in] frame_id frame number for which to estimate a camera
  /// \param [in] landmarks 3D landmarks locations to constrain camera
  /// \param [in] tracks 2D feature tracks in image coordinates
  /// \param [in] initial_calibration
  ///   initial guess on intrinsic parameters of the camera
  /// \param [out] inliers landmark identifiers of inliers
  /// \return estimated camera parameters
  virtual
  kwiver::vital::camera_perspective_sptr
  resection(
    kwiver::vital::frame_id_t frame_id,
    kwiver::vital::landmark_map_sptr landmarks,
    kwiver::vital::feature_track_set_sptr tracks,
    kwiver::vital::camera_intrinsics_sptr initial_calibration,
    std::unordered_set< landmark_id_t >* inliers = nullptr ) const;

protected:
  resection_camera();
};

/// Shared pointer type of base resection_camera algorithm definition class.
using resection_camera_sptr = std::shared_ptr< resection_camera >;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif
