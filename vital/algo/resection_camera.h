// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief resection_camera algorithm definition
 */

#ifndef VITAL_ALGO_RESECTION_CAMERA_H_
#define VITAL_ALGO_RESECTION_CAMERA_H_

#include <vital/vital_config.h>

#include <vector>
#include <memory>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_set.h>
#include <vital/types/match_set.h>
#include <vital/types/camera_perspective.h>

namespace kwiver {
namespace vital {
namespace algo {

/// An abstract base class to resection a camera using 3D feature
/// and point projection pairs.

class VITAL_ALGO_EXPORT resection_camera
  : public kwiver::vital::algorithm_def<resection_camera>
{
public:
  /// Return the name of this algorithm
  static std::string static_type_name() { return "resection_camera"; }

  /// Estimate camera parameters from 3D points and their corresponding projections
  ///
  /// \param [in]  pts2d 2d projections of pts3d in the same order as pts3d
  /// \param [in]  pts3d 3d points in the same order as pts2d.  Both must be same size.
  /// \param [in]  init_cal the initial guess intrinsic parameters of the camera
  /// \param [out] inliers for each point, the value is true if
  ///                      this pair is an inlier to the estimate
  virtual
  kwiver::vital::camera_perspective_sptr
  resection(std::vector<vector_2d> const& pts2d,
            std::vector<vector_3d> const& pts3d,
            kwiver::vital::camera_intrinsics_sptr init_cal = nullptr,
            std::vector<bool>& inliers) const = 0;

  /// Estimate camera parameters for a frame from landmarks and tracks
  ///
  /// \param [in]  frame     frame number for which to estimate a camera
  /// \param [in]  landmarks 3D landmarks locations to constrain camera
  /// \param [in]  tracks    2D feature tracks in image coordinates
  /// \param [in]  init_cal  the initial guess intrinsic parameters of the camera
  virtual
  kwiver::vital::camera_perspective_sptr
  resection(kwiver::vital::frame_id_t const& frame,
            kwiver::vital::landmark_map_sptr landmarks,
            kwiver::vital::feature_track_set_sptr tracks,
            kwiver::vital::camera_intrinsics_sptr init_cal = nullptr) const;

protected:
  resection_camera();

};

/// Shared pointer type of base resection_camera algorithm definition class
typedef std::shared_ptr<resection_camera> resection_camera_sptr;

} } } // end namespace

#endif // VITAL_ALGO_RESECTION_CAMERA_H_
