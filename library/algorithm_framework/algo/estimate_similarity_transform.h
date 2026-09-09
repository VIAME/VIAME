// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief Definition for similarity transform estimation algorithm

#ifndef VITAL_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_
#define VITAL_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_

#include <viame/algorithm_framework/vital_config.h>

#include <string>
#include <vector>

#include <viame/algorithm_framework/algo/algorithm.h>
#include <viame/core_types/camera_map.h>
#include <viame/core_types/camera_perspective.h>
#include <viame/core_types/landmark.h>
#include <viame/core_types/landmark_map.h>
#include <viame/core_types/similarity.h>
#include <viame/core_types/vector.h>

#include <viame/algorithm_framework/config/config_block.h>

namespace kwiver {

namespace vital {

namespace algo {

/// Algorithm for estimating the similarity transform between two point sets
class VITAL_ALGO_EXPORT estimate_similarity_transform
  : public kwiver::vital::algorithm
{
public:
  estimate_similarity_transform();
  PLUGGABLE_INTERFACE( estimate_similarity_transform );

  /// Estimate the similarity transform between two corresponding point sets
  ///
  /// \param src List of length N of 3D points in the src space.
  /// \param dst List of length N of 3D points in the dst space.
  /// \throws algorithm_exception When the src and dst point sets are
  ///                             misaligned, insufficient or degenerate.
  /// \returns An estimated similarity transform mapping 3D points in the
  ///          \c src space to points in the \c dst space.
  virtual kwiver::vital::similarity_d
  estimate_transform(
    std::vector< kwiver::vital::vector_3d > const& src,
    std::vector< kwiver::vital::vector_3d > const& dst ) const = 0;

  /// Estimate the similarity transform between two corresponding sets of
  /// cameras
  ///
  /// \param src List of length N of cameras in the src space.
  /// \param dst List of length N of cameras in the dst space.
  /// \throws algorithm_exception When the src and dst point sets are
  ///                             misaligned, insufficient or degenerate.
  /// \returns An estimated similarity transform mapping camera centers in the
  ///          \c src space to camera centers in the \c dst space.
  virtual kwiver::vital::similarity_d
  estimate_transform(
    std::vector< kwiver::vital::camera_perspective_sptr > const& src,
    std::vector< kwiver::vital::camera_perspective_sptr > const& dst ) const;

  /// Estimate the similarity transform between two corresponding sets of
  /// landmarks.
  ///
  /// \param src List of length N of landmarks in the src space.
  /// \param dst List of length N of landmarks in the dst space.
  /// \throws algorithm_exception When the src and dst point sets are
  ///                             misaligned, insufficient or degenerate.
  /// \returns An estinated similarity transform mapping landmark locations in
  ///          the \c src space to located in the \c dst space.
  virtual kwiver::vital::similarity_d
  estimate_transform(
    std::vector< kwiver::vital::landmark_sptr > const& src,
    std::vector< kwiver::vital::landmark_sptr > const& dst )
  const;

  /// Estimate the similarity transform between two corresponding camera maps
  ///
  /// Cameras with corresponding frame IDs in the two maps are paired for
  /// transform estimation. Cameras with no corresponding frame ID in the other
  /// map are ignored. An algorithm_exception is thrown if there are no shared
  /// frame IDs between the two provided maps (nothing to pair).
  ///
  /// \throws algorithm_exception When the from and to point sets are
  ///                             misaligned, insufficient or degenerate.
  /// \param src Map of original cameras, sharing N frames with the transformed
  ///            cameras, where N > 0.
  /// \param dst Map of transformed cameras, sharing N frames with the original
  ///            cameras, where N > 0.
  /// \returns An estimated similarity transform mapping camera centers in the
  ///          \c src space to camera centers in the \c dst space.
  virtual kwiver::vital::similarity_d
  estimate_transform(
    kwiver::vital::camera_map_sptr const src,
    kwiver::vital::camera_map_sptr const dst ) const;

  /// Estimate the similarity transform between two corresponding landmark maps
  ///
  /// Landmarks with corresponding frame IDs in the two maps are paired for
  /// transform estimation. Landmarks with no corresponding frame ID in the
  /// other map are ignored. An algoirithm_exception is thrown if there are no
  /// shared frame IDs between the two provided maps (nothing to pair).
  ///
  /// \throws algorithm_exception When the from and to point sets are
  ///                             misaligned, insufficient or degenerate.
  /// \param src Map of original landmarks, sharing N frames with the
  ///            transformed landmarks, where N > 0.
  /// \param dst Map of transformed landmarks, sharing N frames with the
  ///            original landmarks, where N > 0.
  /// \returns An estimated similarity transform mapping landmark centers in the
  ///          \c src space to camera centers in the \c dst space.
  virtual kwiver::vital::similarity_d
  estimate_transform(
    kwiver::vital::landmark_map_sptr const src,
    kwiver::vital::landmark_map_sptr const dst ) const;
};

/// Shared pointer for similarity transformation algorithms
typedef std::shared_ptr< estimate_similarity_transform >
  estimate_similarity_transform_sptr;

} // namespace algo

} // namespace vital

} // namespace kwiver

#endif // VITAL_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_
