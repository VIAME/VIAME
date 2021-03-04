// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Definition for similarity transform estimation algorithm
 */

#ifndef VITAL_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_
#define VITAL_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_

#include <vital/vital_config.h>

#include <string>
#include <vector>

#include <vital/algo/algorithm.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark.h>
#include <vital/types/landmark_map.h>
#include <vital/types/similarity.h>
#include <vital/types/vector.h>

#include <vital/config/config_block.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Algorithm for estimating the similarity transform between two point sets
class VITAL_ALGO_EXPORT estimate_similarity_transform
  : public kwiver::vital::algorithm_def<estimate_similarity_transform>
{
public:
  /// Name of this algo definition
  static std::string static_type_name() { return "estimate_similarity_transform"; }

  /// Estimate the similarity transform between two corresponding point sets
  /**
   * \param from List of length N of 3D points in the from space.
   * \param to   List of length N of 3D points in the to space.
   * \throws algorithm_exception When the from and to point sets are
   *                             misaligned, insufficient or degenerate.
   * \returns An estimated similarity transform mapping 3D points in the
   *          \c from space to points in the \c to space.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(std::vector<kwiver::vital::vector_3d> const& from,
                     std::vector<kwiver::vital::vector_3d> const& to) const = 0;

  /// Estimate the similarity transform between two corresponding sets of cameras
  /**
   * \param from List of length N of cameras in the from space.
   * \param to   List of length N of cameras in the to space.
   * \throws algorithm_exception When the from and to point sets are
   *                             misaligned, insufficient or degenerate.
   * \returns An estimated similarity transform mapping camera centers in the
   *          \c from space to camera centers in the \c to space.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(
      std::vector<kwiver::vital::camera_perspective_sptr> const& from,
      std::vector<kwiver::vital::camera_perspective_sptr> const& to) const;

  /// Estimate the similarity transform between two corresponding sets of landmarks.
  /**
   * \param from List of length N of landmarks in the from space.
   * \param to   List of length N of landmarks in the to space.
   * \throws algorithm_exception When the from and to point sets are
   *                             misaligned, insufficient or degenerate.
   * \returns An estinated similarity transform mapping landmark locations in
   *          the \c from space to located in the \c to space.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(std::vector<kwiver::vital::landmark_sptr> const& from,
                     std::vector<kwiver::vital::landmark_sptr> const& to) const;

  /// Estimate the similarity transform between two corresponding camera maps
  /**
   * Cameras with corresponding frame IDs in the two maps are paired for
   * transform estimation. Cameras with no corresponding frame ID in the other
   * map are ignored. An algorithm_exception is thrown if there are no shared
   * frame IDs between the two provided maps (nothing to pair).
   *
   * \throws algorithm_exception When the from and to point sets are
   *                             misaligned, insufficient or degenerate.
   * \param from Map of original cameras, sharing N frames with the transformed
   *             cameras, where N > 0.
   * \param to   Map of transformed cameras, sharing N frames with the original
   *             cameras, where N > 0.
   * \returns An estimated similarity transform mapping camera centers in the
   *          \c from space to camera centers in the \c to space.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(kwiver::vital::camera_map_sptr const from,
                     kwiver::vital::camera_map_sptr const to) const;

  /// Estimate the similarity transform between two corresponding landmark maps
  /**
   * Landmarks with corresponding frame IDs in the two maps are paired for
   * transform estimation. Landmarks with no corresponding frame ID in the
   * other map are ignored. An algoirithm_exception is thrown if there are no
   * shared frame IDs between the two provided maps (nothing to pair).
   *
   * \throws algorithm_exception When the from and to point sets are
   *                             misaligned, insufficient or degenerate.
   * \param from Map of original landmarks, sharing N frames with the
   *             transformed landmarks, where N > 0.
   * \param to   Map of transformed landmarks, sharing N frames with the
   *             original landmarks, where N > 0.
   * \returns An estimated similarity transform mapping landmark centers in the
   *          \c from space to camera centers in the \c to space.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(kwiver::vital::landmark_map_sptr const from,
                     kwiver::vital::landmark_map_sptr const to) const;

protected:
    estimate_similarity_transform();

};

/// Shared pointer for similarity transformation algorithms
typedef std::shared_ptr<estimate_similarity_transform> estimate_similarity_transform_sptr;

} // end namespace algo
} // end namespace vital
} // end namespace kwiver

#endif // VITAL_ALGO_ESTIMATE_SIMILARITY_TRANSFORM_H_
