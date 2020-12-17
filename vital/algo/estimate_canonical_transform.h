// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Definition for canonical transform estimation algorithm
 */

#ifndef VITAL_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_
#define VITAL_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_

#include <vital/vital_config.h>

#include <string>
#include <vector>

#include <vital/algo/algorithm.h>
#include <vital/types/camera_map.h>
#include <vital/types/landmark_map.h>
#include <vital/types/similarity.h>
#include <vital/types/vector.h>

#include <vital/config/config_block.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Algorithm for estimating a canonical transform for cameras and landmarks
/**
 *  A canonical transform is a repeatable transformation that can be recovered
 *  from data.  In this case we assume at most a similarity transformation.
 *  If data sets P1 and P2 are equivalent up to a similarity transformation,
 *  then applying a canonical transform to P1 and separately a
 *  canonical transform to P2 should bring the data into the same coordinates.
 */
class VITAL_ALGO_EXPORT estimate_canonical_transform
  : public kwiver::vital::algorithm_def<estimate_canonical_transform>
{
public:
  /// Name of this algo definition
  static std::string static_type_name() { return "estimate_canonical_transform"; }

  /// Estimate a canonical similarity transform for cameras and points
  /**
   * \param cameras The camera map containing all the cameras
   * \param landmarks The landmark map containing all the 3D landmarks
   * \throws algorithm_exception When the data is insufficient or degenerate.
   * \returns An estimated similarity transform mapping the data to the
   *          canonical space.
   * \note This algorithm does not apply the transformation, it only estimates it.
   */
  virtual kwiver::vital::similarity_d
  estimate_transform(kwiver::vital::camera_map_sptr const cameras,
                     kwiver::vital::landmark_map_sptr const landmarks) const=0;

protected:
    estimate_canonical_transform();

};

/// Shared pointer for similarity transformation algorithms
typedef std::shared_ptr<estimate_canonical_transform> estimate_canonical_transform_sptr;

} } } // end namespace

#endif // VITAL_ALGO_ESTIMATE_CANONICAL_TRANSFORM_H_
