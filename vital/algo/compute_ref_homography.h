// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief compute_ref_homography algorithm definition
 */

#ifndef VITAL_ALGO_COMPUTE_REF_HOMOGRAPHY_H_
#define VITAL_ALGO_COMPUTE_REF_HOMOGRAPHY_H_

#include <vital/vital_config.h>

#include <vector>

#include <vital/algo/algorithm.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/image_container.h>
#include <vital/types/homography_f2f.h>

namespace kwiver {
namespace vital {
namespace algo {

/// Abstract base class for mapping each image to some reference image.
/**
 * This class differs from estimate_homographies in that estimate_homographies
 * simply performs a homography regression from matching feature points. This
 * class is designed to generate different types of homographies from input
 * feature tracks, which can transform each image back to the same coordinate
 * space derived from some initial refrerence image.
 */
class VITAL_ALGO_EXPORT compute_ref_homography
  : public kwiver::vital::algorithm_def<compute_ref_homography>
{
public:

  /// Return the name of this algorithm
  static std::string static_type_name() { return "compute_ref_homography"; }

  /// Estimate the transformation which maps some frame to a reference frame
  /**
   * Similarly to track_features, this class was designed to be called in
   * an online fashion for each sequential frame. The output homography
   * will contain a transformation mapping points from the current frame
   * (with frame_id frame_number) to the earliest possible reference frame
   * via post multiplying points on the current frame with the computed
   * homography.
   *
   * The returned homography is internally allocated and passed back
   * through a smart pointer transferring ownership of the memory to
   * the caller.
   *
   * \param [in]   frame_number frame identifier for the current frame
   * \param [in]   tracks the set of all tracked features from the image
   * \return estimated homography
   */
  virtual kwiver::vital::f2f_homography_sptr
  estimate( kwiver::vital::frame_id_t frame_number,
            kwiver::vital::feature_track_set_sptr tracks ) const = 0;

protected:
  compute_ref_homography();

};

/// Shared pointer type of base compute_ref_homography algorithm definition class
typedef std::shared_ptr<compute_ref_homography> compute_ref_homography_sptr;

} } } // end namespace

#endif // VITAL_COMPUTE_REF_HOMOGRAPHY_H_
