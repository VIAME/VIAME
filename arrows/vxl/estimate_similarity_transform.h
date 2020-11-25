// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief VXL version of similarity transform estimation
 */

#ifndef KWIVER_ARROWS_VXL_ESTIMATE_SIMILARITY_TRANSFORM_H_
#define KWIVER_ARROWS_VXL_ESTIMATE_SIMILARITY_TRANSFORM_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/estimate_similarity_transform.h>
#include <vital/types/vector.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// VXL implementation of similarity transform estimation
class KWIVER_ALGO_VXL_EXPORT estimate_similarity_transform
  : public vital::algo::estimate_similarity_transform
{
public:
  PLUGIN_INFO( "vxl",
               "Use VXL (vpgl) to estimate a 3D similarity transformation "
               "between corresponding landmarks." )

  // No custom configuration at this time
  /// \cond Doxygen Suppress
  virtual void set_configuration(vital::config_block_sptr /*config*/) { };
  virtual bool check_configuration(vital::config_block_sptr /*config*/) const { return true; }
  /// \endcond

  /// Estimate the similarity transform between two corresponding point sets
  /**
   * \param from List of length N of 3D points in the from space.
   * \param to   List of length N of 3D points in the to space.
   * \throws algorithm_exception When the from and to points sets are
   *                             misaligned, insufficient or degenerate.
   * \returns An estimated similarity transform mapping 3D points in the
   *          \c from space to points in the \c to space (i.e. transforms
   *          \c from into \c to).
   */
  virtual vital::similarity_d
  estimate_transform(std::vector<vital::vector_3d> const& from,
                     std::vector<vital::vector_3d> const& to) const;
  using vital::algo::estimate_similarity_transform::estimate_transform;

};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
