// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for VXL bundle adjustment algorithm
 */

#ifndef KWIVER_ARROWS_VXL_BUNDLE_ADJUST_H_
#define KWIVER_ARROWS_VXL_BUNDLE_ADJUST_H_

#include <arrows/vxl/kwiver_algo_vxl_export.h>

#include <vital/algo/bundle_adjust.h>

namespace kwiver {
namespace arrows {
namespace vxl {

/// A class for bundle adjustment of feature tracks using VXL
class KWIVER_ALGO_VXL_EXPORT bundle_adjust
: public vital::algo::bundle_adjust
{
public:
  PLUGIN_INFO( "vxl",
               "Use VXL (vpgl) to bundle adjust cameras and landmarks." )

  /// Constructor
  bundle_adjust();

  /// Destructor
  virtual ~bundle_adjust();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Optimize the camera and landmark parameters given a set of feature tracks
  /**
   * \param [in,out] cameras the cameras to optimize
   * \param [in,out] landmarks the landmarks to optimize
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in] metadata the frame metadata to use as constraints
   */
  virtual void
  optimize(vital::camera_map_sptr& cameras,
           vital::landmark_map_sptr& landmarks,
           vital::feature_track_set_sptr tracks,
           vital::sfm_constraints_sptr constraints = nullptr) const;

  using vital::algo::bundle_adjust::optimize;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace vxl
} // end namespace arrows
} // end namespace kwiver

#endif
