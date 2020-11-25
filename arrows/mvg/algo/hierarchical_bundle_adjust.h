// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header defining the hierarchical_bundle_adjust algorithm
 */

#ifndef KWIVER_ARROWS_MVG_HIERARCHICAL_BUNDLE_ADJUST_H_
#define KWIVER_ARROWS_MVG_HIERARCHICAL_BUNDLE_ADJUST_H_

#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/algo/algorithm.h>
#include <vital/algo/bundle_adjust.h>
#include <vital/config/config_block.h>

namespace kwiver {
namespace arrows {
namespace mvg {

class KWIVER_ALGO_MVG_EXPORT hierarchical_bundle_adjust
  : public vital::algo::bundle_adjust
{
public:
  PLUGIN_INFO( "hierarchical",
               "Run a bundle adjustment algorithm in a temporally hierarchical fashion"
               " (useful for video)" )

  /// Constructor
  hierarchical_bundle_adjust();
  /// Destructor
  virtual ~hierarchical_bundle_adjust() noexcept;

  /// Get this algorithm's \link kwiver::vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's configuration vital::config_block is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Optimize the camera and landmark parameters given a set of tracks
  virtual void optimize(vital::camera_map_sptr & cameras,
                        vital::landmark_map_sptr & landmarks,
                        vital::feature_track_set_sptr tracks,
                        vital::sfm_constraints_sptr constraints = nullptr) const;

  using vital::algo::bundle_adjust::optimize;

private:
  // private implementation class
  class priv;
  std::unique_ptr<priv> d_;
};

/// Type definition for shared pointer for hierarchical_bundle_adjust algorithm
typedef std::shared_ptr<hierarchical_bundle_adjust> hierarchical_bundle_adjust_sptr;

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
