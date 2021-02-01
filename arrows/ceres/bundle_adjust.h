// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for Ceres bundle adjustment algorithm
 */

#ifndef KWIVER_ARROWS_CERES_BUNDLE_ADJUST_H_
#define KWIVER_ARROWS_CERES_BUNDLE_ADJUST_H_

#include <arrows/ceres/kwiver_algo_ceres_export.h>

#include <vital/algo/bundle_adjust.h>

#include <memory>

namespace kwiver {
namespace arrows {
namespace ceres {

/// A class for bundle adjustment of feature tracks using Ceres
class KWIVER_ALGO_CERES_EXPORT bundle_adjust
: public vital::algo::bundle_adjust
{
public:
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

  /// Optimize the camera and landmark parameters given a set of feature tracks
  /**
   * \param [in,out] cameras the cameras to optimize
   * \param [in,out] landmarks the landmarks to optimize
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in] fixed_cameras frame ids for cameras to be fixed in the optimization
   * \param [in] fixed_landmarks landmark ids for landmarks to be fixed in the optimization
   * \param [in] metadata the frame metadata to use as constraints
  */
  virtual void
  optimize(
    kwiver::vital::simple_camera_perspective_map &cameras,
    kwiver::vital::landmark_map::map_landmark_t &landmarks,
    vital::feature_track_set_sptr tracks,
    const std::set<vital::frame_id_t>& fixed_cameras,
    const std::set<vital::landmark_id_t>& fixed_landmarks,
    kwiver::vital::sfm_constraints_sptr constraints = nullptr) const;

  /// Set a callback function to report intermediate progress
  virtual void set_callback(callback_t cb);

  /// This function is called by a Ceres callback to trigger a kwiver callback
  bool trigger_callback();

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif
