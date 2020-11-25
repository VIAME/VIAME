// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief Header for MVG camera and landmark initialization algorithm
 */

#ifndef KWIVER_ARROWS_MVG_INITIALIZE_CAMERAS_LANDMARKS_H_
#define KWIVER_ARROWS_MVG_INITIALIZE_CAMERAS_LANDMARKS_H_

#include <arrows/mvg/kwiver_algo_mvg_export.h>

#include <vital/algo/initialize_cameras_landmarks.h>

namespace kwiver {
namespace arrows {
namespace mvg {

/// A class for initialization of cameras and landmarks
class KWIVER_ALGO_MVG_EXPORT initialize_cameras_landmarks
: public vital::algo::initialize_cameras_landmarks
{
public:
  PLUGIN_INFO( "mvg",
               "Run SfM to estimate new cameras and landmarks "
               "using feature tracks." )

  /// Constructor
  initialize_cameras_landmarks();

  /// Destructor
  virtual ~initialize_cameras_landmarks();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration(vital::config_block_sptr config);
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration(vital::config_block_sptr config) const;

  /// Initialize the camera and landmark parameters given a set of feature tracks
  /**
   * The algorithm creates an initial estimate of any missing cameras and
   * landmarks using the available cameras, landmarks, and feature tracks.
   * It may optionally revise the estimates of exisiting cameras and landmarks.
   *
   * \param [in,out] cameras the cameras to initialize
   * \param [in,out] landmarks the landmarks to initialize
   * \param [in] tracks the feature tracks to use as constraints
   * \param [in] metadata the frame metadata to use as constraints
   */
  virtual void
  initialize(vital::camera_map_sptr& cameras,
             vital::landmark_map_sptr& landmarks,
             vital::feature_track_set_sptr tracks,
             vital::sfm_constraints_sptr constraints = nullptr) const;

  /// Set a callback function to report intermediate progress
  virtual void set_callback(callback_t cb);

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> m_priv;
};

} // end namespace mvg
} // end namespace arrows
} // end namespace kwiver

#endif
