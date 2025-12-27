/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
* \file
* \brief Header defining Opencv algorithm implementation of camera optimization for a stereo setup.
*/

#ifndef VIAME_OPENCV_OPTIMIZE_STEREO_CAMERAS_H
#define VIAME_OPENCV_OPTIMIZE_STEREO_CAMERAS_H

#include <plugins/opencv/viame_opencv_export.h>

#include <vital/algo/optimize_cameras.h>

#include <memory>

namespace viame {

/// A class for optimization of camera paramters using OpenCV
class VIAME_OPENCV_EXPORT optimize_stereo_cameras :
  public kwiver::vital::algorithm_impl<
    optimize_stereo_cameras, kwiver::vital::algo::optimize_cameras >
{
public:
  PLUGIN_INFO( "ocv_optimize_stereo_cameras",
               "Camera optimizer for stereo configurations." )

  /// Constructor
  optimize_stereo_cameras();

  /// Destructor
  virtual ~optimize_stereo_cameras();

  /// Get this algorithm's \link vital::config_block configuration block \endlink
  virtual kwiver::vital::config_block_sptr get_configuration() const;
  /// Set this algorithm's properties via a config block
  virtual void set_configuration( kwiver::vital::config_block_sptr config );
  /// Check that the algorithm's currently configuration is valid
  virtual bool check_configuration( kwiver::vital::config_block_sptr config ) const;

  /// Optimize Stereo camera parameters given sets of landmarks and feature tracks
  /**
   * We only optimize Stereo cameras setup from corresponding feature and landmarks.
   *
   * \throws invalid_value When one or more of the given pointer is Null.
   *
   * \param[in,out] cameras   Cameras to optimize, size needs to be two here.
   * \param[in]     tracks    The feature tracks to use as constraints.
   * \param[in]     landmarks The landmarks the cameras are viewing.
   * \param[in]     metadata  The optional metadata to constrain the
   *                          optimization.
   */
  virtual void
  optimize( kwiver::vital::camera_map_sptr& cameras,
            kwiver::vital::feature_track_set_sptr tracks,
            kwiver::vital::landmark_map_sptr landmarks,
            kwiver::vital::sfm_constraints_sptr constraints = nullptr ) const;

  /// @brief Left / Right camera optimization
  void
  optimize( kwiver::vital::camera_map::map_camera_t cams,
            const std::vector< kwiver::vital::feature_track_set_sptr >& tracks,
            const std::vector< kwiver::vital::landmark_map_sptr >& landmarks ) const;

  /// Optimize a single camera given corresponding features and landmarks
  /**
   * This function assumes that 2D features viewed by this camera have
   * already been put into correspondence with 3D landmarks by aligning
   * them into two parallel vectors
   *
   * \param[in,out] camera    The camera to optimize.
   * \param[in]     features  The vector of features observed by \p camera
   *                          to use as constraints.
   * \param[in]     landmarks The vector of landmarks corresponding to
   *                          \p features.
   * \param[in]     metadata  The optional metadata to constrain the
   *                          optimization.
   */
  virtual void
  optimize( kwiver::vital::camera_perspective_sptr& camera,
            const std::vector< kwiver::vital::feature_sptr >& features,
            const std::vector< kwiver::vital::landmark_sptr >& landmarks,
            kwiver::vital::sfm_constraints_sptr constraints = nullptr ) const;

private:
  /// private implementation class
  class priv;
  const std::unique_ptr<priv> d_;
};

} // end namespace viame

#endif /* VIAME_OPENCV_OPTIMIZE_STEREO_CAMERAS_H */
