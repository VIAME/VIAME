/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
* \file
* \brief Header defining Opencv algorithm implementation of camera optimization for a stereo setup.
*/

#ifndef VIAME_OPENCV_OPTIMIZE_STEREO_CAMERAS_H
#define VIAME_OPENCV_OPTIMIZE_STEREO_CAMERAS_H

#include "viame_opencv_export.h"

#include <vital/algo/optimize_cameras.h>
#include <vital/plugin_management/pluggable_macro_magic.h>

#include "calibrate_stereo_cameras.h"
#include "filter_stereo_feature_tracks.h"

#include <opencv2/core/core.hpp>

namespace viame {

/// A class for optimization of camera paramters using OpenCV
class VIAME_OPENCV_EXPORT optimize_stereo_cameras
  : public kwiver::vital::algo::optimize_cameras
{
public:
  PLUGGABLE_IMPL( optimize_stereo_cameras,
                  kwiver::vital::algo::optimize_cameras,
                  "ocv_optimize_stereo_cameras",
                  "Camera optimizer for stereo configurations.",
    PARAM_DEFAULT( image_width, unsigned, "sensor image width (0 to derive from data)", 0 )
    PARAM_DEFAULT( image_height, unsigned, "sensor image height (0 to derive from data)", 0 )
    PARAM_DEFAULT( frame_count_threshold, unsigned, "max number of frames to use during optimization", 0 )
    PARAM_DEFAULT( output_calibration_directory, std::string, "output path for the generated calibration files (OpenCV YAML format)", "" )
    PARAM_DEFAULT( output_json_file, std::string, "output path for JSON calibration file (compatible with camera_rig_io)", "" )
    PARAM_DEFAULT( square_size, double, "calibration pattern square size in world units (e.g., mm)", 1.0 )
  )

  virtual ~optimize_stereo_cameras() = default;

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
  // Shared calibration utility
  mutable calibrate_stereo_cameras m_calibrator;

  static kwiver::vital::feature_track_set_sptr
  merge_features_track( const kwiver::vital::feature_track_set_sptr& feature_track,
                        const kwiver::vital::feature_track_set_sptr& feature_track1,
                        const kwiver::vital::feature_track_set_sptr& feature_track2 );

  void calibrate_camera( const kwiver::vital::camera_sptr& camera,
                         const kwiver::vital::feature_track_set_sptr& features,
                         const kwiver::vital::landmark_map_sptr& landmarks,
                         const std::string& suffix ) const;

  void calibrate_stereo_camera( kwiver::vital::camera_map::map_camera_t cameras,
                                const kwiver::vital::feature_track_set_sptr& features1,
                                const kwiver::vital::feature_track_set_sptr& features2,
                                const kwiver::vital::landmark_map_sptr& landmarks1,
                                const kwiver::vital::landmark_map_sptr& landmarks2 ) const;

  StereoPointCoordinates
  convert_features_and_landmarks_to_calib_points( const FeatureTracks& features,
                                                  const Landmarks& landmarks,
                                                  bool& success ) const;

  bool try_improve_camera_calibration( const std::vector< std::vector< cv::Point3f > >& world_points,
                                       const std::vector< std::vector< cv::Point2f > >& image_points,
                                       const cv::Size& image_size,
                                       cv::Mat& K1, cv::Mat& D1, cv::Mat& R1, cv::Mat& T1,
                                       int flags, double max_error, double& error,
                                       const std::string& context ) const;

  void write_stereo_calibration_file( const cv::Mat& M1, const cv::Mat& M2,
                                      const std::vector< double >& D1,
                                      const std::vector< double >& D2,
                                      const cv::Mat& R, const cv::Mat& T,
                                      const cv::Mat& R1, const cv::Mat& R2,
                                      const cv::Mat& P1, const cv::Mat& P2,
                                      const cv::Mat& Q ) const;
};

} // end namespace viame

#endif /* VIAME_OPENCV_OPTIMIZE_STEREO_CAMERAS_H */
