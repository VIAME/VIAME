/*ckwg +29
 * Copyright 2016, 2019 by Kitware, Inc.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 *  * Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 *
 *  * Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 *  * Neither name of Kitware, Inc. nor the names of any contributors may be used
 *    to endorse or promote products derived from this software without specific
 *    prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * \file
 * \brief Internal header for helper classes containing Ceres options
 */

#ifndef KWIVER_ARROWS_CERES_CAMERA_OPTIONS_H_
#define KWIVER_ARROWS_CERES_CAMERA_OPTIONS_H_

#include <vital/vital_config.h>
#include <vital/config/config_block.h>
#include <vital/types/camera_perspective.h>
#include <vital/types/camera_map.h>
#include <vital/types/sfm_constraints.h>
#include <arrows/ceres/types.h>

#include <unordered_map>


namespace kwiver {
namespace arrows {
namespace ceres {


/// Ceres solver options class
/**
 * The intended use of this class is for a PIMPL for an algorithm to
 * inherit from this class to share these options with that algorithm
 */
class solver_options
{
public:
  /// Constructor
  solver_options();

  /// Copy Constructor
  solver_options(const solver_options& other);

  /// populate the config block with options
  void get_configuration(vital::config_block_sptr config) const;

  /// set the member variables from the config block
  void set_configuration(vital::config_block_sptr config);

  /// the Ceres solver options
  ::ceres::Solver::Options options;
};


/// Camera options class
/**
 * The intended use of this class is for a PIMPL for an algorithm to
 * inherit from this class to share these options with that algorithm
 */
class camera_options
{
public:
  /// typedef for camera parameter map
  typedef std::unordered_map<vital::frame_id_t, std::vector<double> > cam_param_map_t;
  typedef std::unordered_map<vital::frame_id_t, unsigned int> cam_intrinsic_id_map_t;
  typedef std::vector<std::pair<vital::frame_id_t, double *> > frame_params_t;

  /// Constructor
  camera_options();

  /// Copy Constructor
  camera_options(const camera_options& other);

  /// populate the config block with options
  void get_configuration(vital::config_block_sptr config) const;

  /// set the member variables from the config block
  void set_configuration(vital::config_block_sptr config);

  /// Return true if any options to optimize intrinsic parameters are set
  bool optimize_intrinsics() const;

  /// extract the extrinsic paramters from a camera into the parameter array
  /**
   *  \param [in]  camera The camera object to extract data from
   *  \param [out] params and array of 6 doubles to populate with parameters
   *
   *  This function is the inverse of update_camera_extrinsics
   */
  void extract_camera_extrinsics(const vital::camera_perspective_sptr camera,
                                 double* params) const;

  /// Update a camera object to use extrinsic parameters from an array
  /**
   *  \param [out] camera The simple_camera instance to update
   *  \param [in] params The array of 6 doubles to extract the data from
   *
   *  This function is the inverse of extract_camera_extrinsics
   */
  void update_camera_extrinsics(
    std::shared_ptr<vital::simple_camera_perspective> camera,
    double const* params) const;

  /// extract the paramters from camera intrinsics into the parameter array
  /**
   *  \param [in]  K The camera intrinsics object to extract data from
   *  \param [out] params and array of double to populate with parameters
   *
   *  \note the size of param is at least 5 but may be up to 12 depending
   *  on the number of distortion parameters used.
   *
   *  This function is the inverse of update_camera_intrinsics
   */
  void extract_camera_intrinsics(const vital::camera_intrinsics_sptr K,
                                 double* params) const;

  /// update the camera intrinsics from a parameter array
  /**
   *  \param [out] K The simple_camera_intrinsics instance to update
   *  \param [in] params The array of doubles to extract the data from
   *
   *  This function is the inverse of extract_camera_intrinsics
   */
  void update_camera_intrinsics(std::shared_ptr<vital::simple_camera_intrinsics> K,
                                const double* params) const;

  /// extract the set of all unique intrinsic and extrinsic parameters from a camera map
  /**
   *  \param [in]  cameras    The map of frame numbers to cameras to extract parameters from
   *  \param [out] ext_params A map from frame number to vector of extrinsic parameters
   *  \param [out] int_params A vector of unique camera intrinsic parameter vectors
   *  \param [out] int_map    A map from frame number to index into \p int_params.
   *                          The mapping may be many-to-one for shared intrinsics.
   *
   *  This function is the inverse of update_camera_parameters
   */
  void extract_camera_parameters(vital::camera_map::map_camera_t const& cameras,
                                 cam_param_map_t& ext_params,
                                 std::vector<std::vector<double> >& int_params,
                                 cam_intrinsic_id_map_t& int_map) const;

  /// update the camera objects using the extracted camera parameters
  /**
   *  \param [out] cameras    The map of frame numbers to cameras to update
   *  \param [in]  ext_params A map from frame number to vector of extrinsic parameters
   *  \param [in]  int_params A vector of unique camera intrinsic parameter vectors
   *  \param [in]  int_map    A map from frame number to index into \p int_params.
   *                          The mapping may be many-to-one for shared intrinsics.
   *
   *  The original camera_intrinsic objects are reused if they were not optimized.
   *  Otherwise new camera_intrinsic instances are created.
   *
   *  This function is the inverse of extract_camera_parameters
   */
  void
  update_camera_parameters(vital::camera_map::map_camera_t& cameras,
                           cam_param_map_t const& ext_params,
                           std::vector<std::vector<double> > const& int_params,
                           cam_intrinsic_id_map_t const& int_map) const;

  /// Add the camera position priors costs to the Ceres problem
  int
  add_position_prior_cost(::ceres::Problem& problem,
                          cam_param_map_t& ext_params,
                          vital::sfm_constraints_sptr constraints);

  /// Add the camera intrinsic priors costs to the Ceres problem
  void add_intrinsic_priors_cost(
    ::ceres::Problem& problem,
    std::vector<std::vector<double> >& int_params) const;

  /// Add the camera path smoothness costs to the Ceres problem
  void add_camera_path_smoothness_cost(
    ::ceres::Problem& problem,
    frame_params_t const& ordered_params) const;

  /// Add the camera forward motion damping costs to the Ceres problem
  void add_forward_motion_damping_cost(
    ::ceres::Problem& problem,
    frame_params_t const& ordered_params,
    cam_intrinsic_id_map_t const& frame_to_intr_map) const;

  /// enumerate the intrinsics held constant
  /**
   * Based on the settings of the boolean optimization switches
   * poplulate a vector of indices marking which intrinsic camera
   * paramaters are held constant.  Indices are:
   *   - \b 0 : focal length
   *   - \b 1 : principal point X
   *   - \b 2 : principal point Y
   *   - \b 3 : aspect ratio
   *   - \b 4 : skew
   *   - \b 5 : radial distortion (k1)
   *   - \b 6 : radial distortion (k2)
   *   - \b 7 : tangential distortion (p1)
   *   - \b 8 : tangential distortion (p2)
   *   - \b 9 : radial distortion (k3)
   *   - \b 10 : radial distortion (k4)
   *   - \b 11 : radial distortion (k5)
   *   - \b 12 : radial distortion (k6)
   */
  std::vector<int> enumerate_constant_intrinsics() const;

  /// option to optimize the focal length
  bool optimize_focal_length;
  /// option to optimize aspect ratio
  bool optimize_aspect_ratio;
  /// option to optimize principal point
  bool optimize_principal_point;
  /// option to optimize skew
  bool optimize_skew;
  /// the lens distortion model to use
  LensDistortionType lens_distortion_type;
  /// option to optimize radial distortion parameter k1
  bool optimize_dist_k1;
  /// option to optimize radial distortion parameter k2
  bool optimize_dist_k2;
  /// option to optimize radial distortion parameter k3
  bool optimize_dist_k3;
  /// option to optimize tangential distortions parameters p1, p2
  bool optimize_dist_p1_p2;
  /// option to optimize radial distortion parameters k4, k5, k6
  bool optimize_dist_k4_k5_k6;
  /// the type of sharing of intrinsics between cameras to use
  CameraIntrinsicShareType camera_intrinsic_share_type;
  /// the amount of the camera path smoothness regularization
  double camera_path_smoothness;
  /// the scale of camera forward motion damping regularization
  double camera_forward_motion_damping;
  /// a soft lower bound on the horizontal field of view
  double minimum_hfov;
};


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver

#endif
