/*ckwg +29
 * Copyright 2016 by Kitware, Inc.
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
 * \brief Implementation of Ceres options helper classes
 */

#include "options.h"

#include <arrows/ceres/camera_smoothness.h>
#include <arrows/ceres/camera_position.h>
#include <arrows/ceres/camera_intrinsic_prior.h>

#include <vital/math_constants.h>

using namespace kwiver::vital;


namespace kwiver {
namespace arrows {
namespace ceres {

/// Constructor
solver_options
::solver_options()
  : options()
{
}

/// Copy Constructor
solver_options
::solver_options(const solver_options& other)
  : options( other.options )
{
}


/// populate the config block with options
void
solver_options
::get_configuration(config_block_sptr config) const
{
  ::ceres::Solver::Options const& o = this->options;
  config->set_value("num_threads", o.num_threads,
                    "Number of threads to use");
  config->set_value("num_linear_solver_threads", o.num_linear_solver_threads,
                    "Number of threads to use in the linear solver");
  config->set_value("max_num_iterations", o.max_num_iterations,
                    "Maximum number of iteration of allow");
  config->set_value("function_tolerance", o.function_tolerance,
                    "Solver terminates if relative cost change is below this "
                    "tolerance");
  config->set_value("gradient_tolerance", o.gradient_tolerance,
                    "Solver terminates if the maximum gradient is below this "
                    "tolerance");
  config->set_value("parameter_tolerance", o.parameter_tolerance,
                    "Solver terminates if the relative change in parameters "
                    "is below this tolerance");
  config->set_value("linear_solver_type", o.linear_solver_type,
                    "Linear solver to use."
                    + ceres_options< ::ceres::LinearSolverType >());
  config->set_value("preconditioner_type", o.preconditioner_type,
                    "Preconditioner to use."
                    + ceres_options< ::ceres::PreconditionerType >());
  config->set_value("trust_region_strategy_type", o.trust_region_strategy_type,
                    "Trust region step compution algorithm used by Ceres."
                    + ceres_options< ::ceres::TrustRegionStrategyType >());
  config->set_value("dogleg_type", o.dogleg_type,
                    "Dogleg strategy to use."
                    + ceres_options< ::ceres::DoglegType >());
  config->set_value("update_state_every_iteration", o.update_state_every_iteration,
                    "If true, the updated state is computed and provided in "
                    "the callback on every iteration.  This slows down "
                    "optimization but can be useful for debugging.");
}


/// set the member variables from the config block
void
solver_options
::set_configuration(config_block_sptr config)
{
#define GET_VALUE(vtype, vname) \
  options.vname = config->get_value< vtype >(#vname, options.vname);

  GET_VALUE(int,    num_threads);
  GET_VALUE(int,    num_linear_solver_threads);
  GET_VALUE(int,    max_num_iterations);
  GET_VALUE(double, function_tolerance);
  GET_VALUE(double, gradient_tolerance);
  GET_VALUE(double, parameter_tolerance);
  GET_VALUE(::ceres::LinearSolverType, linear_solver_type);
  GET_VALUE(::ceres::PreconditionerType, preconditioner_type);
  GET_VALUE(::ceres::TrustRegionStrategyType, trust_region_strategy_type);
  GET_VALUE(::ceres::DoglegType, dogleg_type);
  GET_VALUE(bool, update_state_every_iteration);

#undef GET_VALUE
}



/// Constructor
camera_options
::camera_options()
  : optimize_focal_length(true),
    optimize_aspect_ratio(false),
    optimize_principal_point(false),
    optimize_skew(false),
    lens_distortion_type(NO_DISTORTION),
    optimize_dist_k1(true),
    optimize_dist_k2(false),
    optimize_dist_k3(false),
    optimize_dist_p1_p2(false),
    optimize_dist_k4_k5_k6(false),
    camera_intrinsic_share_type(AUTO_SHARE_INTRINSICS),
    camera_path_smoothness(0.0),
    camera_forward_motion_damping(0.0),
    minimum_hfov(0.0)
{
}

/// Copy Constructor
camera_options
::camera_options(const camera_options& other)
  : optimize_focal_length(other.optimize_focal_length),
    optimize_aspect_ratio(other.optimize_aspect_ratio),
    optimize_principal_point(other.optimize_principal_point),
    optimize_skew(other.optimize_skew),
    lens_distortion_type(other.lens_distortion_type),
    optimize_dist_k1(other.optimize_dist_k1),
    optimize_dist_k2(other.optimize_dist_k2),
    optimize_dist_k3(other.optimize_dist_k3),
    optimize_dist_p1_p2(other.optimize_dist_p1_p2),
    optimize_dist_k4_k5_k6(other.optimize_dist_k4_k5_k6),
    camera_intrinsic_share_type(other.camera_intrinsic_share_type),
    camera_path_smoothness(other.camera_path_smoothness),
    camera_forward_motion_damping(other.camera_forward_motion_damping),
    minimum_hfov(other.minimum_hfov)
{
}


/// populate the config block with options
void
camera_options
::get_configuration(config_block_sptr config) const
{
  config->set_value("optimize_focal_length", this->optimize_focal_length,
                    "Include focal length parameters in bundle adjustment.");
  config->set_value("optimize_aspect_ratio", this->optimize_aspect_ratio,
                    "Include aspect ratio parameters in bundle adjustment.");
  config->set_value("optimize_principal_point", this->optimize_principal_point,
                    "Include principal point parameters in bundle adjustment.");
  config->set_value("optimize_skew", this->optimize_skew,
                    "Include skew parameters in bundle adjustment.");
  config->set_value("lens_distortion_type", this->lens_distortion_type,
                    "Lens distortion model to use."
                    + ceres_options< ceres::LensDistortionType >());
  config->set_value("optimize_dist_k1", this->optimize_dist_k1,
                    "Include radial lens distortion parameter k1 in "
                    "bundle adjustment.");
  config->set_value("optimize_dist_k2", this->optimize_dist_k2,
                    "Include radial lens distortion parameter k2 in "
                    "bundle adjustment.");
  config->set_value("optimize_dist_k3", this->optimize_dist_k3,
                    "Include radial lens distortion parameter k3 in "
                    "bundle adjustment.");
  config->set_value("optimize_dist_p1_p2", this->optimize_dist_p1_p2,
                    "Include tangential lens distortion parameters "
                    "p1 and p2 in bundle adjustment.");
  config->set_value("optimize_dist_k4_k5_k6", this->optimize_dist_k4_k5_k6,
                    "Include radial lens distortion parameters "
                    "k4, k5, and k6 in bundle adjustment.");
  config->set_value("camera_intrinsic_share_type", this->camera_intrinsic_share_type,
                    "Determines how to share intrinsics across cameras.\n"
                    "AUTO shares intrinsics between cameras with a common camera_intrinsic_sptr\n"
                    "COMMON enforces that all cameras share common intrinsics\n"
                    "UNIQUE enforces that each camera has its own intrinsics parameters."
                    + ceres_options< ceres::CameraIntrinsicShareType >());
  config->set_value("camera_path_smoothness", this->camera_path_smoothness,
                    "Controls the amount a regularization to apply to the camera path. "
                    "If set to zero the path regularization is disabled.");
  config->set_value("camera_forward_motion_damping", this->camera_forward_motion_damping,
                    "Controls the amount a regularization to apply to limit camera "
                    "forward motion.  This option is useful for zoom lenses at long "
                    "distances.  It causes the algorithm to prefer focal length change "
                    "over fast motion along the principal ray. "
                    "If set to zero the regularization is disabled.");
  config->set_value("minimum_hfov", this->minimum_hfov,
                    "A soft lower bound on the minimum horizontal field of "
                    "view in degrees. This generates a soft upper bound on "
                    "focal length if set greater than zero. If the focal "
                    "length exceeds this limit it will incur a quadratic "
                    "penalty.");
}


/// set the member variables from the config block
void
camera_options
::set_configuration(config_block_sptr config)
{
#define GET_VALUE(vtype, vname) \
  this->vname = config->get_value< vtype >(#vname, this->vname);

  GET_VALUE(bool, optimize_focal_length);
  GET_VALUE(bool, optimize_aspect_ratio);
  GET_VALUE(bool, optimize_principal_point);
  GET_VALUE(bool, optimize_skew);
  GET_VALUE(bool, optimize_dist_k1);
  GET_VALUE(bool, optimize_dist_k2);
  GET_VALUE(bool, optimize_dist_k3);
  GET_VALUE(bool, optimize_dist_p1_p2);
  GET_VALUE(bool, optimize_dist_k4_k5_k6);
  GET_VALUE(ceres::LensDistortionType, lens_distortion_type);
  GET_VALUE(ceres::CameraIntrinsicShareType, camera_intrinsic_share_type);
  GET_VALUE(double, camera_path_smoothness);
  GET_VALUE(double, camera_forward_motion_damping);
  GET_VALUE(double, minimum_hfov);
#undef GET_VALUE
}


/// Return true if any options to optimize intrinsic parameters are set
bool
camera_options
::optimize_intrinsics() const
{
  if( optimize_focal_length ||
      optimize_aspect_ratio ||
      optimize_principal_point ||
      optimize_skew )
  {
    return true;
  }
  switch( lens_distortion_type )
  {
    case POLYNOMIAL_RADIAL_DISTORTION:
      return optimize_dist_k1
          || optimize_dist_k2
          || optimize_dist_k3;
    case POLYNOMIAL_RADIAL_TANGENTIAL_DISTORTION:
      return optimize_dist_k1
          || optimize_dist_k2
          || optimize_dist_k3
          || optimize_dist_p1_p2;
    case RATIONAL_RADIAL_TANGENTIAL_DISTORTION:
      return optimize_dist_k1
          || optimize_dist_k2
          || optimize_dist_k3
          || optimize_dist_p1_p2
          || optimize_dist_k4_k5_k6;
    default:
      break;
  }
  return false;
}


/// enumerate the intrinsics held constant
std::vector<int>
camera_options
::enumerate_constant_intrinsics() const
{
  std::vector<int> constant_intrinsics;

  // number of lens distortion parameters in the selected model
  const unsigned int ndp = num_distortion_params(this->lens_distortion_type);

  if (!this->optimize_focal_length)
  {
    constant_intrinsics.push_back(0);
  }
  if (!this->optimize_principal_point)
  {
    constant_intrinsics.push_back(1);
    constant_intrinsics.push_back(2);
  }
  if (!this->optimize_aspect_ratio)
  {
    constant_intrinsics.push_back(3);
  }
  if (!this->optimize_skew)
  {
    constant_intrinsics.push_back(4);
  }
  if (!this->optimize_dist_k1 && ndp > 0)
  {
    constant_intrinsics.push_back(5);
  }
  if (!this->optimize_dist_k2 && ndp > 1)
  {
    constant_intrinsics.push_back(6);
  }
  if (!this->optimize_dist_p1_p2 && ndp > 3)
  {
    constant_intrinsics.push_back(7);
    constant_intrinsics.push_back(8);
  }
  if (!this->optimize_dist_k3 && ndp > 4)
  {
    constant_intrinsics.push_back(9);
  }
  if (!this->optimize_dist_k4_k5_k6 && ndp > 7)
  {
    constant_intrinsics.push_back(10);
    constant_intrinsics.push_back(11);
    constant_intrinsics.push_back(12);
  }
  return constant_intrinsics;
}


/// extract the extrinsic paramters from a camera into the parameter array
void
camera_options
::extract_camera_extrinsics(const vital::camera_perspective_sptr camera,
                            double* params) const
{
  vector_3d rot = camera->rotation().rodrigues();
  vector_3d center = camera->center();
  std::copy(rot.data(), rot.data()+3, params);
  std::copy(center.data(), center.data()+3, params+3);
}


/// Update a camera object to use extrinsic parameters from an array
void
camera_options
::update_camera_extrinsics(std::shared_ptr<vital::simple_camera_perspective> camera,
                           double const* params) const
{
  camera->set_rotation(rotation_d(vector_3d(Eigen::Map<const vector_3d>(params))));
  camera->set_center(Eigen::Map<const vector_3d>(&params[3]));
}


/// extract the paramters from camera intrinsics into the parameter array
void
camera_options
::extract_camera_intrinsics(const vital::camera_intrinsics_sptr K,
                            double* params) const
{
  params[0] = K->focal_length();
  params[1] = K->principal_point().x();
  params[2] = K->principal_point().y();
  params[3] = K->aspect_ratio();
  params[4] = K->skew();
  const std::vector<double> d = K->dist_coeffs();
  // copy the intersection of parameters provided in K
  // and those that are supported by the requested model type
  const unsigned int num_dp = std::min(num_distortion_params(this->lens_distortion_type),
                                       static_cast<unsigned int>(d.size()));
  if( num_dp > 0 )
  {
    std::copy(d.begin(), d.begin()+num_dp, &params[5]);
  }
}


/// update the camera intrinsics from a parameter array
void
camera_options
::update_camera_intrinsics(std::shared_ptr<vital::simple_camera_intrinsics> K,
                           const double* params) const
{
  K->set_focal_length(params[0]);
  vector_2d pp((Eigen::Map<const vector_2d>(&params[1])));
  K->set_principal_point(pp);
  K->set_aspect_ratio(params[3]);
  K->set_skew(params[4]);

  // distortion parameters
  const unsigned int ndp = num_distortion_params(this->lens_distortion_type);
  if( ndp > 0 )
  {
    Eigen::VectorXd dc(ndp);
    std::copy(&params[5], &params[5]+ndp, dc.data());
    K->set_dist_coeffs(dc);
  }
}


/// extract the set of all unique intrinsic and extrinsic parameters from a camera map
void
camera_options
::extract_camera_parameters(camera_map::map_camera_t const& cameras,
                            cam_param_map_t& ext_params,
                            std::vector<std::vector<double> >& int_params,
                            cam_intrinsic_id_map_t& int_map) const
{
  // We need another map from intrinsics intstances to parameter index to
  // detect when the same intrinsics are shared between cameras
  std::map<camera_intrinsics_sptr, unsigned int> camera_intr_map;

  // number of lens distortion parameters in the selected model
  const unsigned int ndp = num_distortion_params(this->lens_distortion_type);
  std::vector<double> intrinsic_params(5 + ndp, 0.0);
  for(const camera_map::map_camera_t::value_type& c : cameras)
  {
    std::vector<double> params(6);
    auto cam_ptr = std::dynamic_pointer_cast<camera_perspective>(c.second);
    this->extract_camera_extrinsics(cam_ptr, &params[0]);
    camera_intrinsics_sptr K = cam_ptr->intrinsics();
    ext_params[c.first] = params;

    // add a new set of intrisic parameter if one of the following:
    // - we are forcing unique intrinsics for each camera
    // - we are forcing common intrinsics and this is the first frame
    // - we are auto detecting shared intrinisics and this is a new sptr
    if( this->camera_intrinsic_share_type == FORCE_UNIQUE_INTRINSICS
        || ( this->camera_intrinsic_share_type == FORCE_COMMON_INTRINSICS
             && int_params.empty() )
        || camera_intr_map.count(K) == 0 )
    {
      this->extract_camera_intrinsics(K, &intrinsic_params[0]);
      // update the maps with the index of this new parameter vector
      camera_intr_map[K] = static_cast<unsigned int>(int_params.size());
      int_map[c.first] = static_cast<unsigned int>(int_params.size());
      // add the parameter vector
      int_params.push_back(intrinsic_params);
    }
    else if( this->camera_intrinsic_share_type == FORCE_COMMON_INTRINSICS )
    {
      // map to the first parameter vector
      int_map[c.first] = 0;
    }
    else
    {
      // map to a previously seen parameter vector
      int_map[c.first] = camera_intr_map[K];
    }
  }
}


/// update the camera objects using the extracted camera parameters
void
camera_options
::update_camera_parameters(vital::camera_map::map_camera_t& cameras,
                           cam_param_map_t const& ext_params,
                           std::vector<std::vector<double> > const& int_params,
                           cam_intrinsic_id_map_t const& int_map) const
{
  std::vector<camera_intrinsics_sptr> updated_intr;
  if( this->optimize_intrinsics() )
  {
    // Update the camera intrinics with optimized values
    for(const std::vector<double>& cip : int_params)
    {
      auto K = std::make_shared<simple_camera_intrinsics>();
      this->update_camera_intrinsics(K, &cip[0]);
      updated_intr.push_back(camera_intrinsics_sptr(K));
    }
  }

  // Update the cameras with the optimized values
  typedef std::map<frame_id_t, std::vector<double> > cam_param_map_t;
  for(const cam_param_map_t::value_type& cp : ext_params)
  {
    auto orig_cam = std::dynamic_pointer_cast<camera_perspective>(cameras[cp.first]);
    auto simp_cam = std::dynamic_pointer_cast<simple_camera_perspective>(orig_cam);
    if( !simp_cam )
    {
      simp_cam = std::make_shared<simple_camera_perspective>();
      cameras[cp.first] = simp_cam;
    }
    camera_intrinsics_sptr K = orig_cam->intrinsics();
    if( this->optimize_intrinsics() )
    {
      // look-up updated intrinsics
      auto map_itr = int_map.find(cp.first);
      unsigned int intr_idx = map_itr->second;
      K = updated_intr[intr_idx];
    }
    this->update_camera_extrinsics(simp_cam, &cp.second[0]);
    simp_cam->set_intrinsics(K);
  }
}

int
camera_options
::add_position_prior_cost(
  ::ceres::Problem& problem,
  cam_param_map_t& ext_params,
  vital::sfm_constraints_sptr constraints)
{
  int num_priors_applied = 0;
  if (!constraints)
  {
    return num_priors_applied;
  }

  ::ceres::LossFunction *loss = LossFunctionFactory(SOFT_L_ONE_LOSS, 100.0);

  for (auto& ext_par : ext_params)
  {
    auto fid = ext_par.first;
    vector_3d position_prior_local;
    if(!constraints->get_camera_position_prior_local(fid,position_prior_local))
    {
      continue;
    }

    double *cam_state = &ext_par.second[0];
    if (!problem.HasParameterBlock(cam_state))
    {
      continue;
    }
    if (problem.IsParameterBlockConstant(cam_state))
    {
        continue;
    }

    auto position_prior_cost =
      camera_position::create(position_prior_local);

    problem.AddResidualBlock(position_prior_cost,
      loss,
      cam_state);
    ++num_priors_applied;
  }
  return num_priors_applied;
}

/// Add the camera intrinsic priors costs to the Ceres problem
void
camera_options
::add_intrinsic_priors_cost(
  ::ceres::Problem& problem,
  std::vector<std::vector<double> >& int_params) const
{
  if (this->minimum_hfov <= 0.0)
  {
    return;
  }
  // scaling balances the response based on the number of other residuals.
  double scale = static_cast<double>(std::max(1, problem.NumResiduals()));
  auto scaled_loss = new ::ceres::ScaledLoss(NULL, scale,
                           ::ceres::Ownership::TAKE_OWNERSHIP);
  for (auto& int_par : int_params)
  {
    // assume image width is twice the principal point X coordinate
    double width = 2.0 * int_par[1];
    if (width <= 0.0)
    {
      width = 1280.0;
    }
    double max_focal_len = width / (2.0 * std::tan(this->minimum_hfov * kwiver::vital::deg_to_rad / 2.0));
    auto cam_intrin_prior_cost =
      camera_intrinsic_prior::create(max_focal_len, int_par.size());

    double* foc_len = &int_par[0];
    // add the loss with squared error
    problem.AddResidualBlock(cam_intrin_prior_cost,
                             scaled_loss,
                             foc_len);
  }
}

/// Add the camera path smoothness costs to the Ceres problem
void
camera_options
::add_camera_path_smoothness_cost(
    ::ceres::Problem& problem,
    frame_params_t const& ordered_params) const
{
  // Add camera path regularization residuals
  if( this->camera_path_smoothness > 0.0 &&
      ordered_params.size() >= 3 )
  {
    double sum_dist = 0.0;
    auto prev_cam = ordered_params.begin();
    auto curr_cam = prev_cam;
    curr_cam++;
    auto next_cam = curr_cam;
    next_cam++;
    std::vector<std::tuple<decltype(curr_cam), double, double> > constraints;
    for(; next_cam != ordered_params.end();
        prev_cam = curr_cam, curr_cam = next_cam, next_cam++)
    {
      double inv_dist = 1.0 / static_cast<double>(next_cam->first -
                                                  prev_cam->first);
      double frac = (curr_cam->first - prev_cam->first) * inv_dist;
      if (inv_dist > 0.0 && frac > 0.0 && frac < 1.0 )
      {
        sum_dist += (Eigen::Map<vector_3d>(prev_cam->second+3) -
                     Eigen::Map<vector_3d>(next_cam->second+3)).norm();
        constraints.push_back(std::make_tuple(curr_cam, inv_dist, frac));
      }
    }
    // Normalize the weight
    const double weight = this->camera_path_smoothness
                        * problem.NumResiduals();
    const double scale = constraints.size() / sum_dist;
    auto scaled_loss = new ::ceres::ScaledLoss(NULL, weight,
                           ::ceres::Ownership::TAKE_OWNERSHIP);
    for (auto const& t : constraints)
    {
      auto cam_itr = std::get<0>(t);
      const double s = scale * std::get<1>(t);
      ::ceres::CostFunction* smoothness_cost  =
        camera_position_smoothness::create(s, std::get<2>(t));
      problem.AddResidualBlock(smoothness_cost,
                               scaled_loss,
                               (cam_itr-1)->second,
                               cam_itr->second,
                               (cam_itr+1)->second);
    }
  }
}


/// Add the camera forward motion damping costs to the Ceres problem
void
camera_options
::add_forward_motion_damping_cost(
    ::ceres::Problem& problem,
    frame_params_t const& ordered_params,
    cam_intrinsic_id_map_t const& frame_to_intr_map) const
{
  if( this->camera_forward_motion_damping > 0.0 &&
      ordered_params.size() >= 2 )
  {
    double sum_dist = 0.0;
    auto prev_cam = ordered_params.begin();
    auto curr_cam = prev_cam;
    curr_cam++;
    std::vector<decltype(curr_cam) > constraints;
    for (; curr_cam != ordered_params.end();
           prev_cam = curr_cam, curr_cam++)
    {
      // add a forward motion residual only when the camera intrinsic models
      // are not the same instance
      auto prev_idx = frame_to_intr_map.find(prev_cam->first);
      auto curr_idx = frame_to_intr_map.find(curr_cam->first);
      if (prev_idx != frame_to_intr_map.end() &&
          curr_idx != frame_to_intr_map.end() &&
          prev_idx->second != curr_idx->second)
      {
        sum_dist += (Eigen::Map<vector_3d>(prev_cam->second + 3) -
                     Eigen::Map<vector_3d>(curr_cam->second + 3)).norm();
        constraints.push_back(curr_cam);
      }
    }
    // Normalize the weight
    const double weight = this->camera_forward_motion_damping
                        * problem.NumResiduals();
    const double scale = constraints.size() / sum_dist;
    auto scaled_loss = new ::ceres::ScaledLoss(NULL, weight,
                           ::ceres::Ownership::TAKE_OWNERSHIP);
    for (auto curr_cam : constraints)
    {
      auto prev_cam = curr_cam - 1;
      double inv_dist = 1.0 / static_cast<double>(curr_cam->first -
                                                  prev_cam->first);
      ::ceres::CostFunction* fwd_mo_cost =
        camera_limit_forward_motion::create(scale * inv_dist);
      problem.AddResidualBlock(fwd_mo_cost,
                               scaled_loss,
                               prev_cam->second,
                               curr_cam->second);
    }
  }
}

} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
