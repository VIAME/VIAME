/*ckwg +29
 * Copyright 2015-2019 by Kitware, Inc.
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
 * \brief Implementation of Ceres bundle adjustment algorithm
 */

#include "bundle_adjust.h"

#include <iostream>
#include <set>
#include <unordered_set>

#include <vital/io/eigen_io.h>
#include <arrows/ceres/reprojection_error.h>
#include <arrows/ceres/types.h>
#include <arrows/ceres/options.h>
#include <ceres/loss_function.h>

#include <ceres/ceres.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ceres {


// ============================================================================
// A class to register callbacks with Ceres
class StateCallback
  : public ::ceres::IterationCallback
{
public:
  explicit StateCallback(bundle_adjust* b = NULL)
    : bap(b) {}

  ::ceres::CallbackReturnType operator() (const ::ceres::IterationSummary& summary)
  {
    return ( bap && !bap->trigger_callback() )
           ? ::ceres::SOLVER_TERMINATE_SUCCESSFULLY
           : ::ceres::SOLVER_CONTINUE;
  }

  bundle_adjust* bap;
};


// ============================================================================
// Private implementation class
class bundle_adjust::priv
  : public solver_options,
    public camera_options
{
public:
  // Constructor
  priv()
  : solver_options(),
    camera_options(),
    verbose(false),
    loss_function_type(TRIVIAL_LOSS),
    loss_function_scale(1.0),
    ceres_callback()
  {
  }

  // verbose output
  bool verbose;
  // the robust loss function type to use
  LossFunctionType loss_function_type;
  // the scale of the loss function
  double loss_function_scale;


  // the input cameras to update in place
  camera_map::map_camera_t cams;
  // the input landmarks to update in place
  landmark_map::map_landmark_t lms;
  // a map from track id to landmark parameters
  std::unordered_map<track_id_t, std::vector<double> > landmark_params;
  // a map from frame number to extrinsic parameters
  std::unordered_map<frame_id_t, std::vector<double> > camera_params;
  // vector of unique camera intrinsic parameters
  std::vector<std::vector<double> > camera_intr_params;
  // a map from frame number to index of unique camera intrinsics in camera_intr_params
  std::unordered_map<frame_id_t, unsigned int> frame_to_intr_map;
  // the ceres callback class
  StateCallback ceres_callback;
};


// ----------------------------------------------------------------------------
// Constructor
bundle_adjust
::bundle_adjust()
: d_(new priv)
{
  attach_logger( "arrows.ceres.bundle_adjust" );
  d_->ceres_callback.bap = this;
}


// Destructor
bundle_adjust
::~bundle_adjust()
{
}


// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
config_block_sptr
bundle_adjust
::get_configuration() const
{
  // get base config from base class
  config_block_sptr config = vital::algo::bundle_adjust::get_configuration();
  config->set_value("verbose", d_->verbose,
                    "If true, write status messages to the terminal showing "
                    "optimization progress at each iteration");
  config->set_value("loss_function_type", d_->loss_function_type,
                    "Robust loss function type to use."
                    + ceres_options< ceres::LossFunctionType >());
  config->set_value("loss_function_scale", d_->loss_function_scale,
                    "Robust loss function scale factor.");

  // get the solver options
  d_->solver_options::get_configuration(config);

  // get the camera configuation options
  d_->camera_options::get_configuration(config);

  return config;
}


// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
bundle_adjust
::set_configuration(config_block_sptr in_config)
{
  ::ceres::Solver::Options& o = d_->options;
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  d_->verbose = config->get_value<bool>("verbose",
                                        d_->verbose);
  o.minimizer_progress_to_stdout = d_->verbose;
  o.logging_type = d_->verbose ? ::ceres::PER_MINIMIZER_ITERATION
                               : ::ceres::SILENT;
  typedef ceres::LossFunctionType clf_t;
  d_->loss_function_type = config->get_value<clf_t>("loss_function_type",
                                                    d_->loss_function_type);
  d_->loss_function_scale = config->get_value<double>("loss_function_scale",
                                                      d_->loss_function_scale);

  // set the camera configuation options
  d_->solver_options::set_configuration(config);

  // set the camera configuation options
  d_->camera_options::set_configuration(config);

  if(this->m_callback)
  {
    o.callbacks.clear();
    o.callbacks.push_back(&d_->ceres_callback);
    o.update_state_every_iteration = true;
  }
  else
  {
    o.update_state_every_iteration = false;
  }
}


// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
bundle_adjust
::check_configuration(config_block_sptr config) const
{
  std::string msg;
  if( !d_->options.IsValid(&msg) )
  {
    LOG_ERROR( logger(), msg);
    return false;
  }
  return true;
}

class distance_constraint
{
public:
  /// Constructor
  distance_constraint(const double distance_squared)
    : distance_squared_(distance_squared) {}

  template <typename T> bool operator()(
    const T* const pose_0,
    const T* const pose_1,
    T* residuals) const
  {
    const T* center_0 = pose_0 + 3;
    const T* center_1 = pose_1 + 3;
    T dx = center_0[0] - center_1[0];
    T dy = center_0[1] - center_1[1];
    T dz = center_0[2] - center_1[2];

    T dist = dx*dx + dy*dy + dz*dz;

    residuals[0] = (dist - distance_squared_);

    return true;
  }

  /// Cost function factory
  static ::ceres::CostFunction* create(const double distance)
  {
    typedef distance_constraint Self;
    return new ::ceres::AutoDiffCostFunction<Self, 1, 6, 6>(new Self(distance));
  }

  double distance_squared_;
};

/// Optimize the camera and landmark parameters given a set of tracks
void
bundle_adjust
::optimize(camera_map_sptr& cameras,
  landmark_map_sptr& landmarks,
  feature_track_set_sptr tracks,
  sfm_constraints_sptr constraints) const
{
  simple_camera_perspective_map cams;
  for (auto p : cameras->cameras())
  {
    auto c = std::dynamic_pointer_cast<simple_camera_perspective>(p.second);
    if (c)
    {
      cams.insert(p.first, c);
    }
  }
  auto lms = landmarks->landmarks();
  this->optimize(cams, lms, tracks, {}, {}, constraints);
  landmarks = std::make_shared<simple_landmark_map>(lms);
  cameras = std::make_shared<simple_camera_perspective_map>(cams);
}


// ----------------------------------------------------------------------------
// Optimize the camera and landmark parameters given a set of tracks
void
bundle_adjust
::optimize(kwiver::vital::simple_camera_perspective_map &cameras,
           kwiver::vital::landmark_map::map_landmark_t &landmarks,
           vital::feature_track_set_sptr tracks,
           const std::set<vital::frame_id_t>& to_fix_cameras_in,
           const std::set<vital::landmark_id_t>& to_fix_landmarks_in,
           kwiver::vital::sfm_constraints_sptr constraints) const
{
  if(!tracks )
  {
    // TODO throw an exception for missing input data
    return;
  }

  std::unordered_set<vital::frame_id_t> to_fix_cameras;
  for (auto &fid : to_fix_cameras_in)
  {
    to_fix_cameras.insert(fid);
  }

  std::unordered_set<vital::landmark_id_t> to_fix_landmarks;
  for (auto &lid : to_fix_landmarks_in)
  {
    to_fix_landmarks.insert(lid);
  }

  std::set<frame_id_t> fixed_cameras;

  // extract data from containers
  d_->cams = cameras.cameras();
  d_->lms = landmarks;

  // Extract the landmark locations into a mutable map
  d_->landmark_params.clear();
  for(const landmark_map::map_landmark_t::value_type& lm : d_->lms)
  {
    landmark_id_t lm_id = lm.first;

    if (d_->landmark_params.find(lm_id) == d_->landmark_params.end())
    {
      vector_3d loc = lm.second->loc();
      d_->landmark_params[lm_id] = std::vector<double>(loc.data(), loc.data() + 3);
    }
  }

  typedef std::unordered_map<track_id_t, std::vector<double> > lm_param_map_t;

  d_->camera_params.clear();
  d_->camera_intr_params.clear();
  d_->frame_to_intr_map.clear();

  // Extract the raw camera parameter into the provided maps
  d_->extract_camera_parameters(d_->cams,
                                d_->camera_params,
                                d_->camera_intr_params,
                                d_->frame_to_intr_map);

  // the Ceres solver problem
  ::ceres::Problem problem;

  // enumerate the intrinsics held constant
  std::vector<int> constant_intrinsics = d_->enumerate_constant_intrinsics();

  // Create the loss function to use
  ::ceres::LossFunction* loss_func
      = LossFunctionFactory(d_->loss_function_type,
                            d_->loss_function_scale);
  bool loss_func_used = false;

  // Add the residuals for each relevant observation
  std::set<unsigned int> used_intrinsics;

  for (const auto & lm : d_->lms)
  {
    const auto lm_id = lm.first;
    bool lm_visible_in_variable_camera = false;
    //lowest index track is landmark id

    auto t = tracks->get_track(lm_id);
    if (!t)
    {
      continue;
    }

    auto lm_itr = d_->landmark_params.find(lm_id);
    // skip this track if the landmark is not in the set to optimize
    if (lm_itr == d_->landmark_params.end())
    {
      continue;
    }
    for (auto ts: *t)
    {
      if (to_fix_cameras.find(ts->frame()) == to_fix_cameras.end())
      {
        //this landmark is viewed in a variable camera.  So include it in the state to estimate.
        lm_visible_in_variable_camera = true;
        break;
      }
    }

    if (!lm_visible_in_variable_camera)
    {
      //this landmark is not visible in a variable camera, so no need to add measurements for it.
      continue;
    }

    bool fixed_landmark = to_fix_landmarks.find(lm_id) != to_fix_landmarks.end();

    for (auto ts : *t)
    {
      auto cam_itr = d_->camera_params.find(ts->frame());
      if (cam_itr == d_->camera_params.end())
      {
        continue;
      }

      bool fixed_camera = to_fix_cameras.find(cam_itr->first) != to_fix_cameras.end();

      if (fixed_landmark && fixed_camera)
      {
        //skip this measurement because it involves both a fixed camera and fixed landmark.
        continue;
      }

      auto fts = std::dynamic_pointer_cast<feature_track_state>(ts);
      if (!fts || !fts->feature)
      {
        continue;
      }
      if (!fts->inlier)
      {
        continue; // feature is not an inlier so don't use it in ba.
      }

      unsigned intr_idx = d_->frame_to_intr_map[fts->frame()];
      double * intr_params_ptr = &d_->camera_intr_params[intr_idx][0];
      used_intrinsics.insert(intr_idx);
      vector_2d pt = fts->feature->loc();
      problem.AddResidualBlock(create_cost_func(d_->lens_distortion_type,
        pt.x(), pt.y()),
        loss_func,
        intr_params_ptr,
        &cam_itr->second[0],
        &lm_itr->second[0]);

      loss_func_used = true;

    }
  }


  if (d_->camera_path_smoothness > 0.0 ||
      d_->camera_forward_motion_damping > 0.0)
  {
    // sort the camera parameters in order of frame number
    std::vector<std::pair<vital::frame_id_t, double *> > ordered_params;
    for (auto& item : d_->camera_params)
    {
      ordered_params.push_back(std::make_pair(item.first, &item.second[0]));
    }
    std::sort(ordered_params.begin(), ordered_params.end());

    // Add camera path regularization residuals
    d_->add_camera_path_smoothness_cost(problem, ordered_params);

    // Add forward motion regularization residuals
    d_->add_forward_motion_damping_cost(problem, ordered_params, d_->frame_to_intr_map);
  }


  //fix all the cameras in the to_fix_cameras list
  std::unordered_set<unsigned int> to_fix_intrinsics;
  for (auto tfc : to_fix_cameras)
  {
    auto cam_itr = d_->camera_params.find(tfc);
    if (cam_itr == d_->camera_params.end())
    {
      continue;
    }
    double *state_ptr = &cam_itr->second[0];
    if (problem.HasParameterBlock(state_ptr))
    {
      problem.SetParameterBlockConstant(state_ptr);
      fixed_cameras.insert(tfc);
    }
    // Mark the intrinsics for this camera fixed as well.
    // Only optimize intrinsics if no cameras using these
    // intrinsics are fixed
    auto const& intr_itr = d_->frame_to_intr_map.find(tfc);
    if (intr_itr != d_->frame_to_intr_map.end())
    {
      to_fix_intrinsics.insert(intr_itr->second);
    }
  }

  std::set<landmark_id_t> fixed_landmarks;
  //fix all the landmarks in the to_fix_landmarks list
  for (auto tfl: to_fix_landmarks)
  {
    auto lm_id = tfl;

    auto lm_itr = d_->landmark_params.find(lm_id);
    if (lm_itr == d_->landmark_params.end())
    {
      continue;
    }
    double *state_ptr = &lm_itr->second[0];
    if (problem.HasParameterBlock(state_ptr))
    {
      problem.SetParameterBlockConstant(state_ptr);
      fixed_landmarks.insert(tfl);
    }
  }

  // add costs for priors
  int num_position_priors_applied =
    d_->add_position_prior_cost(problem, d_->camera_params, constraints);

  d_->add_intrinsic_priors_cost(problem, d_->camera_intr_params);

  if (num_position_priors_applied < 3)
  {
    //gauge fixing code
    if (fixed_cameras.size() == 0 && fixed_landmarks.size() < 3)
    {
      //If no cameras are fixed, find the first camera and fix it.
      for (auto &fix : d_->camera_params)
      {
        auto fixed_fid = fix.first;
        auto state = &fix.second[0];
        if (problem.HasParameterBlock(state))
        {
          problem.SetParameterBlockConstant(state);
          fixed_cameras.insert(fixed_fid);
          break;
        }
      }
    }

    if (fixed_cameras.size() == 1 && fixed_landmarks.empty())
    {
      //add measurement between the one fixed camera and another arbitrary camera to fix the scale
      auto cam_itr_0 = d_->camera_params.find(*fixed_cameras.begin());
      //get another arbitrary camera
      bool scale_locking_camera_found = false;
      auto cam_itr_1 = d_->camera_params.begin();
      for (; cam_itr_1 != d_->camera_params.end(); ++cam_itr_1)
      {
        if (cam_itr_1->first != cam_itr_0->first && problem.HasParameterBlock(&cam_itr_1->second[0]))
        {
          scale_locking_camera_found = true;
          break;
        }
      }

      if (scale_locking_camera_found)
      {
        double *param0 = &cam_itr_0->second[0];
        double *param1 = &cam_itr_1->second[0];
        double distance_squared =
          (Eigen::Map<vector_3d>(param0 + 3) -
           Eigen::Map<vector_3d>(param1 + 3)).squaredNorm();
        double scale = problem.NumResiduals() / distance_squared;

        auto dist_loss =
          new ::ceres::ScaledLoss(NULL, scale,
                                  ::ceres::Ownership::TAKE_OWNERSHIP);
        problem.AddResidualBlock(distance_constraint::create(distance_squared),
                                 dist_loss, param0, param1);
      }
    }
  }

  const unsigned int ndp = num_distortion_params(d_->lens_distortion_type);
  for (const unsigned int idx : used_intrinsics)
  {
    std::vector<double>& cip = d_->camera_intr_params[idx];
    // apply the constraints
    if (constant_intrinsics.size() > 4 + ndp ||
        to_fix_intrinsics.count(idx) > 0)
    {
      // set all parameters in the block constant
      problem.SetParameterBlockConstant(&cip[0]);
    }
    else if (!constant_intrinsics.empty())
    {
      // set a subset of parameters in the block constant
      problem.SetParameterization(&cip[0],
        new ::ceres::SubsetParameterization(5 + ndp, constant_intrinsics));
    }
  }

  // If the loss function was added to a residual block, ownership was
  // transfered.  If not then we need to delete it.
  if(loss_func && !loss_func_used)
  {
    delete loss_func;
  }

  ::ceres::Solver::Summary summary;
  ::ceres::Solve(d_->options, &problem, &summary);
  if( d_->verbose )
  {
    LOG_DEBUG(logger(), "Ceres Full Report:\n" << summary.FullReport());
  }

  // Update the landmarks with the optimized values
  for(const lm_param_map_t::value_type& lmp : d_->landmark_params)
  {
    auto lmi = std::static_pointer_cast<landmark_d>(d_->lms[lmp.first]);
    lmi->set_loc(Eigen::Map<const vector_3d>(&lmp.second[0]));
  }

  // Update the cameras with the optimized values
  d_->update_camera_parameters(d_->cams, d_->camera_params,
                               d_->camera_intr_params, d_->frame_to_intr_map);
  cameras.set_from_base_camera_map(d_->cams);
}


// ----------------------------------------------------------------------------
// Set a callback function to report intermediate progress
void
bundle_adjust
::set_callback(callback_t cb)
{
  kwiver::vital::algo::bundle_adjust::set_callback(cb);
  ::ceres::Solver::Options& o = d_->options;
  o.callbacks.clear();
  if(this->m_callback)
  {
    o.callbacks.push_back(&d_->ceres_callback);
  }
}


// ----------------------------------------------------------------------------
// This function is called by a Ceres callback to trigger a kwiver callback
bool
bundle_adjust
::trigger_callback()
{
  if(this->m_callback)
  {
    if (!d_->options.update_state_every_iteration)
    {
      return this->m_callback(nullptr, nullptr, nullptr);
    }
    // Update the landmarks with the optimized values
    typedef std::map<track_id_t, std::vector<double> > lm_param_map_t;
    for(const lm_param_map_t::value_type& lmp : d_->landmark_params)
    {
      auto lmi = std::static_pointer_cast<landmark_d>(d_->lms[lmp.first]);
      lmi->set_loc(Eigen::Map<const vector_3d>(&lmp.second[0]));
    }
    landmark_map_sptr landmarks = std::make_shared<simple_landmark_map>(d_->lms);

    // Update the cameras with the optimized values
    d_->update_camera_parameters(d_->cams, d_->camera_params,
                                 d_->camera_intr_params, d_->frame_to_intr_map);
    camera_map_sptr cameras = std::make_shared<simple_camera_map>(d_->cams);

    return this->m_callback(cameras, landmarks,nullptr);
  }
  return true;
}


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
