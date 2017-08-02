/*ckwg +29
 * Copyright 2015-2017 by Kitware, Inc.
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

#include <vital/vital_foreach.h>

#include <vital/logger/logger.h>
#include <vital/io/eigen_io.h>
#include <arrows/ceres/reprojection_error.h>
#include <arrows/ceres/types.h>
#include <arrows/ceres/options.h>

#include <ceres/ceres.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ceres {


/// A class to register callbacks with Ceres
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


/// Private implementation class
class bundle_adjust::priv
  : public solver_options,
    public camera_options
{
public:
  /// Constructor
  priv()
  : solver_options(),
    camera_options(),
    verbose(false),
    loss_function_type(TRIVIAL_LOSS),
    loss_function_scale(1.0),
    ceres_callback(),
    m_logger( vital::get_logger( "arrows.ceres.bundle_adjust" ))
  {
  }

  /// verbose output
  bool verbose;
  /// the robust loss function type to use
  LossFunctionType loss_function_type;
  /// the scale of the loss function
  double loss_function_scale;


  /// the input cameras to update in place
  camera_map::map_camera_t cams;
  /// the input landmarks to update in place
  landmark_map::map_landmark_t lms;
  /// a map from track id to landmark parameters
  std::map<track_id_t, std::vector<double> > landmark_params;
  /// a map from frame number to extrinsic parameters
  std::map<frame_id_t, std::vector<double> > camera_params;
  /// vector of unique camera intrinsic parameters
  std::vector<std::vector<double> > camera_intr_params;
  /// a map from frame number to index of unique camera intrinsics in camera_intr_params
  std::map<frame_id_t, unsigned int> frame_to_intr_map;
  /// the ceres callback class
  StateCallback ceres_callback;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
bundle_adjust
::bundle_adjust()
: d_(new priv)
{
  d_->ceres_callback.bap = this;
}


/// Destructor
bundle_adjust
::~bundle_adjust()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
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


/// Set this algorithm's properties via a config block
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


/// Check that the algorithm's currently configuration is valid
bool
bundle_adjust
::check_configuration(config_block_sptr config) const
{
  std::string msg;
  if( !d_->options.IsValid(&msg) )
  {
    LOG_ERROR( d_->m_logger, msg);
    return false;
  }
  return true;
}


/// Optimize the camera and landmark parameters given a set of tracks
void
bundle_adjust
::optimize(camera_map_sptr& cameras,
           landmark_map_sptr& landmarks,
           feature_track_set_sptr tracks,
           video_metadata_map_sptr metadata) const
{
  if( !cameras || !landmarks || !tracks )
  {
    // TODO throw an exception for missing input data
    return;
  }

  // extract data from containers
  d_->cams = cameras->cameras();
  d_->lms = landmarks->landmarks();
  std::vector<track_sptr> trks = tracks->tracks();

  // Extract the landmark locations into a mutable map
  d_->landmark_params.clear();
  VITAL_FOREACH(const landmark_map::map_landmark_t::value_type& lm, d_->lms)
  {
    vector_3d loc = lm.second->loc();
    d_->landmark_params[lm.first] = std::vector<double>(loc.data(), loc.data()+3);
  }

  typedef std::map<track_id_t, std::vector<double> > lm_param_map_t;
  typedef std::map<frame_id_t, std::vector<double> > cam_param_map_t;

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
  VITAL_FOREACH(const track_sptr& t, trks)
  {
    const track_id_t id = t->id();
    lm_param_map_t::iterator lm_itr = d_->landmark_params.find(id);
    // skip this track if the landmark is not in the set to optimize
    if( lm_itr == d_->landmark_params.end() )
    {
      continue;
    }

    for(track::history_const_itr ts = t->begin(); ts != t->end(); ++ts)
    {
      cam_param_map_t::iterator cam_itr = d_->camera_params.find((*ts)->frame());
      if( cam_itr == d_->camera_params.end() )
      {
        continue;
      }
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*ts);
      if( !fts || !fts->feature )
      {
        continue;
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

  const unsigned int ndp = num_distortion_params(d_->lens_distortion_type);
  VITAL_FOREACH(const unsigned int idx, used_intrinsics)
  {
    std::vector<double>& cip = d_->camera_intr_params[idx];
    // apply the constraints
    if (constant_intrinsics.size() > 4 + ndp)
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

  // Add camera path regularization residuals
  d_->add_camera_path_smoothness_cost(problem, d_->camera_params);

  // Add camera path regularization residuals
  d_->add_forward_motion_damping_cost(problem, d_->camera_params, d_->frame_to_intr_map);

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
    LOG_DEBUG(d_->m_logger, "Ceres Full Report:\n" << summary.FullReport());
  }

  // Update the landmarks with the optimized values
  VITAL_FOREACH(const lm_param_map_t::value_type& lmp, d_->landmark_params)
  {
    auto& lmi = d_->lms[lmp.first];
    auto updated_lm = std::make_shared<landmark_d>(*lmi);
    updated_lm->set_loc(Eigen::Map<const vector_3d>(&lmp.second[0]));
    lmi = updated_lm;
  }
  landmarks = std::make_shared<simple_landmark_map>(d_->lms);

  // Update the cameras with the optimized values
  d_->update_camera_parameters(d_->cams, d_->camera_params,
                               d_->camera_intr_params, d_->frame_to_intr_map);
  cameras = std::make_shared<simple_camera_map>(d_->cams);
}


/// Set a callback function to report intermediate progress
void
bundle_adjust
::set_callback(callback_t cb)
{
  kwiver::vital::algo::bundle_adjust::set_callback(cb);
  ::ceres::Solver::Options& o = d_->options;
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


/// This function is called by a Ceres callback to trigger a kwiver callback
bool
bundle_adjust
::trigger_callback()
{
  if(this->m_callback)
  {
    // Update the landmarks with the optimized values
    typedef std::map<track_id_t, std::vector<double> > lm_param_map_t;
    VITAL_FOREACH(const lm_param_map_t::value_type& lmp, d_->landmark_params)
    {
      auto& lmi = d_->lms[lmp.first];
      auto updated_lm = std::make_shared<landmark_d>(*lmi);
      updated_lm->set_loc(Eigen::Map<const vector_3d>(&lmp.second[0]));
      lmi = updated_lm;
    }
    landmark_map_sptr landmarks = std::make_shared<simple_landmark_map>(d_->lms);

    // Update the cameras with the optimized values
    d_->update_camera_parameters(d_->cams, d_->camera_params,
                                 d_->camera_intr_params, d_->frame_to_intr_map);
    camera_map_sptr cameras = std::make_shared<simple_camera_map>(d_->cams);

    return this->m_callback(cameras, landmarks);
  }
  return true;
}


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
