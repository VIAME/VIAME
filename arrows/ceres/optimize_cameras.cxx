/*ckwg +29
 * Copyright 2016-2017 by Kitware, Inc.
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
* \brief Header defining CERES algorithm implementation of camera optimization.
*/

#include "optimize_cameras.h"
#include <arrows/ceres/options.h>
#include <arrows/ceres/reprojection_error.h>
#include <vital/exceptions.h>
#include <vital/vital_foreach.h>


using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace ceres {


/// Private implementation class
class optimize_cameras::priv
  : public solver_options,
    public camera_options
{
public:
  /// Constructor
  priv()
  : camera_options(),
    verbose(false),
    loss_function_type(TRIVIAL_LOSS),
    loss_function_scale(1.0),
    m_logger( vital::get_logger( "arrows.ceres.optimize_cameras" ))
  {
  }

  priv(const priv& other)
  : camera_options(other),
    verbose(other.verbose),
    loss_function_type(other.loss_function_type),
    loss_function_scale(other.loss_function_scale),
    m_logger( vital::get_logger( "arrows.ceres.optimize_cameras" ))
  {
  }

  /// verbose output
  bool verbose;
  /// the robust loss function type to use
  LossFunctionType loss_function_type;
  /// the scale of the loss function
  double loss_function_scale;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
optimize_cameras
::optimize_cameras()
: d_(new priv)
{
}


/// Copy Constructor
optimize_cameras
::optimize_cameras(const optimize_cameras& other)
: d_(new priv(*other.d_))
{
}


/// Destructor
optimize_cameras
::~optimize_cameras()
{
}


/// Get this algorithm's \link vital::config_block configuration block \endlink
config_block_sptr
optimize_cameras
::get_configuration() const
{
  // get base config from base class
  config_block_sptr config = vital::algo::optimize_cameras::get_configuration();
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
optimize_cameras
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
}


/// Check that the algorithm's currently configuration is valid
bool
optimize_cameras
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


/// Optimize camera parameters given sets of landmarks and feature tracks
void
optimize_cameras
::optimize(vital::camera_map_sptr & cameras,
           vital::feature_track_set_sptr tracks,
           vital::landmark_map_sptr landmarks,
           vital::video_metadata_map_sptr metadata) const
{
  if( !cameras || !landmarks || !tracks )
  {
    throw vital::invalid_value("One or more input data pieces are Null!");
  }
  typedef camera_map::map_camera_t map_camera_t;
  typedef landmark_map::map_landmark_t map_landmark_t;

  // extract data from containers
  map_camera_t cams = cameras->cameras();
  map_landmark_t lms = landmarks->landmarks();
  std::vector<track_sptr> trks = tracks->tracks();

  // Extract the landmark locations into a mutable map
  typedef std::map<track_id_t, std::vector<double> > lm_param_map_t;
  lm_param_map_t landmark_params;
  VITAL_FOREACH(const map_landmark_t::value_type& lm, lms)
  {
    vector_3d loc = lm.second->loc();
    landmark_params[lm.first] = std::vector<double>(loc.data(), loc.data()+3);
  }

  // a map from frame number to extrinsic parameters
  typedef std::map<frame_id_t, std::vector<double> > cam_param_map_t;
  cam_param_map_t camera_params;
  // vector of unique camera intrinsic parameters
  std::vector<std::vector<double> > camera_intr_params;
  // a map from frame number to index of unique camera intrinsics in camera_intr_params
  std::map<frame_id_t, unsigned int> frame_to_intr_map;

  // Extract the raw camera parameter into the provided maps
  d_->extract_camera_parameters(cams,
                                camera_params,
                                camera_intr_params,
                                frame_to_intr_map);

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
  VITAL_FOREACH(const track_sptr& t, trks)
  {
    const track_id_t id = t->id();
    lm_param_map_t::iterator lm_itr = landmark_params.find(id);
    // skip this track if the landmark is not in the set to optimize
    if( lm_itr == landmark_params.end() )
    {
      continue;
    }

    for(track::history_const_itr ts = t->begin(); ts != t->end(); ++ts)
    {
      cam_param_map_t::iterator cam_itr = camera_params.find((*ts)->frame());
      if( cam_itr == camera_params.end() )
      {
        continue;
      }
      unsigned intr_idx = frame_to_intr_map[(*ts)->frame()];
      double * intr_params_ptr = &camera_intr_params[intr_idx][0];
      auto fts = std::dynamic_pointer_cast<feature_track_state>(*ts);
      if( !fts || !fts->feature )
      {
        continue;
      }
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
  VITAL_FOREACH(std::vector<double>& cip, camera_intr_params)
  {
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
  // Set the landmarks constant
  VITAL_FOREACH(lm_param_map_t::value_type& lmp, landmark_params)
  {
    problem.SetParameterBlockConstant(&lmp.second[0]);
  }

  // Add camera path regularization residuals
  d_->add_camera_path_smoothness_cost(problem, camera_params);

  // Add camera path regularization residuals
  d_->add_forward_motion_damping_cost(problem, camera_params, frame_to_intr_map);

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

  // Update the cameras with the optimized values
  d_->update_camera_parameters(cams, camera_params,
                               camera_intr_params, frame_to_intr_map);
  cameras = std::make_shared<simple_camera_map>(cams);
}


/// Optimize a single camera given corresponding features and landmarks
void
optimize_cameras
::optimize(vital::camera_sptr& camera,
           const std::vector<vital::feature_sptr>& features,
           const std::vector<vital::landmark_sptr>& landmarks,
           kwiver::vital::video_metadata_vector metadata) const
{
  // extract camera parameters to optimize
  const unsigned int ndp = num_distortion_params(d_->lens_distortion_type);
  std::vector<double> cam_intrinsic_params(5 + ndp, 0.0);
  std::vector<double> cam_extrinsic_params(6);
  d_->extract_camera_extrinsics(camera, &cam_extrinsic_params[0]);
  camera_intrinsics_sptr K = camera->intrinsics();
  d_->extract_camera_intrinsics(K, &cam_intrinsic_params[0]);

  // extract the landmark parameters
  std::vector<std::vector<double> > landmark_params;
  VITAL_FOREACH(const landmark_sptr lm, landmarks)
  {
    vector_3d loc = lm->loc();
    landmark_params.push_back(std::vector<double>(loc.data(), loc.data()+3));
  }

  // the Ceres solver problem
  ::ceres::Problem problem;

  // enumerate the intrinsics held constant
  std::vector<int> constant_intrinsics = d_->enumerate_constant_intrinsics();

  // Create the loss function to use
  ::ceres::LossFunction* loss_func
      = LossFunctionFactory(d_->loss_function_type,
                            d_->loss_function_scale);

  // Add the residuals for each relevant observation
  for(unsigned int i=0; i<features.size(); ++i)
  {
    vector_2d pt = features[i]->loc();
    problem.AddResidualBlock(create_cost_func(d_->lens_distortion_type,
                                              pt.x(), pt.y()),
                             loss_func,
                             &cam_intrinsic_params[0],
                             &cam_extrinsic_params[0],
                             &landmark_params[i][0]);

    problem.SetParameterBlockConstant(&landmark_params[i][0]);
  }

  // set contraints on the camera intrinsics
  if (constant_intrinsics.size() > 4 + ndp)
  {
    // set all parameters in the block constant
    problem.SetParameterBlockConstant(&cam_intrinsic_params[0]);
  }
  else if (!constant_intrinsics.empty())
  {
    // set a subset of parameters in the block constant
    problem.SetParameterization(&cam_intrinsic_params[0],
        new ::ceres::SubsetParameterization(5 + ndp, constant_intrinsics));
  }

  // If the loss function was added to a residual block, ownership was
  // transfered.  If not then we need to delete it.
  if(loss_func && !features.empty())
  {
    delete loss_func;
  }

  ::ceres::Solver::Summary summary;
  ::ceres::Solve(d_->options, &problem, &summary);
  if( d_->verbose )
  {
    LOG_DEBUG(d_->m_logger, "Ceres Full Report:\n" << summary.FullReport());
  }

  // update the cameras from optimized parameters
  // only create a new intrinsics object if the values were not held constant
  if ( d_->optimize_intrinsics() )
  {
    auto new_K = std::make_shared<simple_camera_intrinsics>();
    d_->update_camera_intrinsics(new_K, &cam_intrinsic_params[0]);
    K = new_K;
  }
  auto new_camera = std::make_shared<simple_camera>();
  new_camera->set_intrinsics(K);
  d_->update_camera_extrinsics(new_camera, &cam_extrinsic_params[0]);
  camera = new_camera;
}


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
