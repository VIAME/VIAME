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
* \brief Header defining CERES algorithm implementation of camera optimization.
*/

#include "optimize_cameras.h"
#include <arrows/ceres/options.h>
#include <arrows/ceres/reprojection_error.h>
#include <vital/exceptions.h>


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
  d_->camera_options::set_configuration(config);

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


/// Optimize camera parameters given sets of landmarks and tracks
void
optimize_cameras
::optimize(vital::camera_map_sptr & cameras,
           vital::track_set_sptr tracks,
           vital::landmark_map_sptr landmarks) const
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

  // Extract the camera parameters into a mutable map
  //
  typedef std::map<frame_id_t, std::vector<double> > cam_param_map_t;
  cam_param_map_t camera_params;
  // We need maps from both frame number and intrinsics_sptr to the
  // index of the camera parameters.  This way we can vary how each
  // frame maps to a set of intrinsic parameters based on config params.
  typedef std::map<camera_intrinsics_sptr, unsigned int> cam_intrin_map_t;
  cam_intrin_map_t camera_intr_map;
  typedef std::map<frame_id_t, unsigned int> frame_to_intrin_map_t;
  frame_to_intrin_map_t frame_to_intr_map;
  // vector of unique camera intrinsic parameters
  std::vector<std::vector<double> > camera_intr_params;
  // number of lens distortion parameters in the selected model
  const unsigned int ndp = num_distortion_params(d_->lens_distortion_type);
  std::vector<double> intrinsic_params(5 + ndp, 0.0);
  VITAL_FOREACH(const map_camera_t::value_type& c, cams)
  {
    vector_3d rot = c.second->rotation().rodrigues();
    vector_3d center = c.second->center();
    std::vector<double> params(6);
    std::copy(rot.data(), rot.data()+3, params.begin());
    std::copy(center.data(), center.data()+3, params.begin()+3);
    camera_intrinsics_sptr K = c.second->intrinsics();
    camera_params[c.first] = params;

    // add a new set of intrisic parameter if one of the following:
    // - we are forcing unique intrinsics for each camera
    // - we are forcing common intrinsics and this is the first frame
    // - we are auto detecting shared intrinisics and this is a new sptr
    if( d_->camera_intrinsic_share_type == FORCE_UNIQUE_INTRINSICS
        || ( d_->camera_intrinsic_share_type == FORCE_COMMON_INTRINSICS
             && camera_intr_params.empty() )
        || camera_intr_map.count(K) == 0 )
    {
      intrinsic_params[0] = K->focal_length();
      intrinsic_params[1] = K->principal_point().x();
      intrinsic_params[2] = K->principal_point().y();
      intrinsic_params[3] = K->aspect_ratio();
      intrinsic_params[4] = K->skew();
      const std::vector<double> d = K->dist_coeffs();
      // copy the intersection of parameters provided in K
      // and those that are supported by the requested model type
      unsigned int num_dp = std::min(ndp, static_cast<unsigned int>(d.size()));
      if( num_dp > 0 )
      {
        std::copy(d.begin(), d.begin()+num_dp, &intrinsic_params[5]);
      }
      // update the maps with the index of this new parameter vector
      camera_intr_map[K] = static_cast<unsigned int>(camera_intr_params.size());
      frame_to_intr_map[c.first] = static_cast<unsigned int>(camera_intr_params.size());
      // add the parameter vector
      camera_intr_params.push_back(intrinsic_params);
    }
    else if( d_->camera_intrinsic_share_type == FORCE_COMMON_INTRINSICS )
    {
      // map to the first parameter vector
      frame_to_intr_map[c.first] = 0;
    }
    else
    {
      // map to a previously seen parameter vector
      frame_to_intr_map[c.first] = camera_intr_map[K];
    }
  }

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
      cam_param_map_t::iterator cam_itr = camera_params.find(ts->frame_id);
      if( cam_itr == camera_params.end() )
      {
        continue;
      }
      unsigned intr_idx = frame_to_intr_map[ts->frame_id];
      double * intr_params_ptr = &camera_intr_params[intr_idx][0];
      vector_2d pt = ts->feat->loc();
      problem.AddResidualBlock(create_cost_func(d_->lens_distortion_type,
                                                pt.x(), pt.y()),
                               loss_func,
                               intr_params_ptr,
                               &cam_itr->second[0],
                               &lm_itr->second[0]);
      loss_func_used = true;
    }
  }
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

  // If the loss function was added to a residual block, ownership was
  // transfered.  If not then we need to delete it.
  if(loss_func && !loss_func_used)
  {
    delete loss_func;
  }

  ::ceres::Solver::Summary summary;
  ::ceres::Solve(d_->options, &problem, &summary);
  LOG_DEBUG(d_->m_logger, "Ceres Full Report:\n" << summary.FullReport());

  // Update the camera intrinics with optimized values
  std::vector<camera_intrinsics_sptr> updated_intr;
  VITAL_FOREACH(const std::vector<double>& cip, camera_intr_params)
  {
    simple_camera_intrinsics* K = new simple_camera_intrinsics();
    K->set_focal_length(cip[0]);
    vector_2d pp((Eigen::Map<const vector_2d>(&cip[1])));
    K->set_principal_point(pp);
    K->set_aspect_ratio(cip[3]);
    K->set_skew(cip[4]);
    if( ndp > 0 )
    {
      Eigen::VectorXd dc(ndp);
      std::copy(&cip[5], &cip[5]+ndp, dc.data());
      K->set_dist_coeffs(dc);
    }
    updated_intr.push_back(camera_intrinsics_sptr(K));
  }

  // Update the cameras with the optimized values
  VITAL_FOREACH(const cam_param_map_t::value_type& cp, camera_params)
  {
    vector_3d center = Eigen::Map<const vector_3d>(&cp.second[3]);
    rotation_d rot(vector_3d(Eigen::Map<const vector_3d>(&cp.second[0])));

    // look-up updated intrinsics
    unsigned int intr_idx = frame_to_intr_map[cp.first];
    camera_intrinsics_sptr K = updated_intr[intr_idx];
    cams[cp.first] = camera_sptr(new simple_camera(center, rot, K));
  }

  cameras = camera_map_sptr(new simple_camera_map(cams));
}


/// Optimize a single camera given corresponding features and landmarks
void
optimize_cameras
::optimize(vital::camera_sptr& camera,
           const std::vector<vital::feature_sptr>& features,
           const std::vector<vital::landmark_sptr>& landmarks) const
{
}


} // end namespace ceres
} // end namespace arrows
} // end namespace kwiver
