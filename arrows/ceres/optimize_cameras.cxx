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
#include "options.h"



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
