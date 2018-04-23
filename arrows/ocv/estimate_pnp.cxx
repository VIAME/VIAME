/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * \brief OCV estimate_pnp algorithm implementation
 */

#include <cmath>

#include "estimate_pnp.h"


#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include "camera_intrinsics.h"

namespace kwiver {
namespace arrows {
namespace ocv
{

/// Private implementation class
class estimate_pnp::priv
{
public:
  /// Constructor
  priv()
    : confidence_threshold(0.99)
    , max_iterations(10000)
    , m_logger( vital::get_logger( "arrows.ocv.estimate_pnp" ))
  {
  }

  double confidence_threshold;
  int max_iterations;

  /// Logger handle
  vital::logger_handle_t m_logger;
};


/// Constructor
estimate_pnp
::estimate_pnp()
: d_(new priv)
{
}


/// Destructor
estimate_pnp
::~estimate_pnp()
{
}

/// Get this algorithm's \link vital::config_block configuration block \endlink
vital::config_block_sptr
estimate_pnp
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
      vital::algo::estimate_pnp::get_configuration();

  config->set_value("confidence_threshold", d_->confidence_threshold,
    "Confidence that estimated matrix is correct, range (0.0, 1.0]");

  config->set_value("max_iterations", d_->max_iterations,
    "maximum number of iterations to run PnP [1, INT_MAX]");

  return config;
}


/// Set this algorithm's properties via a config block
void
estimate_pnp
::set_configuration(vital::config_block_sptr config)
{
  d_->confidence_threshold =
    config->get_value<double>("confidence_threshold", d_->confidence_threshold);
  d_->max_iterations =
    config->get_value<int>("max_iterations", d_->max_iterations);
}


/// Check that the algorithm's configuration vital::config_block is valid
bool
estimate_pnp
::check_configuration(vital::config_block_sptr config) const
{
  bool good_conf = true;
  double confidence_threshold =
    config->get_value<double>("confidence_threshold", d_->confidence_threshold);

  if( confidence_threshold <= 0.0 || confidence_threshold > 1.0 )
  {
    LOG_ERROR(d_->m_logger, "confidence_threshold parameter is "
                            << confidence_threshold
                            << ", needs to be in (0.0, 1.0].");
    good_conf = false;
  }

  int max_iterations =
    config->get_value<int>("max_iterations", d_->max_iterations);

  if (max_iterations < 1)
  {
    LOG_ERROR(d_->m_logger, "max iterations is " << max_iterations
      << ", needs to be greater than zero.");
    good_conf = false;
  }

  return good_conf;
}


/// Estimate a camera pose from corresponding points
vital::camera_perspective_sptr
estimate_pnp
::estimate(const std::vector<vital::vector_2d>& pts2d,
           const std::vector<vital::vector_3d>& pts3d,
           const kwiver::vital::camera_intrinsics_sptr cal,
           std::vector<bool>& inliers) const
{
  if (pts2d.size() < 3 || pts3d.size() < 3)
  {
    LOG_ERROR(d_->m_logger,
      "Not enough points to estimate camera's pose");
    return vital::camera_perspective_sptr();
  }
  if (pts2d.size() != pts3d.size())
  {
    LOG_ERROR(d_->m_logger,
      "Number of 3D points and projections should match.  They don't.");
  }

  std::vector<cv::Point2f> projs;
  std::vector<cv::Point3f> Xs;
  for(const vital::vector_2d& p : pts2d)
  {
    projs.push_back(cv::Point2f(static_cast<float>(p.x()),
                                static_cast<float>(p.y())));
  }
  for(const vital::vector_3d& X : pts3d)
  {
    Xs.push_back(cv::Point3f(static_cast<float>(X.x()),
                             static_cast<float>(X.y()),
                             static_cast<float>(X.z())));
  }

  cv::Mat best_inliers_mat;
  cv::Mat best_rvec, best_tvec;


  int num_iterations = 5;
  double reproj_error = 4;
  double OCV_confidence = 0.98;  //I'm not sure what this is for.

  vital::matrix_3x3d K = cal->as_matrix();
  cv::Mat cv_K;
  cv::eigen2cv(K, cv_K);

  double confidence = 0;
  double confidence_thresh = d_->confidence_threshold;
  int max_iterations = d_->max_iterations;  // set some maximum because we don't
                                            // want to wait forever
  int iterations = 0;
  double best_inlier_ratio = 0;
  const double sample_size = 3;

  std::vector<double> dist_coeffs = get_ocv_dist_coeffs(cal);

  while (confidence < confidence_thresh && iterations < max_iterations)
  {
    cv::Mat inliers_mat;
    cv::Mat rvec, tvec;
    cv::solvePnPRansac(Xs, projs, cv_K, dist_coeffs, rvec, tvec, false,
      num_iterations, reproj_error, OCV_confidence, inliers_mat,
      cv::SOLVEPNP_EPNP);

    iterations += num_iterations;

    if (inliers_mat.rows > best_inliers_mat.rows)
    {
      //we found more inliers
      best_inliers_mat = inliers_mat;
      best_inlier_ratio = ((double)best_inliers_mat.rows / (double)Xs.size());
      best_tvec = tvec;
      best_rvec = rvec;
    }

    confidence = 1.0 - std::pow((1.0 - std::pow(best_inlier_ratio, sample_size)),
      double(iterations));
  }

  if (best_tvec.rows == 0 || best_rvec.rows == 0)
  {
    LOG_DEBUG(d_->m_logger, "no PnP solution after " << iterations << " iterations "
      " with confidence " << confidence << " and best inlier ratio " <<
      best_inlier_ratio );

    return vital::camera_perspective_sptr();
  }

  inliers.assign(Xs.size(), 0);

  for(int i = 0; i < best_inliers_mat.rows; ++i)
  {
    int idx = best_inliers_mat.at<int>(i);
    inliers[idx] = true;
  }

  auto res_cam = std::make_shared<vital::simple_camera_perspective>();
  Eigen::Vector3d rvec_eig, tvec_eig;
  cv::cv2eigen(best_rvec, rvec_eig);
  cv::cv2eigen(best_tvec, tvec_eig);
  vital::rotation_d rot(rvec_eig);
  res_cam->set_rotation(rot);
  res_cam->set_translation(tvec_eig);
  res_cam->set_intrinsics(cal);

  if (!std::isfinite(res_cam->center().x()))
  {
    LOG_DEBUG(d_->m_logger, "best_rvec " << best_rvec.at<double>(0) << " " <<
      best_rvec.at<double>(1) << " " << best_rvec.at<double>(2));
    LOG_DEBUG(d_->m_logger, "best_tvec " << best_tvec.at<double>(0) << " " <<
      best_tvec.at<double>(1) << " " << best_tvec.at<double>(2));
    LOG_DEBUG(d_->m_logger, "rotation angle " << res_cam->rotation().angle());
    LOG_WARN(d_->m_logger, "non-finite camera center found");
    return vital::camera_perspective_sptr();
  }

  return std::dynamic_pointer_cast<vital::camera_perspective>(res_cam);
}


} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
