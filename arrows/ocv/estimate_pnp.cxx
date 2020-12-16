// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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

  const double reproj_error = 4;

  vital::matrix_3x3d K = cal->as_matrix();
  cv::Mat cv_K;
  cv::eigen2cv(K, cv_K);

  std::vector<double> dist_coeffs = get_ocv_dist_coeffs(cal);

  cv::Mat inliers_mat;
  cv::Mat rvec, tvec;
  bool success =
    cv::solvePnPRansac(Xs, projs, cv_K, dist_coeffs, rvec, tvec, false,
                       d_->max_iterations, reproj_error,
                       d_->confidence_threshold, inliers_mat,
                       cv::SOLVEPNP_EPNP);

  double inlier_ratio = ((double)inliers_mat.rows / (double)Xs.size());

  if (!success || tvec.rows == 0 || rvec.rows == 0)
  {
    LOG_DEBUG(d_->m_logger, "no PnP solution after " << d_->max_iterations
              << " iterations with confidence " << d_->confidence_threshold
              << " and best inlier ratio " << inlier_ratio );

    return vital::camera_perspective_sptr();
  }

  inliers.assign(Xs.size(), 0);

  for(int i = 0; i < inliers_mat.rows; ++i)
  {
    int idx = inliers_mat.at<int>(i);
    inliers[idx] = true;
  }

  auto res_cam = std::make_shared<vital::simple_camera_perspective>();
  Eigen::Vector3d rvec_eig, tvec_eig;
  cv::cv2eigen(rvec, rvec_eig);
  cv::cv2eigen(tvec, tvec_eig);
  vital::rotation_d rot(rvec_eig);
  res_cam->set_rotation(rot);
  res_cam->set_translation(tvec_eig);
  res_cam->set_intrinsics(cal);

  if (!std::isfinite(res_cam->center().x()))
  {
    LOG_DEBUG(d_->m_logger, "rvec " << rvec.at<double>(0) << " " <<
              rvec.at<double>(1) << " " << rvec.at<double>(2));
    LOG_DEBUG(d_->m_logger, "tvec " << tvec.at<double>(0) << " " <<
              tvec.at<double>(1) << " " << tvec.at<double>(2));
    LOG_DEBUG(d_->m_logger, "rotation angle " << res_cam->rotation().angle());
    LOG_WARN(d_->m_logger, "non-finite camera center found");
    return vital::camera_perspective_sptr();
  }

  return std::dynamic_pointer_cast<vital::camera_perspective>(res_cam);
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
