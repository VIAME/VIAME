// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/**
 * \file
 * \brief OCV estimate_homography algorithm implementation
 */

#include "estimate_homography.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#if KWIVER_OPENCV_VERSION_MAJOR >= 3
#define OPENCV_METHOD_RANSAC cv::RANSAC
#else
#define OPENCV_METHOD_RANSAC CV_RANSAC
#endif

namespace kwiver {
namespace arrows {
namespace ocv {

/// Estimate a homography matrix from corresponding points
vital::homography_sptr
estimate_homography
::estimate(const std::vector<vital::vector_2d>& pts1,
           const std::vector<vital::vector_2d>& pts2,
           std::vector<bool>& inliers,
           double inlier_scale) const
{
  if (pts1.size() < 4 || pts2.size() < 4)
  {
    vital::logger_handle_t logger( vital::get_logger( "arrows.ocv.estimate_homography" ));
    LOG_ERROR(logger, "Not enough points to estimate a homography");
    return vital::homography_sptr();
  }

  std::vector<cv::Point2f> points1, points2;
  for(const vital::vector_2d& v : pts1)
  {
    points1.push_back(cv::Point2f(static_cast<float>(v.x()),
                                  static_cast<float>(v.y())));
  }
  for(const vital::vector_2d& v : pts2)
  {
    points2.push_back(cv::Point2f(static_cast<float>(v.x()),
                                  static_cast<float>(v.y())));
  }

  cv::Mat inliers_mat;
  cv::Mat H = cv::findHomography( cv::Mat(points1), cv::Mat(points2),
                                  OPENCV_METHOD_RANSAC,
                                  inlier_scale,
                                  inliers_mat );
  inliers.resize(inliers_mat.rows);
  for(unsigned i = 0; i < inliers.size(); ++i)
  {
    inliers[i] = inliers_mat.at<bool>(i);
  }

  vital::matrix_3x3d H_mat;
  cv2eigen(H, H_mat);
  return vital::homography_sptr( new vital::homography_<double>(H_mat) );
}

} // end namespace ocv
} // end namespace arrows
} // end namespace kwiver
