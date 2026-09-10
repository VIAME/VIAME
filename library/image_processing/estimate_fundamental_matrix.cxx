// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV estimate_fundamental_matrix algorithm implementation

#include "estimate_fundamental_matrix.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

namespace kwiver {

namespace arrows {

namespace ocv {

// ----------------------------------------------------------------------------
void
estimate_fundamental_matrix
::initialize()
{
  attach_logger( "arrows.ocv.estimate_fundamental_matrix" );
}

// Destructor
estimate_fundamental_matrix
::~estimate_fundamental_matrix()
{}

// ----------------------------------------------------------------------------
// Check that the algorithm's configuration vital::config_block is valid
bool
estimate_fundamental_matrix
::check_configuration( vital::config_block_sptr config ) const
{
  double confidence_threshold =
    config->get_value< double >(
      "confidence_threshold",
      this->get_confidence_threshold() );
  if( confidence_threshold <= 0.0 || confidence_threshold > 1.0 )
  {
    LOG_ERROR(
      logger(), "confidence_threshold parameter is "
        << confidence_threshold
        << ", needs to be in (0.0, 1.0]." );
    return false;
  }

  return true;
}

// ----------------------------------------------------------------------------
// Estimate a fundamental matrix from corresponding points
vital::fundamental_matrix_sptr
estimate_fundamental_matrix
::estimate(
  const std::vector< vital::vector_2d >& pts1,
  const std::vector< vital::vector_2d >& pts2,
  std::vector< bool >& inliers,
  double inlier_scale ) const
{
  if( pts1.size() < 8 || pts2.size() < 8 )
  {
    LOG_ERROR(logger(), "Not enough points to estimate a fundamental matrix" );
    return vital::fundamental_matrix_sptr();
  }

  std::vector< cv::Point2f > points1, points2;
  for( const vital::vector_2d& v : pts1 )
  {
    points1.push_back(
      cv::Point2f(
        static_cast< float >( v.x() ),
        static_cast< float >( v.y() ) ) );
  }
  for( const vital::vector_2d& v : pts2 )
  {
    points2.push_back(
      cv::Point2f(
        static_cast< float >( v.x() ),
        static_cast< float >( v.y() ) ) );
  }

  cv::Mat inliers_mat;
  cv::Mat F = cv::findFundamentalMat(
    cv::Mat( points1 ), cv::Mat( points2 ),
    cv::FM_RANSAC,
    inlier_scale,
    this->get_confidence_threshold(),
    inliers_mat );
  inliers.resize( inliers_mat.rows );
  for( size_t i = 0; i < inliers.size(); ++i )
  {
    inliers[ i ] = inliers_mat.at< bool >( i );
  }

  vital::matrix_3x3d F_mat;
  cv2eigen( F, F_mat );
  return vital::fundamental_matrix_sptr(
    new vital::fundamental_matrix_< double >( F_mat ) );
}

} // end namespace ocv

} // end namespace arrows

} // end namespace kwiver
