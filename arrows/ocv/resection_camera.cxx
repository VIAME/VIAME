// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

/// \file
/// \brief OCV resection_camera algorithm implementation

#include "resection_camera.h"

#include "camera_intrinsics.h"

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>

#include <vital/range/iota.h>

namespace kvr = kwiver::vital::range;

namespace kwiver {

namespace arrows {

namespace ocv {

// ----------------------------------------------------------------------------
class resection_camera::priv
{
public:
  priv() : m_logger{ vital::get_logger( "arrows.ocv.resection_camera" ) }
  {
  }

  vital::logger_handle_t m_logger;

  double reproj_accuracy = 1.;
  int max_iterations = 300;
};

// ----------------------------------------------------------------------------
resection_camera
::resection_camera()
  : d_{ new priv }
{
}

// ----------------------------------------------------------------------------
resection_camera::~resection_camera()
{
}

// ----------------------------------------------------------------------------
vital::config_block_sptr
resection_camera
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config =
    vital::algo::resection_camera::get_configuration();
  config->set_value( "reproj_accuracy", d_->reproj_accuracy,
                     "desired re-projection positive accuracy" );
  config->set_value( "max_iterations", d_->max_iterations,
                     "maximum number of iterations to run PnP [1, INT_MAX]" );
  return config;
}

// ----------------------------------------------------------------------------
void
resection_camera
::set_configuration( vital::config_block_sptr config )
{
  d_->reproj_accuracy = config->get_value< double >( "reproj_accuracy",
                                                     d_->reproj_accuracy );
  d_->max_iterations = config->get_value< int >( "max_iterations",
                                                 d_->max_iterations );
}

// ----------------------------------------------------------------------------
bool
resection_camera
::check_configuration( vital::config_block_sptr config ) const
{
  auto good_conf = true;
  auto const reproj_accuracy =
    config->get_value< double >( "reproj_accuracy", d_->reproj_accuracy );
  if( reproj_accuracy <= 0.0 )
  {
    LOG_ERROR( d_->m_logger,
               "reproj_accuracy parameter is " << reproj_accuracy <<
               ", but needs to be positive." );
    good_conf = false;
  }

  auto const max_iterations =
    config->get_value< int >( "max_iterations", d_->max_iterations );
  if( max_iterations < 1 )
  {
    LOG_ERROR( d_->m_logger,
               "max iterations is " << max_iterations <<
               ", needs to be greater than zero." );
    good_conf = false;
  }
  return good_conf;
}

// ----------------------------------------------------------------------------
vital::camera_perspective_sptr
resection_camera
::resection(
  std::vector< kwiver::vital::vector_2d > const& image_points,
  std::vector< kwiver::vital::vector_3d > const& world_points,
  vital::camera_intrinsics_sptr cal,
  std::vector< bool >* inliers ) const
{
  if( cal == nullptr )
  {
    LOG_ERROR( d_->m_logger, "camera calibration guess should not be null" );
    return nullptr;
  }

  auto const point_count = image_points.size();
  if( point_count != world_points.size() )
  {
    LOG_WARN( d_->m_logger,
              "counts of 3D points and projections do not match" );
  }

  constexpr size_t min_count = 3;
  if( point_count < min_count )
  {
    LOG_ERROR( d_->m_logger, "not enough points to resection camera" );
    return nullptr;
  }

  std::vector< cv::Point2f > cv_image_points;
  cv_image_points.reserve( point_count );
  for( auto const& p : image_points )
  {
    cv_image_points.emplace_back( p.x(), p.y() );
  }

  std::vector< cv::Point3f > cv_world_points;
  cv_world_points.reserve( point_count );
  for( const auto& p : world_points )
  {
    cv_world_points.emplace_back( p.x(), p.y(), p.z() );
  }

  vital::matrix_3x3d K = cal->as_matrix();
  cv::Mat cv_K;
  eigen2cv( K, cv_K );

  auto dist_coeffs = get_ocv_dist_coeffs( cal );
  cv::Mat inliers_mat;
  std::vector< cv::Mat > vrvec, vtvec;
  auto const world_points_vec =
    std::vector< std::vector< cv::Point3f > >{ cv_world_points };
  auto const image_points_vec =
    std::vector< std::vector< cv::Point2f > >{ cv_image_points };
  auto const image_size =
    cv::Size{ static_cast< int >( cal->image_width() ),
              static_cast< int >( cal->image_height() ) };
  int flags = cv::CALIB_USE_INTRINSIC_GUESS;
  auto const reproj_error = d_->reproj_accuracy;

  auto const err =
    cv::calibrateCamera( world_points_vec, image_points_vec,
                         image_size, cv_K, dist_coeffs,
                         vrvec, vtvec, flags,
                         cv::TermCriteria{
                           cv::TermCriteria::COUNT + cv::TermCriteria::EPS,
                           d_->max_iterations, reproj_error } );

  if( err > reproj_error )
  {
    LOG_WARN( d_->m_logger, "estimated re-projection error " <<
              err << " exceeds expected re-projection error " <<
              reproj_error );
  }

  cv::Mat rvec = vrvec[ 0 ];
  cv::Mat tvec = vtvec[ 0 ];

  if( inliers )
  {
    std::vector< cv::Point2f > projected_points;
    projectPoints( cv_world_points, rvec, tvec, cv_K,
                   dist_coeffs, projected_points );

    inliers->resize( point_count );
    for( auto const i : kvr::iota( point_count ) )
    {
      auto const delta = norm( projected_points[ i ] - cv_image_points[ i ] );
      ( *inliers )[ i ] = ( delta < reproj_error );
    }
  }

  auto res_cam = std::make_shared< vital::simple_camera_perspective >();
  Eigen::Vector3d rvec_eig, tvec_eig;
  auto const dc_size = dist_coeffs.size();

  Eigen::VectorXd dist_eig( dist_coeffs.size() );
  for( auto const i : kvr::iota( dc_size ) )
  {
    dist_eig[ static_cast< int >( i ) ] = dist_coeffs[ i ];
  }
  cv::cv2eigen( rvec, rvec_eig );
  cv::cv2eigen( tvec, tvec_eig );
  cv::cv2eigen( cv_K, K );

  vital::rotation_d rot{ rvec_eig };
  res_cam->set_rotation( rot );
  res_cam->set_translation( tvec_eig );
  cal = std::make_shared< vital::simple_camera_intrinsics >( K, dist_eig );
  res_cam->set_intrinsics( cal );

  if( !res_cam->center().allFinite() )
  {
    LOG_DEBUG( d_->m_logger, "rvec " << rvec.at< double >( 0 ) << " " <<
               rvec.at< double >( 1 ) << " " << rvec.at< double >( 2 ) );
    LOG_DEBUG( d_->m_logger, "tvec " << tvec.at< double >( 0 ) << " " <<
               tvec.at< double >( 1 ) << " " << tvec.at< double >( 2 ) );
    LOG_DEBUG( d_->m_logger,
               "rotation angle " << res_cam->rotation().angle() );
    LOG_WARN( d_->m_logger, "non-finite camera center found" );
    return nullptr;
  }
  return res_cam;
}

} // namespace ocv

} // namespace arrows

} // namespace kwiver
