/**
* \file
* \brief Header defining Opencv algorithm implementation of camera optimization to optimize stereo setup.
*/

#include "optimize_stereo_cameras.h"
#include "calibrate_stereo_cameras.h"
#include "filter_stereo_feature_tracks.h"

#include <vital/types/object_track_set.h>
#include <vital/types/camera_perspective_map.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/types/feature_track_set.h>
#include <vital/types/landmark_map.h>
#include <vital/io/camera_io.h>
#include <arrows/ocv/camera_intrinsics.h>

#include <vital/range/iota.h>

#include <vital/exceptions.h>
#include <vital/vital_config.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/core/persistence.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cmath>
#include <numeric>

namespace kv = kwiver::vital;

namespace viame {

// Private implementation class
class optimize_stereo_cameras::priv
{
public:
  unsigned m_image_width{};
  unsigned m_image_height{};
  unsigned m_frame_count_threshold{};
  std::string m_output_calibration_directory{};
  std::string m_output_json_file{};
  double m_square_size{ 1.0 };

  // Shared calibration utility
  calibrate_stereo_cameras m_calibrator;

  kv::logger_handle_t m_logger;

  static kv::feature_track_set_sptr
  merge_features_track( const kv::feature_track_set_sptr& feature_track,
                        const kv::feature_track_set_sptr& feature_track1,
                        const kv::feature_track_set_sptr& feature_track2 );

  void calibrate_camera( const kv::camera_sptr& camera,
                         const kv::feature_track_set_sptr& features,
                         const kv::landmark_map_sptr& landmarks,
                         const std::string& suffix );

  void calibrate_stereo_camera( kv::camera_map::map_camera_t cameras,
                                const kv::feature_track_set_sptr& features1,
                                const kv::feature_track_set_sptr& features2,
                                const kv::landmark_map_sptr& landmarks1,
                                const kv::landmark_map_sptr& landmarks2 );


  StereoPointCoordinates
  convert_features_and_landmarks_to_calib_points( const FeatureTracks& features,
                                                  const Landmarks& landmarks,
                                                  bool& success );

  bool try_improve_camera_calibration( const std::vector< std::vector< cv::Point3f > >& world_points,
                                       const std::vector< std::vector< cv::Point2f > >& image_points,
                                       const cv::Size& image_size,
                                       cv::Mat& K1, cv::Mat& D1, cv::Mat& R1, cv::Mat& T1,
                                       int flags, double max_error, double& error,
                                       const std::string& context );

  void write_stereo_calibration_file( const cv::Mat& M1, const cv::Mat& M2,
                                       const std::vector< double >& D1,
                                       const std::vector< double >& D2,
                                       const cv::Mat& R, const cv::Mat& T,
                                       const cv::Mat& R1, const cv::Mat& R2,
                                       const cv::Mat& P1, const cv::Mat& P2,
                                       const cv::Mat& Q ) const
  {
    // Use current directory if output directory not specified
    std::string output_dir = m_output_calibration_directory.empty() ? "." : m_output_calibration_directory;

    auto fs = cv::FileStorage( output_dir + "/intrinsics.yml", cv::FileStorage::Mode::WRITE );
    if( fs.isOpened() )
    {
      fs.write( "M1", M1 );
      fs.write( "M2", M2 );
      fs.write( "D1", cv::Mat( D1 ) );
      fs.write( "D2", cv::Mat( D2 ) );
    }
    fs.release();

    // write extrinsic file
    fs = cv::FileStorage( output_dir + "/extrinsics.yml", cv::FileStorage::Mode::WRITE );
    if( fs.isOpened() )
    {
      fs.write( "R", R );
      fs.write( "T", T );
      fs.write( "R1", R1 );
      fs.write( "R2", R2 );
      fs.write( "P1", P1 );
      fs.write( "P2", P2 );
      fs.write( "Q", Q );
    }
    fs.release();
  }
}; // end class optimize_stereo_cameras::priv

kv::feature_track_set_sptr
optimize_stereo_cameras::priv
::merge_features_track( const kv::feature_track_set_sptr& feature_track,
                        const kv::feature_track_set_sptr& feature_track1,
                        const kv::feature_track_set_sptr& feature_track2 )
{
  std::vector< kv::track_id_t > all_track_ids;
  std::vector< kv::track_sptr > all_tracks1 = feature_track1->tracks();
  std::vector< kv::track_sptr > all_tracks2 = feature_track2->tracks();
  for( const auto& track : all_tracks1 )
    all_track_ids.push_back( track->id() );

  kv::track_id_t max_track1_id = *std::max_element( all_track_ids.begin(), all_track_ids.end() );

  for( const auto& track : all_tracks2 )
    all_track_ids.push_back( track->id() + max_track1_id );

  std::vector< kv::track_sptr > all_tracks;
  all_tracks.insert( all_tracks.end(), all_tracks1.begin(), all_tracks1.end() );
  all_tracks.insert( all_tracks.end(), all_tracks2.begin(), all_tracks2.end() );

  std::vector< kv::track_sptr > tracks;
  for( unsigned i = 0; i < all_tracks.size(); i++ )
  {
    kv::track_sptr t = kv::track::create();
    t->set_id( all_track_ids[i] );
    tracks.push_back( t );

    for( auto state : *all_tracks[i] | kv::as_feature_track )
    {
      auto fts = std::make_shared< kv::feature_track_state >( state->frame() );
      fts->feature = std::make_shared< kv::feature_d >( state->feature->loc() );
      fts->inlier = true;
      t->append( fts );
    }
  }

  return std::make_shared< kv::feature_track_set >( tracks );
}


bool
optimize_stereo_cameras::priv
::try_improve_camera_calibration( const std::vector< std::vector< cv::Point3f > >& world_points,
                                  const std::vector< std::vector< cv::Point2f > >& image_points,
                                  const cv::Size& image_size,
                                  cv::Mat& K1, cv::Mat& D1, cv::Mat& R1, cv::Mat& T1,
                                  int flags, double max_error, double& error,
                                  const std::string& context )
{
  cv::Mat prev_K1, prev_D1, prev_R1, prev_T1;
  prev_K1 = K1.clone();
  prev_D1 = D1.clone();
  prev_R1 = R1.clone();
  prev_T1 = T1.clone();

  LOG_INFO( m_logger, "  - Running intrinsic calibration: " << context );
  error = cv::calibrateCamera( world_points, image_points, image_size, K1, D1, R1, T1, flags );
  LOG_INFO( m_logger, "    Calibration error: " << error );

  if( error < max_error )
    return true;

  // Rollback parameters
  LOG_INFO( m_logger, "    Error too high, keeping previous parameters" );
  K1 = prev_K1.clone();
  D1 = prev_D1.clone();
  R1 = prev_R1.clone();
  T1 = prev_T1.clone();

  return false;
}


void
optimize_stereo_cameras::priv
::calibrate_camera( const kv::camera_sptr& camera,
                    const kv::feature_track_set_sptr& features,
                    const kv::landmark_map_sptr& landmarks,
                    const std::string& suffix )
{
  kv::simple_camera_perspective_sptr cam;
  cam = std::dynamic_pointer_cast< kv::simple_camera_perspective >( camera );

  if( !cam )
  {
    LOG_WARN( m_logger, "Unable to get the camera." );
    return;
  }

  bool success;
  auto points = convert_features_and_landmarks_to_calib_points( { features }, { landmarks }, success );
  if( !success )
    return;

  auto image_points = points.image_pts[0];
  auto world_points = points.world_pts;

  // Init left/right camera matrix
  cv::Size image_size = cv::Size( m_image_width, m_image_height );
  cv::Mat cv_K1 = cv::Mat( 3, 3, CV_64F );
  auto dist_coeffs = cv::Mat( 1, 7, CV_64F );
  cv_K1 = cv::initCameraMatrix2D( world_points, image_points, image_size, 0 );

  // Run intrinsic calibration of left camera
  cv::Mat rvec1, tvec1;

  // Performing camera calibration by passing the value of known 3D world points (objpoints)
  // and corresponding pixel coordinates of the detected corners (imgpoints)
  size_t n_frames = world_points.size();
  LOG_INFO( m_logger, "Calibrating " << suffix << " camera (" << n_frames << " frames)..." );

  int flags{};
  double error{};
  double no_error_threshold{ std::numeric_limits< double >::max() };
  try_improve_camera_calibration( world_points, image_points, image_size, cv_K1, dist_coeffs, rvec1, tvec1, flags,
                                  no_error_threshold, error, "Initial" );

  // Fix aspect ratio if necessary
  auto aspect_ratio = cv_K1.at< double >( 0, 0 ) / cv_K1.at< double >( 1, 1 );
  LOG_INFO( m_logger, "  - Aspect ratio: " << aspect_ratio );

  if( 1.0 - std::min( aspect_ratio, 1.0 / aspect_ratio ) < 0.01 )
  {
    flags |= cv::CALIB_FIX_ASPECT_RATIO;
    try_improve_camera_calibration( world_points, image_points, image_size, cv_K1, dist_coeffs, rvec1, tvec1, flags,
                                    no_error_threshold, error, "Fixing aspect ratio at 1.0" );
  }

  // Fix principal point to center if necessary
  auto pp1 = cv_K1.at< double >( 0, 2 );
  auto pp2 = cv_K1.at< double >( 1, 2 );
  LOG_INFO( m_logger, "  - Principal point: (" << pp1 << ", " << pp2 << ")" );

  auto rel_pp_diff_1 = std::abs( pp1 - image_size.width / 2. ) / image_size.width;
  auto rel_pp_diff_2 = std::abs( pp2 - image_size.height / 2. ) / image_size.width;
  auto rel_pp_diff = std::max( rel_pp_diff_1, rel_pp_diff_2 );
  LOG_DEBUG( m_logger, "Relative principal point diff : " << rel_pp_diff );
  if( rel_pp_diff < 0.05 )
  {
    flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    try_improve_camera_calibration( world_points, image_points, image_size, cv_K1, dist_coeffs, rvec1, tvec1, flags,
                                    no_error_threshold, error, "Fixed principal point to image center" );
  }

  // For each distortion, if error is less than 25% of previous error, fix the distortion parameter
  std::vector< std::string > dist_context{ "No tangential distortion", "No K3 distortion", "No K2 distortion",
                                           "No K1 distortion" };
  std::vector< int > dist_flags{ cv::CALIB_ZERO_TANGENT_DIST, cv::CALIB_FIX_K3, cv::CALIB_FIX_K2, cv::CALIB_FIX_K1 };
  auto max_error = 1.25 * error;
  for( size_t i_flag = 0; i_flag < dist_flags.size(); i_flag++ )
  {
    flags |= dist_flags[i_flag];
    if( !try_improve_camera_calibration( world_points, image_points, image_size, cv_K1, dist_coeffs, rvec1, tvec1, flags,
                                         max_error, error, dist_context[i_flag] ) )
      break;
  }

  // Push calibration results to perspective camera
  kv::matrix_3x3d K1;
  auto res_cam1 = std::make_shared< kv::simple_camera_perspective >();

  Eigen::VectorXd dist_eig1( dist_coeffs.cols );
  for( auto const i : kv::range::iota( dist_coeffs.cols ) )
  {
    dist_eig1[static_cast< int >( i )] = dist_coeffs.at< double >( i );
  }
  cv::cv2eigen( cv_K1, K1 );

  kv::camera_intrinsics_sptr cal1;
  cal1 = std::make_shared< kv::simple_camera_intrinsics >( K1, dist_eig1 );
  res_cam1->set_intrinsics( cal1 );

  // copy camera params
  *cam = *res_cam1;

  LOG_DEBUG( m_logger, "Estimated " << suffix << " camera center :\n" << cam->translation().transpose() );
  LOG_DEBUG( m_logger, "Estimated " << suffix << " camera rotation :\n" << cam->rotation().matrix() );
  LOG_DEBUG( m_logger, "Estimated " << suffix << " camera intrinsic matrix :\n" << cam->intrinsics()->as_matrix() );
}

StereoPointCoordinates
optimize_stereo_cameras::priv
::convert_features_and_landmarks_to_calib_points( const FeatureTracks& features,
                                                  const Landmarks& landmarks,
                                                  bool& success )
{
  std::stringstream n_frames;
  for( const auto& feature : features )
    n_frames << feature->all_frame_ids().size() << ",";

  LOG_INFO( m_logger, "Selecting frames for calibration (" << m_frame_count_threshold << "/" << n_frames.str() << ")..." );
  auto points = filter_stereo_feature_tracks::select_frames( features, landmarks, m_frame_count_threshold );
  success = !points.image_pts.empty() && !points.image_pts[0].empty() &&
            ( points.image_pts[0].size() == points.world_pts.size() ) &&
            ( points.image_pts[0].size() == points.image_pts[1].size() );

  if( !success )
    LOG_WARN( m_logger, "Unable to proceed with camera calibration." );

  auto n_cams = std::min( features.size(), points.image_pts.size() );

  LOG_INFO( m_logger, "Calibration data prepared:" );
  LOG_INFO( m_logger, "  - Image size: " << m_image_width << "x" << m_image_height );
  LOG_INFO( m_logger, "  - Number of cameras: " << n_cams );
  for( size_t i_cam = 0; i_cam < n_cams; i_cam++ )
    LOG_INFO( m_logger, "  - Camera " << i_cam << " points: " << points.image_pts[i_cam].size() << " frames" );
  LOG_INFO( m_logger, "  - World points: " << points.world_pts.size() << " frames" );

  return points;
}


void
optimize_stereo_cameras::priv
::calibrate_stereo_camera( kv::camera_map::map_camera_t cameras,
                           const kv::feature_track_set_sptr& features1,
                           const kv::feature_track_set_sptr& features2,
                           const kv::landmark_map_sptr& landmarks1,
                           const kv::landmark_map_sptr& landmarks2 )
{
  if( cameras.size() != 2 )
  {
    LOG_WARN( m_logger, "Only works with two cameras as inputs." );
    return;
  }

  kv::simple_camera_perspective_sptr cam1, cam2;
  cam1 = std::dynamic_pointer_cast< kv::simple_camera_perspective >( cameras[0] );
  cam2 = std::dynamic_pointer_cast< kv::simple_camera_perspective >( cameras[1] );
  cv::Size image_size = cv::Size( m_image_width, m_image_height );

  bool success;
  auto points = convert_features_and_landmarks_to_calib_points( { features1, features2 }, { landmarks1, landmarks2 },
                                                                success );

  if( !success )
    return;

  auto image_points1 = points.image_pts[0];
  auto image_points2 = points.image_pts[1];
  auto world_points = points.world_pts;

  // Get mono camera calibration params
  kv::matrix_3x3d K1 = cam1->intrinsics()->as_matrix();
  cv::Mat cv_K1;
  eigen2cv( K1, cv_K1 );
  auto dist_coeffs1 = kwiver::arrows::ocv::get_ocv_dist_coeffs( cam1->intrinsics() );

  kv::matrix_3x3d K2 = cam2->intrinsics()->as_matrix();
  cv::Mat cv_K2;
  eigen2cv( K2, cv_K2 );
  auto dist_coeffs2 = kwiver::arrows::ocv::get_ocv_dist_coeffs( cam2->intrinsics() );

  LOG_INFO( m_logger, "Running stereo calibration..." );
  cv::Mat cv_R, cv_T, cv_E, cv_F;
  float rms = cv::stereoCalibrate( world_points, image_points1, image_points2, cv_K1, dist_coeffs1, cv_K2, dist_coeffs2,
                                   image_size, cv_R, cv_T, cv_E, cv_F, cv::CALIB_FIX_INTRINSIC );

  LOG_INFO( m_logger, "Stereo calibration complete, RMS error: " << rms );
  cv::Mat cv_R1, cv_P1, cv_R2, cv_P2, cv_Q;
  cv::stereoRectify( cv_K1, dist_coeffs1, cv_K2, dist_coeffs2, image_size, cv_R, cv_T, cv_R1, cv_R2, cv_P1, cv_P2, cv_Q,
                     cv::CALIB_ZERO_DISPARITY );
  cv::Mat rectMap11, rectMap12, rectMap21, rectMap22;
  cv::Mat img1r, img2r;
  cv::initUndistortRectifyMap( cv_K1, dist_coeffs1, cv_R1, cv_P1, image_size, CV_16SC2, rectMap11, rectMap12 );
  cv::initUndistortRectifyMap( cv_K2, dist_coeffs2, cv_R2, cv_P2, image_size, CV_16SC2, rectMap21, rectMap22 );

  LOG_INFO( m_logger, "Computing stereo rectification..." );
  LOG_INFO( m_logger, "Writing calibration files..." );
  write_stereo_calibration_file( cv_K1, cv_K2, dist_coeffs1, dist_coeffs2, cv_R, cv_T, cv_R1, cv_R2, cv_P1, cv_P2, cv_Q );

  // Also write JSON output using the shared calibrator if configured
  if( !m_output_json_file.empty() )
  {
    calibrate_stereo_cameras_result json_result;
    json_result.success = true;
    json_result.image_size = image_size;
    json_result.square_size = m_square_size;
    // Grid size derived from world points - find max x and y indices
    if( !world_points.empty() && !world_points[0].empty() )
    {
      float max_x = 0, max_y = 0;
      for( const auto& pt : world_points[0] )
      {
        max_x = std::max( max_x, pt.x );
        max_y = std::max( max_y, pt.y );
      }
      int grid_w = static_cast< int >( max_x / m_square_size ) + 1;
      int grid_h = static_cast< int >( max_y / m_square_size ) + 1;
      json_result.grid_size = cv::Size( grid_w, grid_h );
    }
    json_result.left.success = true;
    json_result.left.camera_matrix = cv_K1;
    json_result.left.dist_coeffs = cv::Mat( dist_coeffs1 );
    json_result.right.success = true;
    json_result.right.camera_matrix = cv_K2;
    json_result.right.dist_coeffs = cv::Mat( dist_coeffs2 );
    json_result.stereo_rms_error = rms;
    json_result.R = cv_R;
    json_result.T = cv_T;
    json_result.R1 = cv_R1;
    json_result.R2 = cv_R2;
    json_result.P1 = cv_P1;
    json_result.P2 = cv_P2;
    json_result.Q = cv_Q;

    m_calibrator.write_calibration_json( json_result, m_output_json_file );
    LOG_DEBUG( m_logger, "Wrote JSON calibration to: " << m_output_json_file );
  }

  // CALIBRATION QUALITY CHECK
  // because the output fundamental matrix implicitly
  // includes all the output information,
  // we can check the quality of calibration using the
  // epipolar geometry constraint: m2^t*F*m1=0
  double err = 0;
  int nPoints = 0;
  std::vector< cv::Vec3f > leftCameraLines, rightCameraLines;
  for( size_t i = 0; i < image_points1.size(); i++ )
  {
    int npt = (int) image_points1[i].size();
    cv::Mat leftImgPt, rightImgPt;
    leftImgPt = cv::Mat( image_points1[i] );
    rightImgPt = cv::Mat( image_points2[i] );
    cv::undistortPoints( leftImgPt, leftImgPt, cv_K1, dist_coeffs1, cv::Mat(), cv_K1 );
    cv::undistortPoints( rightImgPt, rightImgPt, cv_K2, dist_coeffs2, cv::Mat(), cv_K2 );
    cv::computeCorrespondEpilines( leftImgPt, 1, cv_F, leftCameraLines );
    cv::computeCorrespondEpilines( rightImgPt, 2, cv_F, rightCameraLines );

    for( int j = 0; j < npt; j++ )
    {
      double errij = std::fabs(
          image_points1[i][j].x * rightCameraLines[j][0] + image_points1[i][j].y * rightCameraLines[j][1] +
          rightCameraLines[j][2] ) + std::fabs(
          image_points2[i][j].x * leftCameraLines[j][0] + image_points2[i][j].y * leftCameraLines[j][1] +
          leftCameraLines[j][2] );
      err += errij;
    }
    nPoints += npt;
  }
  float epipolarError = err / nPoints;
  LOG_INFO( m_logger, "Quality check - average epipolar error: " << epipolarError );

  // Setup calibrate stereo camera1
  auto res_cam1 = std::make_shared< kv::simple_camera_perspective >();
  auto const dc_size1 = dist_coeffs1.size();

  Eigen::VectorXd dist_eig1( dist_coeffs1.size() );
  for( auto const i : kv::range::iota( dc_size1 ) )
  {
    dist_eig1[static_cast< int >( i )] = dist_coeffs1.at( i );
  }
  cv::cv2eigen( cv_K1, K1 );

  kv::camera_intrinsics_sptr cal1;
  cal1 = std::make_shared< kv::simple_camera_intrinsics >( K1, dist_eig1 );
  res_cam1->set_intrinsics( cal1 );

  // Setup calibrate stereo camera2
  cv::Mat rvec2;
  auto res_cam2 = std::make_shared< kv::simple_camera_perspective >();
  cv::Rodrigues( cv_R, rvec2 );
  Eigen::Vector3d rvec_eig2, tvec_eig2;
  auto const dc_size2 = dist_coeffs2.size();

  Eigen::VectorXd dist_eig2( dist_coeffs2.size() );
  for( auto const i : kv::range::iota( dc_size2 ) )
  {
    dist_eig2[static_cast< int >( i )] = dist_coeffs2[i];
  }
  cv::cv2eigen( rvec2, rvec_eig2 );
  cv::cv2eigen( cv_T, tvec_eig2 );
  cv::cv2eigen( cv_K2, K2 );
  kv::rotation_d rot2{ rvec_eig2 };
  res_cam2->set_rotation( rot2 );
  res_cam2->set_translation( tvec_eig2 );
  kv::camera_intrinsics_sptr cal2;
  cal2 = std::make_shared< kv::simple_camera_intrinsics >( K2, dist_eig2 );
  res_cam2->set_intrinsics( cal2 );

  *cam1 = *res_cam1;
  *cam2 = *res_cam2;

  LOG_DEBUG( m_logger, "Camera Essential :\n" << cv_E );
  LOG_DEBUG( m_logger, "Camera Fundamental :\n" << cv_F );
  LOG_DEBUG( m_logger, "Camera left translation :\n" << cam1->translation().transpose() );
  LOG_DEBUG( m_logger, "Camera left rotation :\n" << cam1->rotation().matrix() );
  LOG_DEBUG( m_logger, "Camera left intrinsics :\n" << cam1->intrinsics()->as_matrix() );
  for( auto dist_coef : kwiver::arrows::ocv::get_ocv_dist_coeffs( cam1->intrinsics() ) )
    LOG_DEBUG( m_logger, "Camera left distortion : " << dist_coef );

  LOG_DEBUG( m_logger, "Camera right translation :\n" << cam2->translation().transpose() );
  LOG_DEBUG( m_logger, "Camera right rotation :\n" << cam2->rotation().matrix() );
  LOG_DEBUG( m_logger, "Camera right intrinsics :\n" << cam2->intrinsics()->as_matrix() );
  for( auto dist_coef : kwiver::arrows::ocv::get_ocv_dist_coeffs( cam2->intrinsics() ) )
    LOG_DEBUG( m_logger, "Camera right distortion :" << dist_coef );
}


// ----------------------------------------------------------------------------
// Constructor
optimize_stereo_cameras::optimize_stereo_cameras()
  : d_( new priv )
{
  attach_logger( "viame.opencv.optimize_stereo_cameras" );

  d_->m_logger = logger();
}

// Destructor
optimize_stereo_cameras::~optimize_stereo_cameras() {}

// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
kv::config_block_sptr
optimize_stereo_cameras::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();
  config->set_value( "image_width", d_->m_image_width,
                     "sensor image width (0 to derive from data)" );
  config->set_value( "image_height", d_->m_image_height,
                     "sensor image height (0 to derive from data)" );
  config->set_value( "frame_count_threshold", d_->m_frame_count_threshold,
                     "max number of frames to use during optimization" );
  config->set_value( "output_calibration_directory", d_->m_output_calibration_directory,
                     "output path for the generated calibration files (OpenCV YAML format)" );
  config->set_value( "output_json_file", d_->m_output_json_file,
                     "output path for JSON calibration file (compatible with camera_rig_io)" );
  config->set_value( "square_size", d_->m_square_size,
                     "calibration pattern square size in world units (e.g., mm)" );

  return config;
}

// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
optimize_stereo_cameras::set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );
  d_->m_image_width = config->get_value< unsigned >( "image_width" );
  d_->m_image_height = config->get_value< unsigned >( "image_height" );
  d_->m_frame_count_threshold = config->get_value< unsigned >( "frame_count_threshold" );
  d_->m_output_calibration_directory = config->get_value< std::string >( "output_calibration_directory" );
  d_->m_output_json_file = config->get_value< std::string >( "output_json_file" );
  d_->m_square_size = config->get_value< double >( "square_size" );

  // Set logger on shared calibrator
  d_->m_calibrator.set_logger( d_->m_logger );
}

// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
optimize_stereo_cameras::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
// Optimize camera parameters given sets of landmarks and feature tracks
void
optimize_stereo_cameras
::optimize( kv::camera_map_sptr& cameras,
            kv::feature_track_set_sptr tracks,
            kv::landmark_map_sptr landmarks,
            kv::sfm_constraints_sptr constraints ) const
{
  // extract data from containers
  kv::camera_map::map_camera_t cams = cameras->cameras();

  if( cams.size() != 2 )
  {
    LOG_WARN( d_->m_logger, "This optimizer only works for a stereo setup." );
    return;
  }

  std::vector< kv::track_sptr > trks = tracks->tracks();
  std::vector< kv::track_sptr > trks1, trks2;
  kv::feature_track_set_sptr features1, features2;

  kv::landmark_map::map_landmark_t lms = landmarks->landmarks();
  kv::landmark_map::map_landmark_t lms1, lms2;
  kv::landmark_map_sptr landmarks1, landmarks2;

  auto features_half_size = trks.size() / 2;
  auto landmarks_half_size = lms.size() / 2;
  if( features_half_size % 2 || landmarks_half_size % 2 )
  {
    LOG_WARN( d_->m_logger, "Inconsistant features or landmarks number." );
    return;
  }

  for( size_t i_track = 0; i_track < trks.size(); i_track++ )
  {
    auto is_left_cam = i_track < features_half_size;
    size_t track_id = is_left_cam ? i_track : i_track - features_half_size;

    if( is_left_cam )
    {
      trks1.push_back( trks[i_track] );
      lms1[i_track] = lms[i_track];
    }
    else
    {
      trks[i_track]->set_id( track_id );
      trks2.push_back( trks[i_track] );
      lms2[track_id] = lms[i_track];
    }
  }

  features1 = std::make_shared< kv::feature_track_set >( trks1 );
  features2 = std::make_shared< kv::feature_track_set >( trks2 );
  landmarks1 = kv::landmark_map_sptr( new kv::simple_landmark_map( lms1 ) );
  landmarks2 = kv::landmark_map_sptr( new kv::simple_landmark_map( lms2 ) );

  optimize( cams, { features1, features2 }, { landmarks1, landmarks2 } );
}

void
optimize_stereo_cameras
::optimize( kwiver::vital::camera_map::map_camera_t cams,
            const std::vector< kwiver::vital::feature_track_set_sptr >& tracks,
            const std::vector< kwiver::vital::landmark_map_sptr >& landmarks ) const
{
  if( cams.size() != 2 || tracks.size() != 2 || landmarks.size() != 2 )
  {
    LOG_WARN( d_->m_logger, "This optimizer only works for a stereo setup." );
    return;
  }

  // Estimate params for each camera
  // use all fully detected ocv target to compute each calibration params
  d_->calibrate_camera( cams[0], tracks[0], landmarks[0], "left" );
  d_->calibrate_camera( cams[1], tracks[1], landmarks[1], "right" );

  // Refine calibration params and estimate R,T transform
  // use only stereo pair of fully detected ocv target to compute stereo calibration params
  d_->calibrate_stereo_camera( cams, tracks[0], tracks[1], landmarks[0], landmarks[1] );
}

void
optimize_stereo_cameras
::optimize( kwiver::vital::camera_perspective_sptr& camera,
            const std::vector< kwiver::vital::feature_sptr >& features,
            const std::vector< kwiver::vital::landmark_sptr >& landmarks,
            kwiver::vital::sfm_constraints_sptr constraints ) const
{
  return;
}

} // end namespace viame
