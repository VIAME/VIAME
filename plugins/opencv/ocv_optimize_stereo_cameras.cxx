/**
* \file
* \brief Header defining Opencv algorithm implementation of camera optimization to optimize stereo setup.
*/

#include "ocv_optimize_stereo_cameras.h"

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
#include <opencv2/imgproc.hpp>


namespace kv = kwiver::vital;

namespace viame {

// Private implementation class
class ocv_optimize_stereo_cameras::priv
{
public:
  // Constructor
  priv() {}
  
  // Destructor
  ~priv() {}
  
  unsigned m_image_width;
  unsigned m_image_height;
  
  kv::logger_handle_t m_logger;
  
  kv::feature_track_set_sptr
  merge_features_track(kv::feature_track_set_sptr feature_track,
                       kv::feature_track_set_sptr feature_track1,
                       kv::feature_track_set_sptr feature_track2);
  
  void
  calibrate_camera(kv::camera_sptr camera,
                   const kv::feature_track_set_sptr features, 
                   const kv::landmark_map_sptr landmarks);
  
  void
  calibrate_stereo_camera( kv::camera_map::map_camera_t cameras,
                           kv::feature_track_set_sptr features1,
                           kv::feature_track_set_sptr features2, 
                           kv::landmark_map_sptr landmarks1,
                           kv::landmark_map_sptr landmarks2);
}; // end class ocv_optimize_stereo_cameras::priv

kv::feature_track_set_sptr
ocv_optimize_stereo_cameras::priv
::merge_features_track(kv::feature_track_set_sptr feature_track,
                       kv::feature_track_set_sptr feature_track1,
                       kv::feature_track_set_sptr feature_track2)
{
  std::vector<kv::track_id_t> all_track_ids;
  std::vector<kv::track_sptr> all_tracks1 = feature_track1->tracks();
  std::vector<kv::track_sptr> all_tracks2 = feature_track2->tracks();
  for (auto track : all_tracks1 )
    all_track_ids.push_back(track->id());
  
  kv::track_id_t max_track1_id = *std::max_element(all_track_ids.begin(), all_track_ids.end());
  
  for (auto track : all_tracks2 )
    all_track_ids.push_back(track->id() + max_track1_id);
  
  std::vector<kv::track_sptr> all_tracks;
  all_tracks.insert( all_tracks.end(), all_tracks1.begin(), all_tracks1.end() );
  all_tracks.insert( all_tracks.end(), all_tracks2.begin(), all_tracks2.end() );
  
  std::vector<kv::track_sptr> tracks;
  for (unsigned i = 0 ; i < all_tracks.size() ; i++ )
  {
    kv::track_sptr t = kv::track::create();
    t->set_id( all_track_ids[i] );
    tracks.push_back( t );
    
    for ( auto state : *all_tracks[i] | kv::as_feature_track )
    {
      auto fts = std::make_shared<kv::feature_track_state>(state->frame());
      fts->feature = std::make_shared<kv::feature_d>( state->feature->loc() );
      fts->inlier = true;
      t->append( fts );
    }
  }
  
  return std::make_shared<kv::feature_track_set>( tracks );
}

void
ocv_optimize_stereo_cameras::priv
::calibrate_camera(kv::camera_sptr camera,
                   const kv::feature_track_set_sptr features, 
                   const kv::landmark_map_sptr landmarks)
{
  kv::simple_camera_perspective_sptr cam;
  cam = std::dynamic_pointer_cast<kv::simple_camera_perspective>(camera);
  
  if(!cam)
  {
    LOG_WARN(m_logger, "Unable to get the camera.");
    return;
  }
  
  std::vector<kv::track_sptr> tracks = features->tracks();
  
//  std::vector< kv::vector_2d > image_points;
//  std::vector< kv::vector_3d > world_points;
//  for ( auto track : tracks )
//  {
//    for ( auto state : *track | kv::as_feature_track )
//    {
//      if(!landmarks->landmarks().count(track->id()))
//        continue;
      
//      kv::vector_3d pt = landmarks->landmarks()[track->id()]->loc();
//      world_points.push_back(pt);

//      auto center = state->feature->loc();
//      std::cout << center << std::endl;
//      image_points.push_back(kv::vector_2d(center[0], center[1]));
//    }
//  }
  
  // get features and landmarks coorinates and reorder it into image_points and world_points
  std::vector<std::vector< cv::Point2f >> image_pts_temp;
  std::vector<std::vector< cv::Point3f >> world_pts_temp;
  for ( auto track : tracks )
  {
    std::vector< cv::Point2f > pts2d;
    std::vector< cv::Point3f > pts3d;
  
    for ( auto state : *track | kv::as_feature_track )
    {
      if(!landmarks->landmarks().count(track->id()))
        continue;
      
      auto center = state->feature->loc();
      pts2d.push_back(cv::Point2d(center[0], center[1]));
      
      kv::vector_3d pt = landmarks->landmarks()[track->id()]->loc();
      cv::Point3d w(pt[0], pt[1], pt[2]);
      pts3d.push_back(w);
    }
    image_pts_temp.push_back(pts2d);
    world_pts_temp.push_back(pts3d);
  }
  
  if(!image_pts_temp.size() || 
      image_pts_temp.size() != world_pts_temp.size())
  {
    LOG_WARN( m_logger, "Unable calibrate stereo setup." );
    return;
  }
  
  // Reorder image and world points to vector of vector of [frame][track]
  unsigned track_count = image_pts_temp.size();
  unsigned frame_count = image_pts_temp[0].size();
  LOG_DEBUG(m_logger, "track_count : " << track_count);
  LOG_DEBUG(m_logger, "frame_count : " << frame_count);
  
  if(frame_count > 50)
  {
    LOG_WARN(m_logger, "Detected target number too high to launch calibration process.");
    return;
  }
  
  std::vector<std::vector< cv::Point2f >> image_points(frame_count);
  std::vector<std::vector< cv::Point3f >> world_points(frame_count);
  for(unsigned f_id = 0; f_id < frame_count; f_id++)
  {
    for(unsigned t_id = 0; t_id < track_count; t_id++)
    {
      image_points[f_id].push_back(image_pts_temp[t_id][f_id]);
      world_points[f_id].push_back(world_pts_temp[t_id][f_id]);
    }
  }
  
  // Init left/right camera matrix
  cv::Size image_size = cv::Size(m_image_width, m_image_height);
  cv::Mat cv_K1 = cv::Mat(3, 3, CV_64F);
  std::vector<double> dist_coeffs = cv::Mat(1, 7, CV_64F);
  cv_K1 = cv::initCameraMatrix2D(world_points, image_points, image_size, 0);
  
  // Run intrinsic calibration of left camera
  cv::Mat rvec1, tvec1;

  // Performing camera calibration by passing the value of known 3D world points (objpoints)
  // and corresponding pixel coordinates of the detected corners (imgpoints)
  std::cout << "Running intrinsic calibration of left camera..." << std::endl;
  cv::calibrateCamera(world_points, image_points, 
                      image_size, 
                      cv_K1, dist_coeffs, rvec1, tvec1);
  
  std::cout << cv_K1 << std::endl;
  for(auto dcoef : dist_coeffs)
    std::cout << dcoef << std::endl;
  
  // Setup calibrate stereo camera1
  kv::matrix_3x3d K1;
  auto res_cam1 = std::make_shared<kv::simple_camera_perspective>();
  auto const dc_size1 = dist_coeffs.size();

  Eigen::VectorXd dist_eig1( dist_coeffs.size() );
  for( auto const i : kv::range::iota( dc_size1 ) )
  {
    dist_eig1[ static_cast< int >( i ) ] = dist_coeffs.at( i );
  }
  cv::cv2eigen( cv_K1, K1 );

  kv::camera_intrinsics_sptr cal1;
  cal1 = std::make_shared< kv::simple_camera_intrinsics >( K1, dist_eig1 );
  res_cam1->set_intrinsics( cal1 );
  
  // copy camera params
  *cam = *res_cam1;
  
  LOG_DEBUG(m_logger, "Estimated camera center :\n" << cam->translation().transpose());
  LOG_DEBUG(m_logger, "Estimated camera rotation :\n" << cam->rotation().matrix());
}


void
ocv_optimize_stereo_cameras::priv
::calibrate_stereo_camera( kv::camera_map::map_camera_t cameras,
                           kv::feature_track_set_sptr features1,
                           kv::feature_track_set_sptr features2, 
                           kv::landmark_map_sptr landmarks1,
                           kv::landmark_map_sptr landmarks2)
{
  if(cameras.size() != 2)
  {
    LOG_WARN(m_logger, "Only works with two cameras as inputs.");
    return;
  }

  kv::simple_camera_perspective_sptr cam1, cam2;
  cam1 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(cameras[0]);
  cam2 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(cameras[1]);
  cv::Size image_size = cv::Size(m_image_width, m_image_height);
  
  kv::landmark_map::map_landmark_t lms1 = landmarks1->landmarks();
  kv::landmark_map::map_landmark_t lms2 = landmarks2->landmarks();
  
  std::vector<std::vector< cv::Point2f >> image_pts1_temp, image_pts2_temp;
  std::vector<std::vector< cv::Point3f >> world_pts_temp;
  
  // reorder image and world points
  for(auto fts1 : features1->tracks())
  {
    for(auto fts2 : features2->tracks())
    {
      // only keep features with the same id
      if(fts1->id() != fts2->id())
      {
        continue;
      }
      
      std::vector< cv::Point2f > pts2d1, pts2d2;
      std::vector< cv::Point3f > pts3d;
      
      for ( auto state1 : *fts1 | kv::as_feature_track )
      {
        for ( auto state2 : *fts2 | kv::as_feature_track )
        {
          bool world1_found = false;
          bool world2_found = false;
          bool world_found = false;
          
          // Only keep camera1 and 2 points with the same frame id
          if(state1->frame() != state2->frame())
            continue;
          
          world1_found = lms1.count(fts1->id());
          world2_found = lms2.count(fts2->id());
          world_found = false;
          
          cv::Point3d w1, w2;
          if(world1_found)
          {
            w1 = cv::Point3d(lms1[fts1->id()]->loc()[0],
                 lms1[fts1->id()]->loc()[1],
                 lms1[fts1->id()]->loc()[2]);
          }
          
          if(world2_found)
          {
            w2 = cv::Point3d(lms2[fts2->id()]->loc()[0],
                 lms2[fts2->id()]->loc()[1],
                 lms2[fts2->id()]->loc()[2]);
          }
          
          // Check if world points are valid
          if(world1_found && world2_found)
          {
            world_found = w1 == w2 ? true : false;
          }
          else if(world1_found || world2_found)
          {
            world_found = true;
          }
          
          if(world_found)
          {
            auto center1 = state1->feature->loc();
            auto center2 = state2->feature->loc();
            pts2d1.push_back(cv::Point2d(center1[0], center1[1]));
            pts2d2.push_back(cv::Point2d(center2[0], center2[1]));
            
            if(world1_found)
            {
              pts3d.push_back(w1);
            }
            else if (world2_found) 
            {
              pts3d.push_back(w2);
            }
          }
        }
      }
      image_pts1_temp.push_back(pts2d1);
      image_pts2_temp.push_back(pts2d2);
      world_pts_temp.push_back(pts3d);
    }
  }
  
  
  if(!image_pts1_temp.size() || 
     !image_pts2_temp.size() ||
     image_pts1_temp[0].size() != image_pts2_temp[0].size() ||
     image_pts1_temp.size() != image_pts2_temp.size() ||
     image_pts1_temp.size() != world_pts_temp.size())
  {
    LOG_WARN( m_logger, "Unable calibrate stereo setup." );
    return;
  }
  
  // Reorder image and world points to vector of vector of [frame][track]
  unsigned track_count = image_pts1_temp.size();
  unsigned frame_count = image_pts1_temp[0].size();
  LOG_DEBUG(m_logger, "track_count : " << track_count);
  LOG_DEBUG(m_logger, "frame_count : " << frame_count);

  if(frame_count > 50)
  {
    LOG_WARN(m_logger, "Detected target number too high to launch calibration process.");
    return;
  }
  
  std::vector<std::vector< cv::Point2f >> image_points1(frame_count);
  std::vector<std::vector< cv::Point2f >> image_points2(frame_count);
  std::vector<std::vector< cv::Point3f >> world_points(frame_count);
  for(unsigned f_id = 0; f_id < frame_count; f_id++)
  {
    for(unsigned t_id = 0; t_id < track_count; t_id++)
    {
      image_points1[f_id].push_back(image_pts1_temp[t_id][f_id]);
      image_points2[f_id].push_back(image_pts2_temp[t_id][f_id]);
      world_points[f_id].push_back(world_pts_temp[t_id][f_id]);
    }
  }
  
  LOG_DEBUG(m_logger, "image_width : " << m_image_width);
  LOG_DEBUG(m_logger, "image_height : " << m_image_height);
  LOG_DEBUG(m_logger, "image_points1.size() : " << image_points1.size());
  LOG_DEBUG(m_logger, "image_points2.size() : " << image_points1.size());
  LOG_DEBUG(m_logger, "world_points.size() : " << image_points1.size());
  
  
  // Get mono camera calibration params
  LOG_DEBUG(m_logger, "Launch camera calibration...");
  
  kv::matrix_3x3d K1 = cam1->intrinsics()->as_matrix();
  cv::Mat cv_K1;
  eigen2cv( K1, cv_K1 );
  auto dist_coeffs1 = kwiver::arrows::ocv::get_ocv_dist_coeffs( cam1->intrinsics() );
  
  kv::matrix_3x3d K2 = cam2->intrinsics()->as_matrix();
  cv::Mat cv_K2;
  eigen2cv( K2, cv_K2 );
  auto dist_coeffs2 = kwiver::arrows::ocv::get_ocv_dist_coeffs( cam2->intrinsics() );
  
  LOG_DEBUG(m_logger, "Launch stereo calibration...");
  cv::Mat cv_R, cv_T, cv_E, cv_F;
  float rms = cv::stereoCalibrate(world_points, image_points1, image_points2,
                                  cv_K1, dist_coeffs1,
                                  cv_K2, dist_coeffs2,
                                  image_size, 
                                  cv_R, cv_T, cv_E, cv_F,
                                  cv::CALIB_ZERO_TANGENT_DIST +
                                  cv::CALIB_USE_INTRINSIC_GUESS +
                                  cv::CALIB_RATIONAL_MODEL,
                                  cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 200, 1e-5) );
  
  LOG_DEBUG(m_logger, "Stereo calibration run with RMS error = " << rms);
  
  // CALIBRATION QUALITY CHECK
  // because the output fundamental matrix implicitly
  // includes all the output information,
  // we can check the quality of calibration using the
  // epipolar geometry constraint: m2^t*F*m1=0
  double err = 0;
  int nPoints = 0;
  std::vector<cv::Vec3f> leftCameraLines, rightCameraLines;
  for( int i = 0; i < image_points1.size(); i++ )
  {
    int npt = (int)image_points1[i].size();
    cv::Mat leftImgPt, rightImgPt;
    leftImgPt = cv::Mat(image_points1[i]);
    rightImgPt = cv::Mat(image_points2[i]);
    cv::undistortPoints(leftImgPt, leftImgPt, cv_K1, dist_coeffs1, cv::Mat(), cv_K1);
    cv::undistortPoints(rightImgPt, rightImgPt, cv_K2, dist_coeffs2, cv::Mat(), cv_K2);
    cv::computeCorrespondEpilines(leftImgPt, 1, cv_F, leftCameraLines);
    cv::computeCorrespondEpilines(rightImgPt, 2, cv_F, rightCameraLines);

    for( int j = 0; j < npt; j++ )
    {
      double errij = fabs(image_points1[i][j].x * rightCameraLines[j][0] +
                          image_points1[i][j].y * rightCameraLines[j][1] +
                          rightCameraLines[j][2]) +
                     fabs(image_points2[i][j].x * leftCameraLines[j][0] +
                          image_points2[i][j].y * leftCameraLines[j][1] +
                          leftCameraLines[j][2]);
      err += errij;
    }
    nPoints += npt;
  }
  float epipolarError = err/nPoints;
  std::cout << "average epipolar err = " <<  epipolarError << std::endl; 
  
  
  // Setup calibrate stereo camera1
  auto res_cam1 = std::make_shared<kv::simple_camera_perspective>();
  auto const dc_size1 = dist_coeffs1.size();

  Eigen::VectorXd dist_eig1( dist_coeffs1.size() );
  for( auto const i : kv::range::iota( dc_size1 ) )
  {
    dist_eig1[ static_cast< int >( i ) ] = dist_coeffs1.at( i );
  }
  cv::cv2eigen( cv_K1, K1 );

  kv::camera_intrinsics_sptr cal1;
  cal1 = std::make_shared< kv::simple_camera_intrinsics >( K1, dist_eig1 );
  res_cam1->set_intrinsics( cal1 );
  
  // Setup calibrate stereo camera2
  cv::Mat rvec2;
  auto res_cam2 = std::make_shared<kv::simple_camera_perspective>();
  cv::Rodrigues(cv_R, rvec2);
  Eigen::Vector3d rvec_eig2, tvec_eig2;
  auto const dc_size2 = dist_coeffs2.size();
  
  Eigen::VectorXd dist_eig2( dist_coeffs2.size() );
  for( auto const i : kv::range::iota( dc_size2 ) )
  {
    dist_eig2[ static_cast< int >( i ) ] = dist_coeffs2[ i ];
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
  
  LOG_DEBUG(m_logger, "Camera Essential :\n" << cv_E);
  LOG_DEBUG(m_logger, "Camera Fundamental :\n" << cv_F);
  LOG_DEBUG(m_logger, "Camera left translation :\n" << cam1->translation().transpose());
  LOG_DEBUG(m_logger, "Camera left rotation :\n" << cam1->rotation().matrix());
  LOG_DEBUG(m_logger, "Camera left intrinsics :\n" << cam1->intrinsics()->as_matrix());
  for(auto dist_coef : kwiver::arrows::ocv::get_ocv_dist_coeffs( cam1->intrinsics() ))
    LOG_DEBUG(m_logger, "Camera left distortion : " << dist_coef);
  
  LOG_DEBUG(m_logger, "Camera right translation :\n" << cam2->translation().transpose());
  LOG_DEBUG(m_logger, "Camera right rotation :\n" << cam2->rotation().matrix());
  LOG_DEBUG(m_logger, "Camera right intrinsics :\n" << cam2->intrinsics()->as_matrix());
  for(auto dist_coef : kwiver::arrows::ocv::get_ocv_dist_coeffs( cam2->intrinsics() ))
    LOG_DEBUG(m_logger, "Camera left distortion :" << dist_coef);
}



// ----------------------------------------------------------------------------
// Constructor
ocv_optimize_stereo_cameras::
ocv_optimize_stereo_cameras()
  : d_( new priv )
{
  attach_logger( "viame.opencv.ocv_optimize_stereo_cameras" );

  d_->m_logger = logger();
}

// Destructor
ocv_optimize_stereo_cameras::
  ~ocv_optimize_stereo_cameras()
{}

// ----------------------------------------------------------------------------
// Get this algorithm's \link vital::config_block configuration block \endlink
kv::config_block_sptr
ocv_optimize_stereo_cameras::
get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();
  config->set_value( "image_width", d_->m_image_width,
                     "sensor image width" );
  config->set_value( "image_height", d_->m_image_height,
                     "sensor image height" );
  
  return config;
}

// ----------------------------------------------------------------------------
// Set this algorithm's properties via a config block
void
ocv_optimize_stereo_cameras::
set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );
  d_->m_image_width = config->get_value< double >( "image_width" );
  d_->m_image_height = config->get_value< double >( "image_height" );
}

// ----------------------------------------------------------------------------
// Check that the algorithm's currently configuration is valid
bool
ocv_optimize_stereo_cameras::
check_configuration( kv::config_block_sptr config ) const
{
  return true;
}

// ----------------------------------------------------------------------------
// Optimize camera parameters given sets of landmarks and feature tracks
void
ocv_optimize_stereo_cameras::
optimize(  kv::camera_map_sptr & cameras,
           kv::feature_track_set_sptr tracks,
           kv::landmark_map_sptr landmarks,
           kv::sfm_constraints_sptr constraints) const
{
  // extract data from containers
  kv::camera_map::map_camera_t cams = cameras->cameras();
  
  if(cams.size() != 2)
  {
    LOG_WARN(d_->m_logger, "This optimizer only works for a stereo setup.");
    return;
  }
  
  std::vector<kv::track_sptr> trks = tracks->tracks();
  std::vector<kv::track_sptr> trks1, trks2;
  kv::feature_track_set_sptr features1, features2;
  
  kv::landmark_map::map_landmark_t lms = landmarks->landmarks();
  kv::landmark_map::map_landmark_t lms1, lms2;
  kv::landmark_map_sptr landmarks1, landmarks2;
  
  
  std::size_t features_half_size = trks.size() / 2;
  std::size_t landmarks_half_size = lms.size() / 2;
  if(features_half_size%2 || landmarks_half_size%2)
  {
    LOG_WARN(d_->m_logger, "Inconsistant features or landmarks number.");
    return;
  }
  
  // Split features and their id for camera1
  for (unsigned i = 0 ; i < features_half_size ; i++ )
  {
    kv::track_sptr t = kv::track::create();
    t->set_id( trks[i]->id() );
    trks1.push_back( t );
    
    for ( auto state : *trks[i] | kv::as_feature_track )
    {
      auto fts = std::make_shared<kv::feature_track_state>(state->frame());
      fts->feature = std::make_shared<kv::feature_d>( state->feature->loc() );
      fts->inlier = true;
      t->append( fts );
    }
  }
  features1 = std::make_shared<kv::feature_track_set>( trks1 );
  
  // Split features and their id for camera2
  for (unsigned i = features_half_size ; i < 2*features_half_size ; i++ )
  {
    kv::track_sptr t = kv::track::create();
    t->set_id( trks[i]->id() - trks1.back()->id() - 1 );
    trks2.push_back( t );
    
    for ( auto state : *trks[i] | kv::as_feature_track )
    {
      auto fts = std::make_shared<kv::feature_track_state>(state->frame());
      fts->feature = std::make_shared<kv::feature_d>( state->feature->loc() );
      fts->inlier = true;
      t->append( fts );
    }
  }
  
  features2 = std::make_shared<kv::feature_track_set>( trks2 );
  
  // Split landmarks of each camera2
  for ( kv::landmark_map::map_landmark_t::iterator it=lms.begin(); it!=std::next(lms.begin(), landmarks_half_size); ++it)
    lms1[it->first] = it->second;
  
  for ( kv::landmark_map::map_landmark_t::iterator it=std::next(lms.begin(), landmarks_half_size); it!=lms.end(); ++it)
    lms2[it->first - lms1.rbegin()->first - 1] = it->second;
   
  landmarks1 = kv::landmark_map_sptr(new kv::simple_landmark_map( lms1 ));
  landmarks2 = kv::landmark_map_sptr(new kv::simple_landmark_map( lms2 ));
  
  // Estimate params for each camera
  // use all fully detected ocv target to compute each calibration params
  d_->calibrate_camera(cams[0], features1, landmarks1);
  d_->calibrate_camera(cams[1], features2, landmarks2);
  
  // Refine calibration params and estimate R,T transform
  // use only stereo pair of fully detected ocv target to compute stereo calibration params
  d_->calibrate_stereo_camera(cams, features1, features2, landmarks1, landmarks2);
}

void ocv_optimize_stereo_cameras::
optimize(  kwiver::vital::camera_perspective_sptr &camera,
           const std::vector<kwiver::vital::feature_sptr> &features,
           const std::vector<kwiver::vital::landmark_sptr> &landmarks,
           kwiver::vital::sfm_constraints_sptr constraints) const
{
  return;
}

} // end namespace viame
