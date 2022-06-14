#include "ocv_rectified_stereo_disparity_map.h"

#include <vital/vital_config.h>
#include <vital/types/image_container.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/types/camera_map.h>
#include <vital/io/camera_io.h>
#include <vital/exceptions.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/eigen.hpp>
#include <arrows/ocv/camera_intrinsics.h>
#include <arrows/ocv/image_container.h>

namespace kv = kwiver::vital;

namespace viame {

class ocv_rectified_stereo_disparity_map::priv
{
public:

  std::string algorithm;
  int min_disparity;
  int num_disparities;
  int sad_window_size;
  int block_size;
  int speckle_window_size;
  int speckle_range;

  bool m_computed_rectification = false;
  kv::camera_map::map_camera_t m_cameras;
  std::string m_cameras_directory;

  // intermadiate variables
  cv::Mat m_rectification_map11;
  cv::Mat m_rectification_map12;
  cv::Mat m_rectification_map21;
  cv::Mat m_rectification_map22;

  kv::logger_handle_t m_logger;


#ifdef VIAME_OPENCV_VER_2
  cv::StereoBM algo;
#else
  cv::Ptr< cv::StereoMatcher > algo;
#endif

  priv()
    : algorithm( "BM" )
    , min_disparity( 0 )
    , num_disparities( 16 )
    , sad_window_size( 21 )
    , block_size( 3 )
    , speckle_window_size( 50 )
    , speckle_range( 5 )
    , m_computed_rectification(false)
    , m_cameras()
    , m_cameras_directory( "" )
  {}

  ~priv()
  {}

  kv::camera_map::map_camera_t
  load_camera_map(std::string const& camera1_name,
                  std::string const& camera2_name,
                  std::string const& cameras_dir)
  {
    kv::camera_map::map_camera_t cameras;

    try
    {
      cameras[0] = kv::read_krtd_file( camera1_name, cameras_dir );
      cameras[1] = kv::read_krtd_file( camera2_name, cameras_dir );
    }
    catch ( const kv::file_not_found_exception& )
    {
      VITAL_THROW( kv::invalid_data, "Calibration file not found" );
    }

    if ( cameras.empty() )
    {
      VITAL_THROW( kv::invalid_data, "No krtd files found" );
    }

    return cameras;
  }
};


ocv_rectified_stereo_disparity_map
::ocv_rectified_stereo_disparity_map()
: d( new priv() )
{
}


ocv_rectified_stereo_disparity_map
::~ocv_rectified_stereo_disparity_map()
{
}


// ---------------------------------------------------------------------------------------
kv::config_block_sptr
ocv_rectified_stereo_disparity_map
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "algorithm", d->algorithm, "Algorithm: BM or SGBM" );
  config->set_value( "min_disparity", d->min_disparity, "Min Disparity" );
  config->set_value( "num_disparities", d->num_disparities, "Disparity count" );
  config->set_value( "sad_window_size", d->sad_window_size, "SAD window size" );
  config->set_value( "block_size", d->block_size, "Block size" );
  config->set_value( "speckle_window_size", d->speckle_window_size, "Speckle Window Size" );
  config->set_value( "speckle_range", d->speckle_range, "Speckle Range" );

  config->set_value("input_cameras_directory", d->m_cameras_directory,
    "Path to a directory to read cameras from.");

  return config;
}

// ---------------------------------------------------------------------------------------
void ocv_rectified_stereo_disparity_map
::set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->algorithm = config->get_value< std::string >( "algorithm" );
  d->min_disparity = config->get_value< int >( "min_disparity" );
  d->num_disparities = config->get_value< int >( "num_disparities" );
  d->sad_window_size = config->get_value< int >( "sad_window_size" );
  d->block_size = config->get_value< int >( "block_size" );
  d->speckle_window_size = config->get_value< int >( "speckle_window_size" );
  d->speckle_range = config->get_value< int >( "speckle_range" );

  d->m_computed_rectification = false;
  d->m_cameras_directory = config->get_value< std::string >( "input_cameras_directory" );

  d->m_cameras = d->load_camera_map("camera1", "camera2", d->m_cameras_directory);

#ifdef VIAME_OPENCV_VER_2
  if( d->algorithm == "BM" )
  {
    d->algo.init( 0, d->num_disparities, d->sad_window_size );
  }
  else if( d->algorithm == "SGBM" )
  {
    throw std::runtime_error( "Unable to use type SGBM with OpenCV 2" );
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type " + d->algorithm );
  }
#else
  if( d->algorithm == "BM" )
  {
    d->algo = cv::StereoBM::create( d->num_disparities, d->sad_window_size );
    d->algo->setSpeckleWindowSize(d->speckle_window_size);
    d->algo->setSpeckleRange (d->speckle_range);
  }
  else if( d->algorithm == "SGBM" )
  {
    d->algo = cv::StereoSGBM::create( d->min_disparity, d->num_disparities, d->block_size );
    d->algo->setSpeckleWindowSize(d->speckle_window_size);
    d->algo->setSpeckleRange (d->speckle_range);
  }
  else
  {
    throw std::runtime_error( "Invalid algorithm type " + d->algorithm );
  }
#endif
}


// ---------------------------------------------------------------------------------------
bool ocv_rectified_stereo_disparity_map
::check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// ---------------------------------------------------------------------------------------
kv::image_container_sptr ocv_rectified_stereo_disparity_map
::compute( kv::image_container_sptr left_image,
           kv::image_container_sptr right_image ) const
{
  if(d->m_cameras.size() != 2)
  {
    LOG_WARN(d->m_logger, "Only works with two cameras as inputs.");
    return kwiver::vital::image_container_sptr();
  }

  if(left_image->get_image().size() != right_image->get_image().size())
  {
    LOG_WARN(d->m_logger, "Inconsistent left/right images size.");
    return kwiver::vital::image_container_sptr();
  }

  // Load cameras and compute needed rectification matrix
  if( !d->m_computed_rectification )
  {
    kv::simple_camera_perspective_sptr cam1, cam2;
    cam1 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(d->m_cameras[0]);
    cam2 = std::dynamic_pointer_cast<kv::simple_camera_perspective>(d->m_cameras[1]);

    kv::matrix_3x3d K1 = cam1->intrinsics()->as_matrix();
    cv::Mat cv_K1;
    cv::eigen2cv( K1, cv_K1 );
    auto dist_coeffs1 = kwiver::arrows::ocv::get_ocv_dist_coeffs( cam1->intrinsics() );

    kv::matrix_3x3d K2 = cam2->intrinsics()->as_matrix();
    cv::Mat cv_K2;
    cv::eigen2cv( K2, cv_K2 );
    auto dist_coeffs2 = kwiver::arrows::ocv::get_ocv_dist_coeffs( cam2->intrinsics() );

    kv::matrix_3x3d R = cam2->rotation().matrix();
    kv::vector_3d T = cam2->translation();
    cv::Mat cv_R, cv_T;
    cv::eigen2cv( R, cv_R );
    cv::eigen2cv( T, cv_T );

    cv::Size img_size = cv::Size(left_image->get_image().width(),left_image->get_image().height());
    cv::Mat cv_R1, cv_P1, cv_R2, cv_P2, cv_Q;
    cv::stereoRectify(cv_K1, dist_coeffs1,
                      cv_K2, dist_coeffs2,
                      img_size,
                      cv_R, cv_T,
                      cv_R1, cv_R2, cv_P1, cv_P2, cv_Q,
                      cv::CALIB_ZERO_DISPARITY);

    // compute rectification maps to be used for each new stereo frames
    std::cout << "cv_R1\n" << cv_R1 << std::endl;
    std::cout << "cv_P1\n" << cv_P1 << std::endl;
    std::cout << "cv_R2\n" << cv_R2 << std::endl;
    std::cout << "cv_P2\n" << cv_P2 << std::endl;
    std::cout << "cv_Q\n" << cv_Q << std::endl;
    cv::initUndistortRectifyMap(cv_K1, dist_coeffs1, cv_R1, cv_P1,
                                img_size, CV_16SC2, d->m_rectification_map11, d->m_rectification_map12);
    cv::initUndistortRectifyMap(cv_K2, dist_coeffs2, cv_R2, cv_P2,
                                img_size, CV_16SC2, d->m_rectification_map21, d->m_rectification_map22);

    if(!d->m_rectification_map11.empty() ||
       !d->m_rectification_map12.empty() ||
       !d->m_rectification_map21.empty() ||
       !d->m_rectification_map22.empty())
    {
      d->m_computed_rectification = true;
    }
  }

  // apply rectification then compute depth map
  cv::Mat ocv1 = kwiver::arrows::ocv::image_container::vital_to_ocv( left_image->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv2 = kwiver::arrows::ocv::image_container::vital_to_ocv( right_image->get_image(),
    kwiver::arrows::ocv::image_container::BGR_COLOR  );

  cv::Mat ocv1_gray, ocv2_gray;

  if( ocv1.channels() > 1 )
  {
    cvtColor(ocv1, ocv1_gray, CV_BGR2GRAY);
    cvtColor(ocv2, ocv2_gray, CV_BGR2GRAY);
  }
  else
  {
    ocv1_gray = ocv1;
    ocv2_gray = ocv2;
  }

  cv::Mat img1r, img2r;
  cv::remap(ocv1_gray, img1r, d->m_rectification_map11, d->m_rectification_map12, cv::INTER_LINEAR);
  cv::remap(ocv2_gray, img2r, d->m_rectification_map21, d->m_rectification_map22, cv::INTER_LINEAR);

  // compute disparity map
  cv::Mat disparity_map;

#ifdef VIAME_OPENCV_VER_2
  d->algo( ocv1_gray, ocv2_gray, disparity_map );
#else
  d->algo->compute( img1r, img2r, disparity_map );
#endif

  // Convert 16 bits fixed-point disparity map (where each disparity value has 4 fractional bits)
  // from  StereoBM or StereoSGBM
  // cf https://docs.opencv.org/3.4/d2/d6e/classcv_1_1StereoMatcher.html
  cv::Mat disparity_map_float;
  disparity_map.convertTo(disparity_map_float, CV_32F);
  disparity_map_float *= std::pow(2.0, -4.0);

  return kv::image_container_sptr( new kwiver::arrows::ocv::image_container( disparity_map_float,
    kwiver::arrows::ocv::image_container::BGR_COLOR ) );
}

} //end namespace viame
