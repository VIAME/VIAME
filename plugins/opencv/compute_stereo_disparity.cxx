/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "compute_stereo_disparity.h"
#include "calibrate_stereo_cameras.h"

#include <vital/vital_config.h>
#include <vital/types/image_container.h>
#include <vital/types/image.h>
#include <vital/exceptions.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/ximgproc.hpp>

#include <arrows/ocv/image_container.h>

namespace kv = kwiver::vital;

namespace viame {

class compute_stereo_disparity::priv
{
public:
  // Algorithm selection
  std::string algorithm{ "SGBM" };

  // SGBM/BM parameters
  int min_disparity{ 0 };
  int num_disparities{ 128 };
  int sad_window_size{ 21 };
  int block_size{ 5 };
  int speckle_window_size{ 100 };
  int speckle_range{ 32 };

  // Output format: "raw", "float32", or "uint16_scaled"
  bool compute_depth{ false };
  bool export_as_alpha{ false };
  bool output_rectified{ true };
  std::string output_format{ "raw" };
  double uint16_scale_factor{ 256.0 };

  // WLS filtering
  bool use_wls_filter{ false };
  double wls_lambda{ 8000.0 };
  double wls_sigma{ 1.5 };

  // Rectification (optional - if calibration_file is set)
  std::string calibration_file;
  bool rectify_images{ false };
  mutable bool rectification_computed{ false };
  mutable cv::Mat rectification_map_left_x;
  mutable cv::Mat rectification_map_left_y;
  mutable cv::Mat rectification_map_right_x;
  mutable cv::Mat rectification_map_right_y;
  mutable cv::Mat unrectification_map_x;
  mutable cv::Mat unrectification_map_y;

  // Calibration data (loaded if calibration_file is set). Mutable because
  // single-file formats carry no rectification transforms, so those are filled
  // in lazily once the first image gives us an image size.
  mutable calibrate_stereo_cameras_result calibration;
  calibrate_stereo_cameras calibrator;

  // Stereo matchers
  cv::Ptr<cv::StereoMatcher> left_matcher;
  cv::Ptr<cv::StereoMatcher> right_matcher;
  cv::Ptr<cv::ximgproc::DisparityWLSFilter> wls_filter;

  kv::logger_handle_t logger;

  void create_matchers()
  {
    if( algorithm == "BM" )
    {
      left_matcher = cv::StereoBM::create( num_disparities, sad_window_size );
      left_matcher->setMinDisparity( min_disparity );
      left_matcher->setSpeckleWindowSize( speckle_window_size );
      left_matcher->setSpeckleRange( speckle_range );
    }
    else if( algorithm == "SGBM" )
    {
      int p1 = 8 * block_size * block_size;
      int p2 = 32 * block_size * block_size;
      left_matcher = cv::StereoSGBM::create(
        min_disparity, num_disparities, block_size,
        p1, p2,
        1,    // disp12MaxDiff
        0,    // preFilterCap
        10,   // uniquenessRatio
        speckle_window_size,
        speckle_range,
        cv::StereoSGBM::MODE_SGBM_3WAY );
    }
    else
    {
      throw std::runtime_error( "Invalid algorithm type: " + algorithm );
    }

    if( use_wls_filter )
    {
      wls_filter = cv::ximgproc::createDisparityWLSFilter( left_matcher );
      wls_filter->setLambda( wls_lambda );
      wls_filter->setSigmaColor( wls_sigma );
      right_matcher = cv::ximgproc::createRightMatcher( left_matcher );
    }
    else
    {
      wls_filter.release();
      right_matcher.release();
    }
  }

  void load_calibration()
  {
    if( calibration_file.empty() )
    {
      rectify_images = false;
      return;
    }

    if( !calibrator.load_calibration( calibration_file, calibration ) )
    {
      VITAL_THROW( kv::invalid_data,
        "Failed to load calibration from: " + calibration_file );
    }
    rectify_images = true;
    rectification_computed = false;
  }

  void compute_rectification_maps( const cv::Size& img_size ) const
  {
    if( rectification_computed )
    {
      return;
    }

    if( !calibrate_stereo_cameras::ensure_rectification( calibration, img_size ) )
    {
      VITAL_THROW( kv::invalid_data,
        "Calibration lacks the rectification transforms needed for " +
        std::to_string( img_size.width ) + "x" +
        std::to_string( img_size.height ) + " images: " + calibration_file );
    }

    cv::initUndistortRectifyMap(
      calibration.left.camera_matrix, calibration.left.dist_coeffs,
      calibration.R1, calibration.P1,
      img_size, CV_32FC1,
      rectification_map_left_x, rectification_map_left_y );

    cv::initUndistortRectifyMap(
      calibration.right.camera_matrix, calibration.right.dist_coeffs,
      calibration.R2, calibration.P2,
      img_size, CV_32FC1,
      rectification_map_right_x, rectification_map_right_y );

    cv::Mat original_grid( img_size.height * img_size.width, 1, CV_32FC2 );
    for( int y = 0; y < img_size.height; y++ )
    {
      for( int x = 0; x < img_size.width; x++ )
      {
        original_grid.at< cv::Vec2f >( y * img_size.width + x, 0 ) = cv::Vec2f( x, y );
      }
    }

    cv::Mat rectified_grid;
    cv::undistortPoints( original_grid, rectified_grid,
                         calibration.left.camera_matrix,
                         calibration.left.dist_coeffs,
                         calibration.R1,
                         calibration.P1 );

    rectified_grid = rectified_grid.reshape( 2, img_size.height );

    cv::Mat maps[2];
    cv::split( rectified_grid, maps );
    unrectification_map_x = maps[0];
    unrectification_map_y = maps[1];

    rectification_computed = true;
  }
};


compute_stereo_disparity::compute_stereo_disparity()
: d( new priv() )
{
  attach_logger( "viame.opencv.compute_stereo_disparity" );
  d->logger = logger();
  d->calibrator.set_logger( d->logger );
}


compute_stereo_disparity::~compute_stereo_disparity()
{
}


// ---------------------------------------------------------------------------------------
kv::config_block_sptr
compute_stereo_disparity
::get_configuration() const
{
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "algorithm", d->algorithm,
    "Stereo matching algorithm: 'BM' (Block Matching) or 'SGBM' (Semi-Global Block Matching). "
    "SGBM generally produces better results but is slower." );

  config->set_value( "min_disparity", d->min_disparity,
    "Minimum possible disparity value. Normally 0, but can be negative for "
    "cameras with convergent optical axes." );

  config->set_value( "num_disparities", d->num_disparities,
    "Maximum disparity minus minimum disparity. Must be divisible by 16. "
    "Larger values allow matching objects closer to the camera." );

  config->set_value( "sad_window_size", d->sad_window_size,
    "SAD (Sum of Absolute Differences) window size for BM algorithm. Must be odd, typically 5-21." );

  config->set_value( "block_size", d->block_size,
    "Block size for SGBM algorithm. Must be odd, typically 3-11." );

  config->set_value( "speckle_window_size", d->speckle_window_size,
    "Maximum size of smooth disparity regions to consider for speckle filtering. "
    "Set to 0 to disable speckle filtering." );

  config->set_value( "speckle_range", d->speckle_range,
    "Maximum disparity variation within each connected component for speckle filtering." );

  config->set_value( "use_wls_filter", d->use_wls_filter,
    "Apply Weighted Least Squares (WLS) filtering to smooth the disparity map while "
    "preserving edges. Requires computing disparity for both left and right images." );

  config->set_value( "wls_lambda", d->wls_lambda,
    "WLS filter regularization parameter. Larger values produce smoother disparity maps." );

  config->set_value( "wls_sigma", d->wls_sigma,
    "WLS filter sigma parameter for color similarity weighting." );

  config->set_value( "calibration_file", d->calibration_file,
    "Path to stereo calibration file (OpenCV YAML/XML format). If specified, images will be "
    "rectified before computing disparity. Leave empty if input images are already rectified "
    "(e.g., when called from measurement_utilities which handles its own rectification)." );

  config->set_value( "compute_depth", d->compute_depth,
  "If true, computes depth Z = (fx * baseline) / disparity. "
  "If false, computes disparity. Depth requires a valid calibration file." );

  config->set_value( "export_as_alpha", d->export_as_alpha,
    "If true, outputs the original left color image with the computed map (depth/disparity) "
    "in the 4th (Alpha) channel. If false, outputs a 1-channel image of the map." );

  config->set_value( "output_rectified", d->output_rectified,
    "If true, the output map (and color image if export_as_alpha is true) will be kept "
    "in the rectified coordinate space. If false, they will be mapped back to the original image." );

  config->set_value( "output_format", d->output_format,
    "Output format: 'float32' (best for TIFF), 'uint16_scaled' (best for 16-bit PNG), or 'raw'." );

  config->set_value( "uint16_scale_factor", d->uint16_scale_factor,
    "Multiplier used ONLY when output_format is 'uint16_scaled'. "
    "e.g., if depth is in meters, a factor of 1000 converts it to millimeters to save as integers in PNG." );

  return config;
}

// ---------------------------------------------------------------------------------------
void compute_stereo_disparity
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
  d->use_wls_filter = config->get_value< bool >( "use_wls_filter" );
  d->wls_lambda = config->get_value< double >( "wls_lambda" );
  d->wls_sigma = config->get_value< double >( "wls_sigma" );
  d->calibration_file = config->get_value< std::string >( "calibration_file" );
  d->compute_depth = config->get_value< bool >( "compute_depth" );
  d->export_as_alpha = config->get_value< bool >( "export_as_alpha" );
  d->output_rectified = config->get_value< bool >( "output_rectified" );
  d->output_format = config->get_value< std::string >( "output_format" );
  d->uint16_scale_factor = config->get_value< double >( "uint16_scale_factor" );

  // Ensure num_disparities is divisible by 16
  if( d->num_disparities % 16 != 0 )
  {
    d->num_disparities = ( ( d->num_disparities / 16 ) + 1 ) * 16;
    LOG_WARN( d->logger, "num_disparities adjusted to " << d->num_disparities
              << " (must be divisible by 16)" );
  }

  // Ensure block_size is odd
  if( d->block_size % 2 == 0 )
  {
    d->block_size++;
    LOG_WARN( d->logger, "block_size adjusted to " << d->block_size << " (must be odd)" );
  }

  // Load calibration if specified
  d->load_calibration();

  // Create stereo matchers
  d->create_matchers();
}


// ---------------------------------------------------------------------------------------
bool compute_stereo_disparity
::check_configuration( kv::config_block_sptr config ) const
{
  std::string algorithm = config->get_value< std::string >( "algorithm" );
  if( algorithm != "BM" && algorithm != "SGBM" )
  {
    LOG_ERROR( logger(), "Invalid algorithm: " << algorithm << ". Must be 'BM' or 'SGBM'." );
    return false;
  }

  std::string output_format = config->get_value< std::string >( "output_format" );
  if( output_format != "raw" && output_format != "float32" && output_format != "uint16_scaled" )
  {
    LOG_ERROR( logger(), "Invalid output_format: " << output_format
               << ". Must be 'raw', 'float32', or 'uint16_scaled'." );
    return false;
  }

  return true;
}


// ---------------------------------------------------------------------------------------
kv::image_container_sptr compute_stereo_disparity
::compute( kv::image_container_sptr left_image,
           kv::image_container_sptr right_image ) const
{
  if( !left_image || !right_image )
  {
    LOG_WARN( d->logger, "Null input image(s)" );
    return kv::image_container_sptr();
  }

  if( left_image->get_image().size() != right_image->get_image().size() )
  {
    LOG_WARN( d->logger, "Inconsistent left/right image sizes" );
    return kv::image_container_sptr();
  }

  // Convert to OpenCV format
  cv::Mat ocv_left = kwiver::arrows::ocv::image_container::vital_to_ocv(
    left_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );
  cv::Mat ocv_right = kwiver::arrows::ocv::image_container::vital_to_ocv(
    right_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

  // Convert to grayscale
  cv::Mat left_gray = calibrate_stereo_cameras::to_grayscale( ocv_left );
  cv::Mat right_gray = calibrate_stereo_cameras::to_grayscale( ocv_right );

  // Rectify if calibration is loaded
  cv::Mat left_rect, right_rect;
  cv::Mat left_color_rect;  // For export_as_alpha mode
  if( d->rectify_images )
  {
    d->compute_rectification_maps( left_gray.size() );
    cv::remap( left_gray, left_rect, d->rectification_map_left_x,
               d->rectification_map_left_y, cv::INTER_LINEAR );
    cv::remap( right_gray, right_rect, d->rectification_map_right_x,
               d->rectification_map_right_y, cv::INTER_LINEAR );

    // Also rectify color image if we need it for alpha channel output
    if( d->export_as_alpha )
    {
      cv::remap( ocv_left, left_color_rect, d->rectification_map_left_x,
                 d->rectification_map_left_y, cv::INTER_LINEAR );
    }
  }
  else
  {
    left_rect = left_gray;
    right_rect = right_gray;
    if( d->export_as_alpha )
    {
      left_color_rect = ocv_left;
    }
  }

  cv::Mat left_disparity_raw;
  d->left_matcher->compute( left_rect, right_rect, left_disparity_raw );

  if( d->use_wls_filter && d->right_matcher && d->wls_filter )
  {
    cv::Mat right_disparity_raw, filtered_disparity;
    d->right_matcher->compute( right_rect, left_rect, right_disparity_raw );

    d->wls_filter->filter( left_disparity_raw, left_rect, filtered_disparity,
                           right_disparity_raw, cv::Rect(), right_rect );

    left_disparity_raw = filtered_disparity;
  }

  cv::Mat float_map;
  left_disparity_raw.convertTo( float_map, CV_32F, 1.0 / 16.0 );
  float_map.setTo( 0, float_map < 0 );

  if( d->compute_depth )
  {
    if( d->calibration.P1.empty() ) {
      VITAL_THROW( kv::invalid_data, "Cannot compute depth: calibration data missing." );
    }

    double fx = d->calibration.P1.at<double>( 0, 0 );
    double baseline = -d->calibration.P2.at<double>( 0, 3 ) / d->calibration.P2.at<double>( 0, 0 );
    double depth_scale = fx * baseline;

    cv::Mat depth_map = cv::Mat::zeros( float_map.size(), CV_32F );
    cv::Mat mask = float_map > 0;
    cv::divide( depth_scale, float_map, depth_map );
    depth_map.setTo( 0, ~mask );

    float_map = depth_map;
  }

  cv::Mat aligned_map;
  if( d->rectify_images && !d->output_rectified )
  {
    cv::remap( float_map, aligned_map,
               d->unrectification_map_x, d->unrectification_map_y,
               cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar( 0 ) );
  }
  else
  {
    aligned_map = float_map;
  }

  cv::Mat formatted_map;
  if( d->output_format == "float32" )
  {
    formatted_map = aligned_map;
  }
  else if( d->output_format == "uint16_scaled" )
  {
    aligned_map.convertTo( formatted_map, CV_16U, d->uint16_scale_factor );
  }
  else // "raw"
  {
    if( d->compute_depth ) formatted_map = aligned_map;
    else {
      aligned_map.convertTo( formatted_map, CV_16S, 16.0 );
    }
  }
  
  cv::Mat output;
  if( d->export_as_alpha )
  {
    cv::Mat color_base;
    if( d->rectify_images && d->output_rectified ) {
      color_base = left_color_rect;
    } else {
      color_base = ocv_left;
    }

    cv::Mat color_converted;
    if( formatted_map.depth() == CV_32F ) {
      color_base.convertTo( color_converted, CV_32F, 1.0/255.0 );
    }
    else if( formatted_map.depth() == CV_16U ) {
      color_base.convertTo( color_converted, CV_16U, 257.0 );
    }
    else {
      color_base.convertTo( color_converted, formatted_map.depth() );
    }

    cv::cvtColor( color_converted, output, cv::COLOR_BGR2BGRA );
    std::vector<cv::Mat> channels;
    cv::split( output, channels );
    channels[3] = formatted_map;
    cv::merge( channels, output );
  }
  else
  {
    output = formatted_map;
  }

  if (output.channels() == 1)
  {
    return kv::image_container_sptr(
      new kwiver::arrows::ocv::image_container( output, kwiver::arrows::ocv::image_container::OTHER_COLOR ) );
  }

  return kv::image_container_sptr(
    new kwiver::arrows::ocv::image_container( output, kwiver::arrows::ocv::image_container::BGR_COLOR ) );
}

} //end namespace viame
