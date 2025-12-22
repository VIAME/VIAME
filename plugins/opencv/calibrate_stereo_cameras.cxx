/*ckwg +29
 * Copyright 2025 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
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
 * \brief Stereo camera calibration utilities implementation
 */

#include "calibrate_stereo_cameras.h"

#include <vital/types/camera_intrinsics.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/eigen.hpp>

#include <fstream>
#include <algorithm>
#include <cmath>

namespace kv = kwiver::vital;

namespace viame {

// =============================================================================
// Private implementation
// =============================================================================
class calibrate_stereo_cameras::priv
{
public:
  priv()
    : m_logger( kv::get_logger( "viame.opencv.calibrate_stereo_cameras" ) )
  {}

  kv::logger_handle_t m_logger;
};

// =============================================================================
// calibrate_stereo_cameras implementation
// =============================================================================

calibrate_stereo_cameras::calibrate_stereo_cameras()
  : d( new priv )
{
}

calibrate_stereo_cameras::~calibrate_stereo_cameras()
{
}

// -----------------------------------------------------------------------------
void
calibrate_stereo_cameras::set_logger( kv::logger_handle_t logger )
{
  d->m_logger = logger;
}

// -----------------------------------------------------------------------------
ChessboardDetectionResult
calibrate_stereo_cameras::detect_chessboard(
  const cv::Mat& image,
  const cv::Size& grid_size,
  int max_dim ) const
{
  ChessboardDetectionResult result;
  result.grid_size = grid_size;

  if( image.empty() )
  {
    return result;
  }

  // Compute scale factor for large images
  int min_len = std::min( image.rows, image.cols );
  double scale = 1.0;
  while( scale * min_len > max_dim )
  {
    scale /= 2.0;
  }

  // Termination criteria for corner refinement
  cv::TermCriteria criteria(
    cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.001 );

  int flags = cv::CALIB_CB_ADAPTIVE_THRESH;

  if( scale < 1.0 )
  {
    // Detect on downscaled image first
    cv::Mat small;
    cv::resize( image, small, cv::Size(), scale, scale );

    result.found = cv::findChessboardCorners(
      small, grid_size, result.corners, flags );

    if( result.found )
    {
      // Refine at low resolution
      cv::cornerSubPix( small, result.corners, cv::Size( 11, 11 ),
                        cv::Size( -1, -1 ), criteria );

      // Scale corners back to full resolution
      for( auto& corner : result.corners )
      {
        corner.x /= static_cast<float>( scale );
        corner.y /= static_cast<float>( scale );
      }
    }
  }
  else
  {
    result.found = cv::findChessboardCorners(
      image, grid_size, result.corners, flags );
  }

  if( result.found )
  {
    // Refine at full resolution
    cv::cornerSubPix( image, result.corners, cv::Size( 11, 11 ),
                      cv::Size( -1, -1 ), criteria );
  }

  return result;
}

// -----------------------------------------------------------------------------
ChessboardDetectionResult
calibrate_stereo_cameras::detect_chessboard_auto(
  const cv::Mat& image,
  int max_dim,
  int min_size,
  int max_size ) const
{
  // Common grid sizes to try first
  std::vector< cv::Size > common_sizes = {
    cv::Size( 6, 5 ), cv::Size( 7, 6 ), cv::Size( 8, 6 ),
    cv::Size( 9, 6 ), cv::Size( 5, 4 ), cv::Size( 8, 5 ),
    cv::Size( 7, 5 )
  };

  // Build list of all sizes to try
  std::vector< cv::Size > all_sizes = common_sizes;
  for( int x = min_size; x <= max_size; ++x )
  {
    for( int y = min_size; y <= x; ++y )
    {
      cv::Size size( x, y );
      cv::Size size_t( y, x );

      // Check if already in list
      bool found = false;
      for( const auto& s : all_sizes )
      {
        if( ( s.width == size.width && s.height == size.height ) ||
            ( s.width == size_t.width && s.height == size_t.height ) )
        {
          found = true;
          break;
        }
      }
      if( !found )
      {
        all_sizes.push_back( size );
      }
    }
  }

  // Try each size
  for( const auto& grid_size : all_sizes )
  {
    auto result = detect_chessboard( image, grid_size, max_dim );
    if( result.found )
    {
      LOG_DEBUG( d->m_logger, "Auto-detected grid size: "
                 << grid_size.width << "x" << grid_size.height );
      return result;
    }

    // Also try transposed if not square
    if( grid_size.width != grid_size.height )
    {
      cv::Size transposed( grid_size.height, grid_size.width );
      result = detect_chessboard( image, transposed, max_dim );
      if( result.found )
      {
        LOG_DEBUG( d->m_logger, "Auto-detected grid size: "
                   << transposed.width << "x" << transposed.height );
        return result;
      }
    }
  }

  // Not found
  ChessboardDetectionResult result;
  LOG_WARN( d->m_logger, "Could not auto-detect chessboard grid size" );
  return result;
}

// -----------------------------------------------------------------------------
MonoCalibrationResult
calibrate_stereo_cameras::calibrate_single_camera(
  const std::vector< std::vector< cv::Point2f > >& image_points,
  const std::vector< std::vector< cv::Point3f > >& object_points,
  const cv::Size& image_size,
  const std::string& camera_name ) const
{
  MonoCalibrationResult result;

  if( image_points.empty() || object_points.empty() )
  {
    LOG_ERROR( d->m_logger, "No calibration points for " << camera_name );
    return result;
  }

  // Initialize camera matrix
  cv::Mat K = cv::Mat::eye( 3, 3, CV_64F );
  K.at<double>( 0, 0 ) = 1000.0;  // fx
  K.at<double>( 1, 1 ) = 1000.0;  // fy
  K.at<double>( 0, 2 ) = image_size.width / 2.0;   // cx
  K.at<double>( 1, 2 ) = image_size.height / 2.0;  // cy

  cv::Mat dist = cv::Mat::zeros( 5, 1, CV_64F );
  std::vector< cv::Mat > rvecs, tvecs;

  int flags = 0;

  // Step 1: Initial calibration
  LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Initial calibration" );
  result.rms_error = cv::calibrateCamera(
    object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );
  LOG_DEBUG( d->m_logger, camera_name << " initial RMS: " << result.rms_error );

  // Step 2: Test aspect ratio
  double aspect_ratio = K.at<double>( 0, 0 ) / K.at<double>( 1, 1 );
  if( 1.0 - std::min( aspect_ratio, 1.0 / aspect_ratio ) < 0.01 )
  {
    LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Fixing aspect ratio" );
    flags |= cv::CALIB_FIX_ASPECT_RATIO;
    result.rms_error = cv::calibrateCamera(
      object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );
  }

  // Step 3: Test principal point
  double pp_x = K.at<double>( 0, 2 );
  double pp_y = K.at<double>( 1, 2 );
  double rel_pp_diff_x = std::abs( pp_x - image_size.width / 2.0 ) / image_size.width;
  double rel_pp_diff_y = std::abs( pp_y - image_size.height / 2.0 ) / image_size.height;
  if( std::max( rel_pp_diff_x, rel_pp_diff_y ) < 0.05 )
  {
    LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Fixing principal point" );
    flags |= cv::CALIB_FIX_PRINCIPAL_POINT;
    result.rms_error = cv::calibrateCamera(
      object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );
  }

  // Set threshold at 25% above baseline error
  double error_threshold = 1.25 * result.rms_error;
  double last_error = result.rms_error;
  cv::Mat last_K = K.clone();
  cv::Mat last_dist = dist.clone();
  int last_flags = flags;

  // Step 4: Test tangential distortion
  LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Testing tangential distortion" );
  flags |= cv::CALIB_ZERO_TANGENT_DIST;
  double test_error = cv::calibrateCamera(
    object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );

  if( test_error > error_threshold )
  {
    // Revert
    K = last_K.clone();
    dist = last_dist.clone();
    flags = last_flags;
    result.rms_error = last_error;
  }
  else
  {
    last_error = test_error;
    last_K = K.clone();
    last_dist = dist.clone();
    last_flags = flags;
    result.rms_error = test_error;

    // Step 5: Test K3
    LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Testing K3 distortion" );
    flags |= cv::CALIB_FIX_K3;
    test_error = cv::calibrateCamera(
      object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );

    if( test_error > error_threshold )
    {
      K = last_K.clone();
      dist = last_dist.clone();
      flags = last_flags;
      result.rms_error = last_error;
    }
    else
    {
      last_error = test_error;
      last_K = K.clone();
      last_dist = dist.clone();
      last_flags = flags;
      result.rms_error = test_error;

      // Step 6: Test K2
      LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Testing K2 distortion" );
      flags |= cv::CALIB_FIX_K2;
      test_error = cv::calibrateCamera(
        object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );

      if( test_error > error_threshold )
      {
        K = last_K.clone();
        dist = last_dist.clone();
        flags = last_flags;
        result.rms_error = last_error;
      }
      else
      {
        last_error = test_error;
        last_K = K.clone();
        last_dist = dist.clone();
        last_flags = flags;
        result.rms_error = test_error;

        // Step 7: Test K1
        LOG_DEBUG( d->m_logger, "Calibrating " << camera_name << ": Testing K1 distortion" );
        flags |= cv::CALIB_FIX_K1;
        test_error = cv::calibrateCamera(
          object_points, image_points, image_size, K, dist, rvecs, tvecs, flags );

        if( test_error > error_threshold )
        {
          K = last_K.clone();
          dist = last_dist.clone();
          flags = last_flags;
          result.rms_error = last_error;
        }
        else
        {
          result.rms_error = test_error;
        }
      }
    }
  }

  result.success = true;
  result.camera_matrix = K;
  result.dist_coeffs = dist;
  result.calibration_flags = flags;

  LOG_DEBUG( d->m_logger, camera_name << " final RMS: " << result.rms_error );

  return result;
}

// -----------------------------------------------------------------------------
calibrate_stereo_cameras_result
calibrate_stereo_cameras::calibrate_stereo(
  const std::vector< std::vector< cv::Point2f > >& left_image_points,
  const std::vector< std::vector< cv::Point2f > >& right_image_points,
  const std::vector< std::vector< cv::Point3f > >& object_points,
  const cv::Size& image_size,
  const cv::Size& grid_size,
  double square_size ) const
{
  calibrate_stereo_cameras_result result;
  result.image_size = image_size;
  result.grid_size = grid_size;
  result.square_size = square_size;

  if( left_image_points.empty() || right_image_points.empty() ||
      object_points.empty() )
  {
    LOG_ERROR( d->m_logger, "No calibration points for stereo calibration" );
    return result;
  }

  // Calibrate each camera individually
  LOG_DEBUG( d->m_logger, "Calibrating left camera..." );
  result.left = calibrate_single_camera(
    left_image_points, object_points, image_size, "left" );

  if( !result.left.success )
  {
    LOG_ERROR( d->m_logger, "Left camera calibration failed" );
    return result;
  }

  LOG_DEBUG( d->m_logger, "Calibrating right camera..." );
  result.right = calibrate_single_camera(
    right_image_points, object_points, image_size, "right" );

  if( !result.right.success )
  {
    LOG_ERROR( d->m_logger, "Right camera calibration failed" );
    return result;
  }

  // Perform stereo calibration with fixed intrinsics
  LOG_DEBUG( d->m_logger, "Computing stereo extrinsics..." );

  cv::Mat K1 = result.left.camera_matrix.clone();
  cv::Mat K2 = result.right.camera_matrix.clone();
  cv::Mat D1 = result.left.dist_coeffs.clone();
  cv::Mat D2 = result.right.dist_coeffs.clone();

  result.stereo_rms_error = cv::stereoCalibrate(
    object_points, left_image_points, right_image_points,
    K1, D1, K2, D2, image_size,
    result.R, result.T,
    cv::noArray(), cv::noArray(),
    cv::CALIB_FIX_INTRINSIC );

  LOG_DEBUG( d->m_logger, "Stereo RMS error: " << result.stereo_rms_error );

  // Compute rectification transforms
  LOG_DEBUG( d->m_logger, "Computing rectification..." );
  cv::stereoRectify(
    K1, D1, K2, D2, image_size,
    result.R, result.T,
    result.R1, result.R2, result.P1, result.P2, result.Q,
    cv::CALIB_ZERO_DISPARITY, 0 );

  result.success = true;

  // Compute baseline for logging
  double baseline = cv::norm( result.T );
  LOG_DEBUG( d->m_logger, "Baseline distance: " << baseline << " (world units)" );

  return result;
}

// -----------------------------------------------------------------------------
bool
calibrate_stereo_cameras::write_calibration_json(
  const calibrate_stereo_cameras_result& result,
  const std::string& filename ) const
{
  if( !result.success )
  {
    LOG_ERROR( d->m_logger, "Cannot write JSON: calibration not successful" );
    return false;
  }

  std::ofstream ofs( filename );
  if( !ofs.is_open() )
  {
    LOG_ERROR( d->m_logger, "Cannot open file for writing: " << filename );
    return false;
  }

  // Helper to write a double value
  auto write_value = [&ofs]( const std::string& name, double value, bool comma = true )
  {
    ofs << "  \"" << name << "\": " << value;
    if( comma ) ofs << ",";
    ofs << "\n";
  };

  ofs << "{\n";

  // Image dimensions and grid
  write_value( "image_width", result.image_size.width );
  write_value( "image_height", result.image_size.height );
  write_value( "grid_width", result.grid_size.width );
  write_value( "grid_height", result.grid_size.height );
  write_value( "square_size_mm", result.square_size );

  // Calibration quality metrics
  write_value( "rms_error_left", result.left.rms_error );
  write_value( "rms_error_right", result.right.rms_error );
  write_value( "rms_error_stereo", result.stereo_rms_error );

  // Left camera intrinsics
  const cv::Mat& K_left = result.left.camera_matrix;
  const cv::Mat& D_left = result.left.dist_coeffs;
  write_value( "fx_left", K_left.at<double>( 0, 0 ) );
  write_value( "fy_left", K_left.at<double>( 1, 1 ) );
  write_value( "cx_left", K_left.at<double>( 0, 2 ) );
  write_value( "cy_left", K_left.at<double>( 1, 2 ) );
  write_value( "k1_left", D_left.rows > 0 ? D_left.at<double>( 0 ) : 0.0 );
  write_value( "k2_left", D_left.rows > 1 ? D_left.at<double>( 1 ) : 0.0 );
  write_value( "p1_left", D_left.rows > 2 ? D_left.at<double>( 2 ) : 0.0 );
  write_value( "p2_left", D_left.rows > 3 ? D_left.at<double>( 3 ) : 0.0 );
  write_value( "k3_left", D_left.rows > 4 ? D_left.at<double>( 4 ) : 0.0 );

  // Right camera intrinsics
  const cv::Mat& K_right = result.right.camera_matrix;
  const cv::Mat& D_right = result.right.dist_coeffs;
  write_value( "fx_right", K_right.at<double>( 0, 0 ) );
  write_value( "fy_right", K_right.at<double>( 1, 1 ) );
  write_value( "cx_right", K_right.at<double>( 0, 2 ) );
  write_value( "cy_right", K_right.at<double>( 1, 2 ) );
  write_value( "k1_right", D_right.rows > 0 ? D_right.at<double>( 0 ) : 0.0 );
  write_value( "k2_right", D_right.rows > 1 ? D_right.at<double>( 1 ) : 0.0 );
  write_value( "p1_right", D_right.rows > 2 ? D_right.at<double>( 2 ) : 0.0 );
  write_value( "p2_right", D_right.rows > 3 ? D_right.at<double>( 3 ) : 0.0 );
  write_value( "k3_right", D_right.rows > 4 ? D_right.at<double>( 4 ) : 0.0 );

  // Translation vector
  ofs << "  \"T\": [" << result.T.at<double>( 0 ) << ", "
      << result.T.at<double>( 1 ) << ", "
      << result.T.at<double>( 2 ) << "],\n";

  // Rotation matrix (flattened row-major)
  ofs << "  \"R\": [";
  for( int i = 0; i < 3; ++i )
  {
    for( int j = 0; j < 3; ++j )
    {
      ofs << result.R.at<double>( i, j );
      if( i < 2 || j < 2 ) ofs << ", ";
    }
  }
  ofs << "]\n";

  ofs << "}\n";
  ofs.close();

  LOG_DEBUG( d->m_logger, "Wrote calibration to: " << filename );
  return true;
}

// -----------------------------------------------------------------------------
bool
calibrate_stereo_cameras::write_calibration_opencv(
  const calibrate_stereo_cameras_result& result,
  const std::string& output_directory ) const
{
  if( !result.success )
  {
    LOG_ERROR( d->m_logger, "Cannot write OpenCV files: calibration not successful" );
    return false;
  }

  // Write intrinsics
  std::string intrinsics_file = output_directory + "/intrinsics.yml";
  cv::FileStorage fs_intr( intrinsics_file, cv::FileStorage::WRITE );
  if( !fs_intr.isOpened() )
  {
    LOG_ERROR( d->m_logger, "Cannot open intrinsics file: " << intrinsics_file );
    return false;
  }

  fs_intr << "M1" << result.left.camera_matrix;
  fs_intr << "D1" << result.left.dist_coeffs;
  fs_intr << "M2" << result.right.camera_matrix;
  fs_intr << "D2" << result.right.dist_coeffs;
  fs_intr.release();

  // Write extrinsics
  std::string extrinsics_file = output_directory + "/extrinsics.yml";
  cv::FileStorage fs_extr( extrinsics_file, cv::FileStorage::WRITE );
  if( !fs_extr.isOpened() )
  {
    LOG_ERROR( d->m_logger, "Cannot open extrinsics file: " << extrinsics_file );
    return false;
  }

  fs_extr << "R" << result.R;
  fs_extr << "T" << result.T;
  fs_extr << "R1" << result.R1;
  fs_extr << "R2" << result.R2;
  fs_extr << "P1" << result.P1;
  fs_extr << "P2" << result.P2;
  fs_extr << "Q" << result.Q;
  fs_extr.release();

  LOG_DEBUG( d->m_logger, "Wrote OpenCV calibration to: " << output_directory );
  return true;
}

// -----------------------------------------------------------------------------
bool
calibrate_stereo_cameras::write_calibration_npz(
  const calibrate_stereo_cameras_result& result,
  const std::string& filename ) const
{
  // NPZ format is Python-specific; this is a placeholder
  // In practice, the Python tool would handle NPZ output
  LOG_WARN( d->m_logger, "NPZ output not implemented in C++; use Python tool" );
  return false;
}

// -----------------------------------------------------------------------------
bool
calibrate_stereo_cameras::load_calibration_opencv(
  const std::string& input_directory,
  calibrate_stereo_cameras_result& result ) const
{
  std::string intrinsics_file = input_directory + "/intrinsics.yml";
  std::string extrinsics_file = input_directory + "/extrinsics.yml";

  // Load intrinsics
  cv::FileStorage fs_intr( intrinsics_file, cv::FileStorage::READ );
  if( !fs_intr.isOpened() )
  {
    LOG_ERROR( d->m_logger, "Cannot open intrinsics file: " << intrinsics_file );
    return false;
  }

  fs_intr["M1"] >> result.left.camera_matrix;
  fs_intr["D1"] >> result.left.dist_coeffs;
  fs_intr["M2"] >> result.right.camera_matrix;
  fs_intr["D2"] >> result.right.dist_coeffs;
  fs_intr.release();

  result.left.success = !result.left.camera_matrix.empty();
  result.right.success = !result.right.camera_matrix.empty();

  // Load extrinsics
  cv::FileStorage fs_extr( extrinsics_file, cv::FileStorage::READ );
  if( !fs_extr.isOpened() )
  {
    LOG_ERROR( d->m_logger, "Cannot open extrinsics file: " << extrinsics_file );
    return false;
  }

  fs_extr["R"] >> result.R;
  fs_extr["T"] >> result.T;
  fs_extr["R1"] >> result.R1;
  fs_extr["R2"] >> result.R2;
  fs_extr["P1"] >> result.P1;
  fs_extr["P2"] >> result.P2;
  fs_extr["Q"] >> result.Q;
  fs_extr.release();

  result.success = result.left.success && result.right.success &&
                   !result.R.empty() && !result.T.empty();

  LOG_DEBUG( d->m_logger, "Loaded calibration from: " << input_directory );
  return result.success;
}

// -----------------------------------------------------------------------------
std::vector< cv::Point3f >
calibrate_stereo_cameras::make_object_points(
  const cv::Size& grid_size,
  double square_size )
{
  std::vector< cv::Point3f > points;
  points.reserve( grid_size.width * grid_size.height );

  for( int j = 0; j < grid_size.height; ++j )
  {
    for( int i = 0; i < grid_size.width; ++i )
    {
      points.emplace_back(
        static_cast< float >( i * square_size ),
        static_cast< float >( j * square_size ),
        0.0f );
    }
  }

  return points;
}

// -----------------------------------------------------------------------------
cv::Mat
calibrate_stereo_cameras::to_grayscale(
  const cv::Mat& image,
  bool is_bayer )
{
  if( image.empty() )
  {
    return cv::Mat();
  }

  if( is_bayer )
  {
    cv::Mat gray;
    if( image.channels() == 3 )
    {
      // Extract first channel and debayer
      std::vector< cv::Mat > channels;
      cv::split( image, channels );
      cv::cvtColor( channels[0], gray, cv::COLOR_BayerBG2GRAY );
    }
    else
    {
      cv::cvtColor( image, gray, cv::COLOR_BayerBG2GRAY );
    }
    return gray;
  }
  else if( image.channels() == 1 )
  {
    return image.clone();
  }
  else if( image.channels() == 3 )
  {
    cv::Mat gray;
    cv::cvtColor( image, gray, cv::COLOR_BGR2GRAY );
    return gray;
  }
  else if( image.channels() == 4 )
  {
    cv::Mat gray;
    cv::cvtColor( image, gray, cv::COLOR_BGRA2GRAY );
    return gray;
  }
  else
  {
    // Unknown format - return as-is
    return image.clone();
  }
}

// -----------------------------------------------------------------------------
kv::camera_intrinsics_sptr
calibrate_stereo_cameras::to_kwiver_intrinsics(
  const MonoCalibrationResult& result )
{
  if( !result.success )
  {
    return nullptr;
  }

  const cv::Mat& K = result.camera_matrix;
  const cv::Mat& D = result.dist_coeffs;

  double fx = K.at<double>( 0, 0 );
  double fy = K.at<double>( 1, 1 );
  double cx = K.at<double>( 0, 2 );
  double cy = K.at<double>( 1, 2 );

  // Convert distortion coefficients to Eigen vector
  Eigen::VectorXd dist( D.rows );
  for( int i = 0; i < D.rows; ++i )
  {
    dist[i] = D.at<double>( i );
  }

  double focal_length = fx;
  double aspect_ratio = fx / fy;
  kv::vector_2d principal_point( cx, cy );

  return std::make_shared<kv::simple_camera_intrinsics>(
    focal_length, principal_point, aspect_ratio, 0.0, dist );
}

// -----------------------------------------------------------------------------
void
calibrate_stereo_cameras::to_kwiver_cameras(
  const calibrate_stereo_cameras_result& result,
  kv::simple_camera_perspective_sptr& left_camera,
  kv::simple_camera_perspective_sptr& right_camera )
{
  if( !result.success )
  {
    left_camera = nullptr;
    right_camera = nullptr;
    return;
  }

  // Left camera: at origin with identity rotation
  left_camera = std::make_shared<kv::simple_camera_perspective>();
  left_camera->set_center( kv::vector_3d( 0, 0, 0 ) );
  left_camera->set_rotation( kv::rotation_d() );
  left_camera->set_intrinsics( to_kwiver_intrinsics( result.left ) );

  // Right camera: with rotation R and translation T
  right_camera = std::make_shared<kv::simple_camera_perspective>();

  // Convert R to Eigen matrix
  Eigen::Matrix3d R_eigen;
  cv::cv2eigen( result.R, R_eigen );
  kv::rotation_d rotation( R_eigen );

  // Convert T to Eigen vector
  Eigen::Vector3d T_eigen;
  cv::cv2eigen( result.T, T_eigen );

  right_camera->set_rotation( rotation );
  right_camera->set_translation( T_eigen );
  right_camera->set_intrinsics( to_kwiver_intrinsics( result.right ) );
}

} // namespace viame
