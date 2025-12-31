/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stereo camera calibration utilities
 *
 * This module provides common functionality for stereo camera calibration
 * including chessboard detection, single camera calibration with automatic
 * distortion model selection, and stereo calibration.
 */

#ifndef VIAME_OPENCV_CALIBRATE_STEREO_CAMERAS_H
#define VIAME_OPENCV_CALIBRATE_STEREO_CAMERAS_H

#include "viame_opencv_export.h"

#include <vital/types/camera_perspective.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>

#include <memory>
#include <string>
#include <vector>

namespace viame {

// =============================================================================
/// Result of chessboard corner detection
struct VIAME_OPENCV_EXPORT ChessboardDetectionResult
{
  bool found;                           ///< Whether corners were detected
  std::vector< cv::Point2f > corners;   ///< Detected corner positions (image coords)
  cv::Size grid_size;                   ///< Grid size used for detection (width x height)

  ChessboardDetectionResult() : found( false ), grid_size( 0, 0 ) {}
};

// =============================================================================
/// Result of single camera calibration
struct VIAME_OPENCV_EXPORT MonoCalibrationResult
{
  bool success;                       ///< Whether calibration succeeded
  double rms_error;                   ///< RMS reprojection error in pixels
  cv::Mat camera_matrix;              ///< 3x3 camera intrinsic matrix
  cv::Mat dist_coeffs;                ///< Distortion coefficients [k1,k2,p1,p2,k3]
  int calibration_flags;              ///< OpenCV calibration flags used

  MonoCalibrationResult() : success( false ), rms_error( 0.0 ), calibration_flags( 0 ) {}
};

// =============================================================================
/// Result of stereo camera calibration
struct VIAME_OPENCV_EXPORT calibrate_stereo_cameras_result
{
  bool success;                       ///< Whether calibration succeeded

  // Per-camera results
  MonoCalibrationResult left;         ///< Left camera calibration
  MonoCalibrationResult right;        ///< Right camera calibration

  // Stereo extrinsics
  double stereo_rms_error;            ///< Stereo RMS reprojection error
  cv::Mat R;                          ///< 3x3 rotation matrix (right w.r.t. left)
  cv::Mat T;                          ///< 3x1 translation vector (right w.r.t. left)

  // Rectification matrices (computed by stereoRectify)
  cv::Mat R1, R2;                     ///< Rectification rotations
  cv::Mat P1, P2;                     ///< Projection matrices in rectified coords
  cv::Mat Q;                          ///< Disparity-to-depth mapping matrix

  // Image dimensions
  cv::Size image_size;                ///< Image size used for calibration

  // Grid information
  cv::Size grid_size;                 ///< Chessboard grid size (inner corners)
  double square_size;                 ///< Square size in world units (e.g., mm)

  calibrate_stereo_cameras_result()
    : success(false), stereo_rms_error(0.0), square_size(0.0) {}
};

// =============================================================================
/// Stereo calibration utility class
///
/// Provides methods for chessboard detection, mono calibration with automatic
/// distortion model selection, and stereo calibration. Mirrors functionality
/// from calibrate_cameras.py for consistency.
class VIAME_OPENCV_EXPORT calibrate_stereo_cameras
{
public:
  calibrate_stereo_cameras();
  ~calibrate_stereo_cameras();

  // -------------------------------------------------------------------------
  // Chessboard Detection
  // -------------------------------------------------------------------------

  /// Detect chessboard corners in an image
  ///
  /// \param image Grayscale input image
  /// \param grid_size Expected grid size (inner corners: width x height)
  /// \param max_dim Maximum dimension for initial detection (larger images downscaled)
  /// \return Detection result with corners if found
  ChessboardDetectionResult detect_chessboard(
    const cv::Mat& image,
    const cv::Size& grid_size,
    int max_dim = 5000 ) const;

  /// Auto-detect chessboard grid size by trying common configurations
  ///
  /// Tries common grid sizes (6x5, 7x6, 8x6, etc.) and systematic search
  /// to find a matching chessboard pattern.
  ///
  /// \param image Grayscale input image
  /// \param max_dim Maximum dimension for detection
  /// \param min_size Minimum grid dimension to try
  /// \param max_size Maximum grid dimension to try
  /// \return Detection result with detected grid size if found
  ChessboardDetectionResult detect_chessboard_auto(
    const cv::Mat& image,
    int max_dim = 5000,
    int min_size = 4,
    int max_size = 15 ) const;

  // -------------------------------------------------------------------------
  // Single Camera Calibration
  // -------------------------------------------------------------------------

  /// Calibrate a single camera with automatic distortion model selection
  ///
  /// Performs iterative calibration, testing progressively simpler distortion
  /// models and selecting the simplest model that doesn't significantly
  /// increase reprojection error.
  ///
  /// \param image_points Vector of detected corner positions per frame
  /// \param object_points Vector of 3D object points per frame
  /// \param image_size Image dimensions
  /// \param camera_name Name for logging purposes
  /// \return Calibration result
  MonoCalibrationResult calibrate_single_camera(
    const std::vector< std::vector< cv::Point2f > >& image_points,
    const std::vector< std::vector< cv::Point3f > >& object_points,
    const cv::Size& image_size,
    const std::string& camera_name = "camera" ) const;

  // -------------------------------------------------------------------------
  // Stereo Calibration
  // -------------------------------------------------------------------------

  /// Perform full stereo calibration
  ///
  /// Calibrates both cameras individually, then performs stereo calibration
  /// to estimate the relative pose.
  ///
  /// \param left_image_points Left camera corner positions per frame
  /// \param right_image_points Right camera corner positions per frame
  /// \param object_points 3D object points per frame
  /// \param image_size Image dimensions
  /// \param grid_size Chessboard grid size
  /// \param square_size Square size in world units
  /// \return Stereo calibration result
  calibrate_stereo_cameras_result calibrate_stereo(
    const std::vector< std::vector< cv::Point2f > >& left_image_points,
    const std::vector< std::vector< cv::Point2f > >& right_image_points,
    const std::vector< std::vector< cv::Point3f > >& object_points,
    const cv::Size& image_size,
    const cv::Size& grid_size,
    double square_size ) const;

  // -------------------------------------------------------------------------
  // Output Writing
  // -------------------------------------------------------------------------

  /// Write calibration to JSON file (camera_rig_io compatible format)
  ///
  /// \param result Stereo calibration result
  /// \param filename Output JSON file path
  /// \return true on success
  bool write_calibration_json(
    const calibrate_stereo_cameras_result& result,
    const std::string& filename ) const;

  /// Write calibration to OpenCV YAML files (intrinsics.yml + extrinsics.yml)
  ///
  /// \param result Stereo calibration result
  /// \param output_directory Directory for output files
  /// \return true on success
  bool write_calibration_opencv(
    const calibrate_stereo_cameras_result& result,
    const std::string& output_directory ) const;

  /// Write calibration to NPZ file
  ///
  /// Note: Requires external handling as C++ doesn't natively support NPZ.
  /// This method writes individual .npy-compatible binary files.
  ///
  /// \param result Stereo calibration result
  /// \param filename Output file path
  /// \return true on success
  bool write_calibration_npz(
    const calibrate_stereo_cameras_result& result,
    const std::string& filename ) const;

  /// Load calibration from OpenCV YAML files (intrinsics.yml + extrinsics.yml)
  ///
  /// \param input_directory Directory containing calibration files
  /// \param[out] result Loaded calibration result
  /// \return true on success
  bool load_calibration_opencv(
    const std::string& input_directory,
    calibrate_stereo_cameras_result& result ) const;

  // -------------------------------------------------------------------------
  // Utility Methods
  // -------------------------------------------------------------------------

  /// Generate object points for a chessboard pattern
  ///
  /// \param grid_size Grid size (inner corners)
  /// \param square_size Square size in world units
  /// \return Vector of 3D points in the pattern plane (Z=0)
  static std::vector< cv::Point3f > make_object_points(
    const cv::Size& grid_size,
    double square_size = 1.0 );

  /// Convert image to grayscale, handling color, grayscale, and Bayer inputs
  ///
  /// \param image Input image
  /// \param is_bayer If true, treat as Bayer pattern image
  /// \return Grayscale image
  static cv::Mat to_grayscale(
    const cv::Mat& image,
    bool is_bayer = false );

  /// Convert MonoCalibrationResult to KWIVER camera intrinsics
  ///
  /// \param result Mono calibration result
  /// \return KWIVER camera intrinsics object
  static kwiver::vital::camera_intrinsics_sptr to_kwiver_intrinsics(
    const MonoCalibrationResult& result );

  /// Convert calibrate_stereo_cameras_result to KWIVER camera perspective pair
  ///
  /// \param result Stereo calibration result
  /// \param[out] left_camera Left camera
  /// \param[out] right_camera Right camera
  static void to_kwiver_cameras(
    const calibrate_stereo_cameras_result& result,
    kwiver::vital::simple_camera_perspective_sptr& left_camera,
    kwiver::vital::simple_camera_perspective_sptr& right_camera );

  // -------------------------------------------------------------------------
  // Configuration
  // -------------------------------------------------------------------------

  /// Set logger for status messages
  void set_logger( kwiver::vital::logger_handle_t logger );

private:
  class priv;
  std::unique_ptr<priv> d;
};

} // namespace viame

#endif // VIAME_OPENCV_CALIBRATE_STEREO_CAMERAS_H
