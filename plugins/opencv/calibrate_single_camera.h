/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Single camera calibration utility functions
 *
 * This module provides utility functions for single camera calibration
 * including extraction of calibration data from object tracks and
 * writing calibration results to various file formats.
 */

#ifndef VIAME_OPENCV_CALIBRATE_SINGLE_CAMERA_H
#define VIAME_OPENCV_CALIBRATE_SINGLE_CAMERA_H

#include "viame_opencv_export.h"

#include <vital/types/object_track_set.h>
#include <vital/logger/logger.h>

#include <opencv2/core/core.hpp>

#include <string>
#include <utility>
#include <vector>

namespace viame {

// Forward declaration
struct MonoCalibrationResult;

// =============================================================================
// Attribute Parsing
// =============================================================================

/// Parse an attribute name-value pair from a detection note string.
///
/// Expected format: "(trk) :name=value"
/// Returns empty name and 0.0 value if parsing fails.
///
/// \param note The note string to parse
/// \return Pair of (attribute_name, value)
VIAME_OPENCV_EXPORT
std::pair< std::string, float >
parse_detection_attribute( const std::string& note );

// =============================================================================
// Calibration Data Extraction
// =============================================================================

/// Options for extracting calibration data from object tracks
struct VIAME_OPENCV_EXPORT CalibrationExtractionOptions
{
  double square_size;             ///< Calibration pattern square size in world units
  unsigned frame_count_threshold; ///< Maximum frames to use (0 for unlimited)

  CalibrationExtractionOptions()
    : square_size( 80.0 )
    , frame_count_threshold( 50 )
  {
  }
};

/// Extract image points and world points from object tracks for calibration.
///
/// Parses object tracks containing detected calibration pattern corners
/// with stereo3d_x/y/z attributes and extracts the 2D-3D point correspondences
/// needed for camera calibration.
///
/// \param object_track Object track set containing detected corners
/// \param options Extraction options (square size, frame threshold)
/// \param[out] image_points Vector of 2D image points per frame
/// \param[out] object_points Vector of 3D world points per frame
/// \param[out] grid_size Detected calibration grid size
/// \param logger Optional logger for status messages
/// \return true if extraction succeeded with valid data
VIAME_OPENCV_EXPORT
bool
extract_calibration_data_from_tracks(
  const kwiver::vital::object_track_set_sptr& object_track,
  const CalibrationExtractionOptions& options,
  std::vector< std::vector< cv::Point2f > >& image_points,
  std::vector< std::vector< cv::Point3f > >& object_points,
  cv::Size& grid_size,
  kwiver::vital::logger_handle_t logger = nullptr );

/// Estimate image size from object track bounding boxes.
///
/// Computes a reasonable image size estimate by finding the maximum
/// bounding box extents across all detections, with some margin added.
///
/// \param object_track Object track set to analyze
/// \return Estimated image size
VIAME_OPENCV_EXPORT
cv::Size
estimate_image_size_from_tracks(
  const kwiver::vital::object_track_set_sptr& object_track );

// =============================================================================
// Calibration Output
// =============================================================================

/// Options for writing mono calibration results
struct VIAME_OPENCV_EXPORT MonoCalibrationWriteOptions
{
  std::string output_directory;   ///< Directory for output files
  std::string json_filename;      ///< JSON output filename (empty to skip)
  double square_size;             ///< Square size for metadata
  cv::Size grid_size;             ///< Grid size for metadata
  cv::Size image_size;            ///< Image size for metadata

  MonoCalibrationWriteOptions()
    : output_directory( "." )
    , json_filename( "calibration.json" )
    , square_size( 80.0 )
    , grid_size( 0, 0 )
    , image_size( 0, 0 )
  {
  }
};

/// Write mono calibration results to OpenCV YAML format.
///
/// Writes camera matrix and distortion coefficients to intrinsics.yml.
///
/// \param result Calibration result to write
/// \param output_directory Output directory path
/// \param logger Optional logger for status messages
/// \return true on success
VIAME_OPENCV_EXPORT
bool
write_mono_calibration_opencv(
  const MonoCalibrationResult& result,
  const std::string& output_directory,
  kwiver::vital::logger_handle_t logger = nullptr );

/// Write mono calibration results to JSON format.
///
/// Writes a human-readable JSON file with all calibration parameters
/// including image/grid dimensions and reprojection error.
///
/// \param result Calibration result to write
/// \param options Output options including paths and metadata
/// \param logger Optional logger for status messages
/// \return true on success
VIAME_OPENCV_EXPORT
bool
write_mono_calibration_json(
  const MonoCalibrationResult& result,
  const MonoCalibrationWriteOptions& options,
  kwiver::vital::logger_handle_t logger = nullptr );

/// Write mono calibration results to all configured formats.
///
/// Convenience function that writes both OpenCV YAML and JSON formats
/// based on the provided options.
///
/// \param result Calibration result to write
/// \param options Output options
/// \param logger Optional logger for status messages
/// \return true if all writes succeeded
VIAME_OPENCV_EXPORT
bool
write_mono_calibration(
  const MonoCalibrationResult& result,
  const MonoCalibrationWriteOptions& options,
  kwiver::vital::logger_handle_t logger = nullptr );

} // namespace viame

#endif // VIAME_OPENCV_CALIBRATE_SINGLE_CAMERA_H
