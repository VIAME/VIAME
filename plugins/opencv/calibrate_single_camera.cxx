/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Single camera calibration utility function implementations
 */

#include "calibrate_single_camera.h"
#include "calibrate_stereo_cameras.h"

#include <vital/range/transform.h>

#include <opencv2/calib3d/calib3d.hpp>

#include <fstream>
#include <map>

namespace kv = kwiver::vital;

namespace viame {

// =============================================================================
// Attribute Parsing
// =============================================================================

// -----------------------------------------------------------------------------
std::pair< std::string, float >
parse_detection_attribute( const std::string& note )
{
  // Read a formatted note in detection "(trk) :name=value"
  std::size_t pos = note.find_first_of( ':' );
  std::size_t pos2 = note.find_first_of( '=' );

  std::string attr_name = "";
  float value = 0.0f;

  if( pos == std::string::npos || pos2 == std::string::npos || pos2 == pos + 1 )
  {
    return std::make_pair( attr_name, value );
  }

  attr_name = note.substr( pos + 1, pos2 - pos - 1 );
  value = std::stof( note.substr( pos2 + 1 ) );

  return std::make_pair( attr_name, value );
}

// =============================================================================
// Calibration Data Extraction
// =============================================================================

// -----------------------------------------------------------------------------
bool
extract_calibration_data_from_tracks(
  const kv::object_track_set_sptr& object_track,
  const CalibrationExtractionOptions& options,
  std::vector< std::vector< cv::Point2f > >& image_points,
  std::vector< std::vector< cv::Point3f > >& object_points,
  cv::Size& grid_size,
  kv::logger_handle_t logger )
{
  if( !object_track || object_track->empty() )
  {
    if( logger )
    {
      LOG_ERROR( logger, "Empty object track set" );
    }
    return false;
  }

  // Group detections by frame
  std::map< kv::frame_id_t, std::vector< cv::Point2f > > frame_image_pts;
  std::map< kv::frame_id_t, std::vector< cv::Point3f > > frame_world_pts;

  for( const auto& track : object_track->tracks() )
  {
    for( auto state : *track | kv::as_object_track )
    {
      if( state->detection()->notes().empty() )
      {
        continue;
      }

      std::map< std::string, double > attrs;
      for( const auto& note : state->detection()->notes() )
      {
        attrs.insert( parse_detection_attribute( note ) );
      }

      // Only keep points that have xyz world coordinates
      if( attrs.count( "stereo3d_x" ) && attrs.count( "stereo3d_y" ) &&
          attrs.count( "stereo3d_z" ) )
      {
        cv::Point3f world_pt(
          static_cast< float >( attrs["stereo3d_x"] ),
          static_cast< float >( attrs["stereo3d_y"] ),
          static_cast< float >( attrs["stereo3d_z"] ) );

        auto center = state->detection()->bounding_box().center();
        cv::Point2f image_pt(
          static_cast< float >( center[0] ),
          static_cast< float >( center[1] ) );

        frame_image_pts[state->frame()].push_back( image_pt );
        frame_world_pts[state->frame()].push_back( world_pt );
      }
    }
  }

  if( frame_image_pts.empty() )
  {
    if( logger )
    {
      LOG_ERROR( logger, "No valid calibration points found in tracks" );
    }
    return false;
  }

  // Determine grid size from first frame's world points
  if( !frame_world_pts.empty() )
  {
    const auto& first_world_pts = frame_world_pts.begin()->second;
    float max_x = 0, max_y = 0;
    for( const auto& pt : first_world_pts )
    {
      max_x = std::max( max_x, pt.x );
      max_y = std::max( max_y, pt.y );
    }
    int grid_w = static_cast< int >( max_x / options.square_size ) + 1;
    int grid_h = static_cast< int >( max_y / options.square_size ) + 1;
    grid_size = cv::Size( grid_w, grid_h );

    if( logger )
    {
      LOG_DEBUG( logger, "Detected grid size: " << grid_w << "x" << grid_h );
    }
  }

  // Convert to vectors, applying frame count threshold
  unsigned frame_count = 0;
  unsigned max_frames = ( options.frame_count_threshold > 0 ) ?
    options.frame_count_threshold : static_cast< unsigned >( frame_image_pts.size() );

  for( const auto& kv : frame_image_pts )
  {
    if( frame_count >= max_frames )
    {
      break;
    }

    // Only use frames with the expected number of corners
    size_t expected_corners = static_cast< size_t >( grid_size.width * grid_size.height );
    if( kv.second.size() == expected_corners )
    {
      image_points.push_back( kv.second );
      object_points.push_back( frame_world_pts[kv.first] );
      frame_count++;
    }
  }

  if( logger )
  {
    LOG_DEBUG( logger, "Extracted " << image_points.size() << " valid frames for calibration" );
  }

  return !image_points.empty();
}

// -----------------------------------------------------------------------------
cv::Size
estimate_image_size_from_tracks(
  const kv::object_track_set_sptr& object_track )
{
  cv::Size image_size( 0, 0 );

  if( !object_track )
  {
    return image_size;
  }

  for( const auto& track : object_track->tracks() )
  {
    for( auto state : *track | kv::as_object_track )
    {
      auto bbox = state->detection()->bounding_box();
      image_size.width = std::max( image_size.width,
        static_cast< int >( bbox.max_x() + bbox.width() / 2 ) );
      image_size.height = std::max( image_size.height,
        static_cast< int >( bbox.max_y() + bbox.height() / 2 ) );
    }
  }

  // Add some margin and round up to reasonable values
  image_size.width = ( ( image_size.width + 100 ) / 100 ) * 100;
  image_size.height = ( ( image_size.height + 100 ) / 100 ) * 100;

  return image_size;
}

// =============================================================================
// Calibration Output
// =============================================================================

// -----------------------------------------------------------------------------
bool
write_mono_calibration_opencv(
  const MonoCalibrationResult& result,
  const std::string& output_directory,
  kv::logger_handle_t logger )
{
  std::string output_dir = output_directory.empty() ? "." : output_directory;
  std::string intrinsics_file = output_dir + "/intrinsics.yml";

  cv::FileStorage fs( intrinsics_file, cv::FileStorage::WRITE );
  if( !fs.isOpened() )
  {
    if( logger )
    {
      LOG_ERROR( logger, "Failed to open " << intrinsics_file << " for writing" );
    }
    return false;
  }

  fs << "M1" << result.camera_matrix;
  fs << "D1" << result.dist_coeffs;
  fs.release();

  if( logger )
  {
    LOG_DEBUG( logger, "Wrote intrinsics to: " << intrinsics_file );
  }

  return true;
}

// -----------------------------------------------------------------------------
bool
write_mono_calibration_json(
  const MonoCalibrationResult& result,
  const MonoCalibrationWriteOptions& options,
  kv::logger_handle_t logger )
{
  if( options.json_filename.empty() )
  {
    return true;  // Nothing to do
  }

  std::string output_dir = options.output_directory.empty() ? "." : options.output_directory;
  std::string json_file = options.json_filename;

  // If not absolute path, prepend output directory
  if( json_file[0] != '/' && json_file.find( ":/" ) == std::string::npos )
  {
    json_file = output_dir + "/" + json_file;
  }

  std::ofstream ofs( json_file );
  if( !ofs.is_open() )
  {
    if( logger )
    {
      LOG_ERROR( logger, "Failed to open " << json_file << " for writing" );
    }
    return false;
  }

  ofs << "{\n";
  ofs << "  \"image_width\": " << options.image_size.width << ",\n";
  ofs << "  \"image_height\": " << options.image_size.height << ",\n";
  ofs << "  \"grid_width\": " << options.grid_size.width << ",\n";
  ofs << "  \"grid_height\": " << options.grid_size.height << ",\n";
  ofs << "  \"square_size_mm\": " << options.square_size << ",\n";
  ofs << "  \"rms_error\": " << result.rms_error << ",\n";

  // Intrinsics
  ofs << "  \"fx\": " << result.camera_matrix.at< double >( 0, 0 ) << ",\n";
  ofs << "  \"fy\": " << result.camera_matrix.at< double >( 1, 1 ) << ",\n";
  ofs << "  \"cx\": " << result.camera_matrix.at< double >( 0, 2 ) << ",\n";
  ofs << "  \"cy\": " << result.camera_matrix.at< double >( 1, 2 ) << ",\n";

  // Distortion coefficients
  ofs << "  \"k1\": " << result.dist_coeffs.at< double >( 0 ) << ",\n";
  ofs << "  \"k2\": " << result.dist_coeffs.at< double >( 1 ) << ",\n";
  ofs << "  \"p1\": " << result.dist_coeffs.at< double >( 2 ) << ",\n";
  ofs << "  \"p2\": " << result.dist_coeffs.at< double >( 3 ) << ",\n";

  double k3 = ( result.dist_coeffs.cols > 4 || result.dist_coeffs.rows > 4 ) ?
    result.dist_coeffs.at< double >( 4 ) : 0.0;
  ofs << "  \"k3\": " << k3 << "\n";
  ofs << "}\n";
  ofs.close();

  if( logger )
  {
    LOG_DEBUG( logger, "Wrote JSON calibration to: " << json_file );
  }

  return true;
}

// -----------------------------------------------------------------------------
bool
write_mono_calibration(
  const MonoCalibrationResult& result,
  const MonoCalibrationWriteOptions& options,
  kv::logger_handle_t logger )
{
  bool success = true;

  // Write OpenCV YAML format
  if( !write_mono_calibration_opencv( result, options.output_directory, logger ) )
  {
    success = false;
  }

  // Write JSON format
  if( !write_mono_calibration_json( result, options, logger ) )
  {
    success = false;
  }

  return success;
}

} // namespace viame
