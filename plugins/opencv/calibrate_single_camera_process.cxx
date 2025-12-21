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
 * \brief Calibrate a single camera from object track set
 */

#include "calibrate_single_camera_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/range/transform.h>
#include <vital/exceptions.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include "stereo_calibration.h"

#include <memory>
#include <fstream>

namespace kv = kwiver::vital;

namespace viame {


create_config_trait( output_directory, std::string, "./",
  "Output directory for calibration files" );
create_config_trait( output_json_file, std::string, "calibration.json",
  "Output JSON calibration file path" );
create_config_trait( frame_count_threshold, unsigned, "50",
  "Maximum number of frames to use during calibration. 0 to use all." );
create_config_trait( square_size, double, "80.0",
  "Calibration pattern square size in world units (e.g., mm)" );

create_port_trait( tracks, object_track_set, "Object track set with detected corners." );

// =============================================================================
// Private implementation class
class calibrate_single_camera_process::priv
{
public:
  explicit priv( calibrate_single_camera_process* parent );
  ~priv();

  // Configuration settings
  std::string m_output_directory;
  std::string m_output_json_file;
  unsigned m_frame_count_threshold;
  double m_square_size;

  // Parent pointer
  calibrate_single_camera_process* parent;

  // Logger
  kv::logger_handle_t m_logger;

  // Shared calibration utility
  stereo_calibration m_calibrator;

  // Extract image points and world points from object tracks
  bool extract_calibration_data(
    const kv::object_track_set_sptr& object_track,
    std::vector<std::vector<cv::Point2f>>& image_points,
    std::vector<std::vector<cv::Point3f>>& object_points,
    cv::Size& grid_size );

  // Write calibration results to files
  void write_calibration(
    const MonoCalibrationResult& result,
    const cv::Size& image_size,
    const cv::Size& grid_size );

  static std::pair<std::string, float> get_attribute_value( const std::string& note );
};


// -----------------------------------------------------------------------------
std::pair<std::string, float>
calibrate_single_camera_process::priv::get_attribute_value(
  const std::string& note )
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


// -----------------------------------------------------------------------------
bool
calibrate_single_camera_process::priv::extract_calibration_data(
  const kv::object_track_set_sptr& object_track,
  std::vector<std::vector<cv::Point2f>>& image_points,
  std::vector<std::vector<cv::Point3f>>& object_points,
  cv::Size& grid_size )
{
  if( !object_track || object_track->empty() )
  {
    LOG_ERROR( m_logger, "Empty object track set" );
    return false;
  }

  // Group detections by frame
  std::map<kv::frame_id_t, std::vector<cv::Point2f>> frame_image_pts;
  std::map<kv::frame_id_t, std::vector<cv::Point3f>> frame_world_pts;

  for( const auto& track : object_track->tracks() )
  {
    for( auto state : *track | kv::as_object_track )
    {
      if( state->detection()->notes().empty() )
      {
        continue;
      }

      std::map<std::string, double> attrs;
      for( const auto& note : state->detection()->notes() )
      {
        attrs.insert( get_attribute_value( note ) );
      }

      // Only keep points that have xyz world coordinates
      if( attrs.count( "stereo3d_x" ) && attrs.count( "stereo3d_y" ) &&
          attrs.count( "stereo3d_z" ) )
      {
        cv::Point3f world_pt(
          static_cast<float>( attrs["stereo3d_x"] ),
          static_cast<float>( attrs["stereo3d_y"] ),
          static_cast<float>( attrs["stereo3d_z"] ) );

        auto center = state->detection()->bounding_box().center();
        cv::Point2f image_pt(
          static_cast<float>( center[0] ),
          static_cast<float>( center[1] ) );

        frame_image_pts[state->frame()].push_back( image_pt );
        frame_world_pts[state->frame()].push_back( world_pt );
      }
    }
  }

  if( frame_image_pts.empty() )
  {
    LOG_ERROR( m_logger, "No valid calibration points found in tracks" );
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
    int grid_w = static_cast<int>( max_x / m_square_size ) + 1;
    int grid_h = static_cast<int>( max_y / m_square_size ) + 1;
    grid_size = cv::Size( grid_w, grid_h );

    LOG_DEBUG( m_logger, "Detected grid size: " << grid_w << "x" << grid_h );
  }

  // Convert to vectors, applying frame count threshold
  unsigned frame_count = 0;
  unsigned max_frames = ( m_frame_count_threshold > 0 ) ?
    m_frame_count_threshold : static_cast<unsigned>( frame_image_pts.size() );

  for( const auto& kv : frame_image_pts )
  {
    if( frame_count >= max_frames )
    {
      break;
    }

    // Only use frames with the expected number of corners
    size_t expected_corners = static_cast<size_t>( grid_size.width * grid_size.height );
    if( kv.second.size() == expected_corners )
    {
      image_points.push_back( kv.second );
      object_points.push_back( frame_world_pts[kv.first] );
      frame_count++;
    }
  }

  LOG_DEBUG( m_logger, "Extracted " << image_points.size() << " valid frames for calibration" );
  return !image_points.empty();
}


// -----------------------------------------------------------------------------
void
calibrate_single_camera_process::priv::write_calibration(
  const MonoCalibrationResult& result,
  const cv::Size& image_size,
  const cv::Size& grid_size )
{
  std::string output_dir = m_output_directory.empty() ? "." : m_output_directory;

  // Write OpenCV YAML format (intrinsics only)
  std::string intrinsics_file = output_dir + "/intrinsics.yml";
  cv::FileStorage fs( intrinsics_file, cv::FileStorage::WRITE );
  if( fs.isOpened() )
  {
    fs << "M1" << result.camera_matrix;
    fs << "D1" << result.dist_coeffs;
    fs.release();
    LOG_DEBUG( m_logger, "Wrote intrinsics to: " << intrinsics_file );
  }

  // Write JSON format
  if( !m_output_json_file.empty() )
  {
    std::string json_file = m_output_json_file;
    // If not absolute path, prepend output directory
    if( json_file[0] != '/' && json_file.find( ":/" ) == std::string::npos )
    {
      json_file = output_dir + "/" + json_file;
    }

    std::ofstream ofs( json_file );
    if( ofs.is_open() )
    {
      ofs << "{\n";
      ofs << "  \"image_width\": " << image_size.width << ",\n";
      ofs << "  \"image_height\": " << image_size.height << ",\n";
      ofs << "  \"grid_width\": " << grid_size.width << ",\n";
      ofs << "  \"grid_height\": " << grid_size.height << ",\n";
      ofs << "  \"square_size_mm\": " << m_square_size << ",\n";
      ofs << "  \"rms_error\": " << result.rms_error << ",\n";

      // Intrinsics
      ofs << "  \"fx\": " << result.camera_matrix.at<double>( 0, 0 ) << ",\n";
      ofs << "  \"fy\": " << result.camera_matrix.at<double>( 1, 1 ) << ",\n";
      ofs << "  \"cx\": " << result.camera_matrix.at<double>( 0, 2 ) << ",\n";
      ofs << "  \"cy\": " << result.camera_matrix.at<double>( 1, 2 ) << ",\n";

      // Distortion coefficients
      ofs << "  \"k1\": " << result.dist_coeffs.at<double>( 0 ) << ",\n";
      ofs << "  \"k2\": " << result.dist_coeffs.at<double>( 1 ) << ",\n";
      ofs << "  \"p1\": " << result.dist_coeffs.at<double>( 2 ) << ",\n";
      ofs << "  \"p2\": " << result.dist_coeffs.at<double>( 3 ) << ",\n";
      double k3 = ( result.dist_coeffs.cols > 4 || result.dist_coeffs.rows > 4 ) ?
        result.dist_coeffs.at<double>( 4 ) : 0.0;
      ofs << "  \"k3\": " << k3 << "\n";
      ofs << "}\n";
      ofs.close();
      LOG_DEBUG( m_logger, "Wrote JSON calibration to: " << json_file );
    }
  }
}


// -----------------------------------------------------------------------------
calibrate_single_camera_process::priv::priv(
  calibrate_single_camera_process* ptr )
  : m_output_directory( "./" )
  , m_output_json_file( "calibration.json" )
  , m_frame_count_threshold( 50 )
  , m_square_size( 80.0 )
  , parent( ptr )
{
}


calibrate_single_camera_process::priv::~priv()
{
}


// =============================================================================
calibrate_single_camera_process::calibrate_single_camera_process(
  kv::config_block_sptr const& config )
  : process( config )
  , d( new calibrate_single_camera_process::priv( this ) )
{
  make_ports();
  make_config();

  d->m_logger = logger();
  d->m_calibrator.set_logger( d->m_logger );
}


calibrate_single_camera_process::~calibrate_single_camera_process()
{
}


// -----------------------------------------------------------------------------
void calibrate_single_camera_process::make_ports()
{
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( tracks, required );
}


// -----------------------------------------------------------------------------
void calibrate_single_camera_process::make_config()
{
  declare_config_using_trait( output_directory );
  declare_config_using_trait( output_json_file );
  declare_config_using_trait( frame_count_threshold );
  declare_config_using_trait( square_size );
}


// -----------------------------------------------------------------------------
void calibrate_single_camera_process::_configure()
{
  d->m_output_directory = config_value_using_trait( output_directory );
  d->m_output_json_file = config_value_using_trait( output_json_file );
  d->m_frame_count_threshold = config_value_using_trait( frame_count_threshold );
  d->m_square_size = config_value_using_trait( square_size );
}


// -----------------------------------------------------------------------------
void calibrate_single_camera_process::_step()
{
  kv::object_track_set_sptr object_track_set;
  object_track_set = grab_from_port_using_trait( tracks );

  if( !object_track_set )
  {
    LOG_WARN( d->m_logger, "Received null object track set" );
    mark_process_as_complete();
    return;
  }

  // Extract calibration data from tracks
  std::vector<std::vector<cv::Point2f>> image_points;
  std::vector<std::vector<cv::Point3f>> object_points;
  cv::Size grid_size;

  if( !d->extract_calibration_data( object_track_set, image_points, object_points, grid_size ) )
  {
    LOG_ERROR( d->m_logger, "Failed to extract calibration data from tracks" );
    mark_process_as_complete();
    return;
  }

  // Determine image size from the detection bounding boxes or use a default
  // In practice, this should come from the actual images
  cv::Size image_size( 0, 0 );
  for( const auto& track : object_track_set->tracks() )
  {
    for( auto state : *track | kv::as_object_track )
    {
      auto bbox = state->detection()->bounding_box();
      image_size.width = std::max( image_size.width,
        static_cast<int>( bbox.max_x() + bbox.width() / 2 ) );
      image_size.height = std::max( image_size.height,
        static_cast<int>( bbox.max_y() + bbox.height() / 2 ) );
    }
  }

  // Add some margin and round up to reasonable values
  image_size.width = ( ( image_size.width + 100 ) / 100 ) * 100;
  image_size.height = ( ( image_size.height + 100 ) / 100 ) * 100;

  LOG_DEBUG( d->m_logger, "Estimated image size: "
    << image_size.width << "x" << image_size.height );

  // Perform calibration using shared utility
  MonoCalibrationResult result = d->m_calibrator.calibrate_single_camera(
    image_points, object_points, image_size, "camera" );

  if( !result.success )
  {
    LOG_ERROR( d->m_logger, "Camera calibration failed" );
    mark_process_as_complete();
    return;
  }

  LOG_INFO( d->m_logger, "Calibration complete. RMS error: " << result.rms_error );

  // Write calibration files
  d->write_calibration( result, image_size, grid_size );

  mark_process_as_complete();
}


} // end namespace viame
