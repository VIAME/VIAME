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
#include "calibrate_single_camera.h"
#include "calibrate_stereo_cameras.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/object_track_set.h>
#include <vital/types/camera_intrinsics.h>
#include <vital/exceptions.h>

#include <sprokit/processes/kwiver_type_traits.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>

#include <memory>

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
  calibrate_stereo_cameras m_calibrator;
};


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

  // Set up extraction options
  CalibrationExtractionOptions extract_options;
  extract_options.square_size = d->m_square_size;
  extract_options.frame_count_threshold = d->m_frame_count_threshold;

  // Extract calibration data from tracks
  std::vector< std::vector< cv::Point2f > > image_points;
  std::vector< std::vector< cv::Point3f > > object_points;
  cv::Size grid_size;

  if( !extract_calibration_data_from_tracks(
        object_track_set, extract_options, image_points, object_points,
        grid_size, d->m_logger ) )
  {
    LOG_ERROR( d->m_logger, "Failed to extract calibration data from tracks" );
    mark_process_as_complete();
    return;
  }

  // Estimate image size from detections
  cv::Size image_size = estimate_image_size_from_tracks( object_track_set );

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

  // Set up write options
  MonoCalibrationWriteOptions write_options;
  write_options.output_directory = d->m_output_directory;
  write_options.json_filename = d->m_output_json_file;
  write_options.square_size = d->m_square_size;
  write_options.grid_size = grid_size;
  write_options.image_size = image_size;

  // Write calibration files
  write_mono_calibration( result, write_options, d->m_logger );

  mark_process_as_complete();
}


} // end namespace viame
