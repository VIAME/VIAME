#include "detect_calibration_targets.h"
#include "calibrate_stereo_cameras.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>


#include <cmath>

namespace kv = kwiver::vital;

namespace viame {

// -----------------------------------------------------------------------------------------------
/**
 * @brief Storage class for private member variables
 */
class detect_calibration_targets::priv
{
public:

  /// Constructor
  priv()
    : m_target_width(7),
      m_target_height(5),
      m_square_size(1.0),
      m_object_type("unknown"),
      m_auto_detect_grid(false),
      m_grid_detected(false)
  {}

  /// Destructor
  ~priv() {}

  /// Parameters
  std::string m_config_file;
  unsigned m_target_width;
  unsigned m_target_height;
  float m_square_size;
  std::string m_object_type;
  bool m_auto_detect_grid;

  /// State for auto-detection
  mutable bool m_grid_detected;
  mutable cv::Size m_detected_grid_size;

  /// Calibration utility
  calibrate_stereo_cameras m_calibrator;

  kv::logger_handle_t m_logger;
}; // end class detect_calibration_targets::priv


// =================================================================================================

detect_calibration_targets::
detect_calibration_targets()
  : d( new priv )
{
  attach_logger( "viame.opencv.detect_calibration_targets" );

  d->m_logger = logger();
}


detect_calibration_targets::
~detect_calibration_targets()
{}


// -------------------------------------------------------------------------------------------------
kv::config_block_sptr
detect_calibration_targets::
get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  config->set_value( "config_file", d->m_config_file,
                     "Name of OCV Target Detector configuration file." );

  config->set_value( "target_width", d->m_target_width, "Number of width corners of the detected ocv target" );
  config->set_value( "target_height", d->m_target_height, "Number of height corners of the detected ocv target" );
  config->set_value( "square_size", d->m_square_size, "Square size of the detected ocv target" );
  config->set_value( "object_type", d->m_object_type, "The detected object type" );
  config->set_value( "auto_detect_grid", d->m_auto_detect_grid,
                     "Automatically detect grid size from the first image" );

  return config;
}


// -------------------------------------------------------------------------------------------------
void
detect_calibration_targets::
set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_config_file = config->get_value< std::string >( "config_file" );
  d->m_target_width = config->get_value< unsigned >( "target_width" );
  d->m_target_height = config->get_value< unsigned >( "target_height" );
  d->m_square_size = config->get_value< float >( "square_size" );
  d->m_object_type = config->get_value< std::string >( "object_type" );
  d->m_auto_detect_grid = config->get_value< bool >( "auto_detect_grid" );

  // Set logger on calibrator
  d->m_calibrator.set_logger( d->m_logger );
}


// -------------------------------------------------------------------------------------------------
bool
detect_calibration_targets::
check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kv::detected_object_set_sptr
detect_calibration_targets::
detect( kv::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kv::detected_object_set >();

  if( !image_data )
  {
    return detected_set;
  }

  LOG_DEBUG( d->m_logger, "Start OCV target detection." );

  // Convert image to OpenCV format and grayscale
  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv(
    image_data->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR );
  cv::Mat gray = calibrate_stereo_cameras::to_grayscale( src );

  // Determine grid size to use
  cv::Size grid_size;
  ChessboardDetectionResult detection;

  if( d->m_auto_detect_grid && !d->m_grid_detected )
  {
    // Auto-detect grid size on first frame
    LOG_DEBUG( d->m_logger, "Auto-detecting grid size..." );
    detection = d->m_calibrator.detect_chessboard_auto( gray );

    if( detection.found )
    {
      d->m_grid_detected = true;
      d->m_detected_grid_size = detection.grid_size;
      LOG_INFO( d->m_logger, "Auto-detected grid size: "
                << detection.grid_size.width << "x" << detection.grid_size.height );
    }
  }
  else if( d->m_auto_detect_grid && d->m_grid_detected )
  {
    // Use previously detected grid size
    grid_size = d->m_detected_grid_size;
    detection = d->m_calibrator.detect_chessboard( gray, grid_size );
  }
  else
  {
    // Use configured grid size
    grid_size = cv::Size( d->m_target_width, d->m_target_height );
    detection = d->m_calibrator.detect_chessboard( gray, grid_size );
  }

  if( !detection.found )
  {
    LOG_WARN( d->m_logger, "Unable to find an OCV target" );
    return detected_set;
  }

  // Get the actual grid size used (may have been auto-detected)
  grid_size = detection.grid_size;

  // Generate world points for the detected grid
  std::vector<cv::Point3f> world_corners =
    calibrate_stereo_cameras::make_object_points( grid_size, d->m_square_size );

  if( detection.corners.size() != world_corners.size() )
  {
    LOG_WARN( d->m_logger, "Corner count mismatch: detected "
              << detection.corners.size() << ", expected " << world_corners.size() );
    return detected_set;
  }

  const unsigned targetWidth = 5;
  for( unsigned i = 0; i < detection.corners.size(); ++i )
  {
    // Create kwiver style bounding box
    kv::bounding_box_d bbox(
      kv::bounding_box_d::vector_type(
        detection.corners[i].x - targetWidth / 2.0,
        detection.corners[i].y - targetWidth / 2.0 ),
      targetWidth, targetWidth );

    // Create possible object types
    auto dot = std::make_shared< kv::detected_object_type >( d->m_object_type, 1.0 );

    // Add detected OCV target corners and world coordinates corners into notes
    kv::detected_object_sptr detected_object =
      std::make_shared< kv::detected_object >( bbox, 1.0, dot );
    detected_object->add_note( ":stereo3d_x=" + std::to_string( world_corners[i].x ) );
    detected_object->add_note( ":stereo3d_y=" + std::to_string( world_corners[i].y ) );
    detected_object->add_note( ":stereo3d_z=" + std::to_string( world_corners[i].z ) );
    detected_set->add( detected_object );
  }

  LOG_DEBUG( d->m_logger, "End of OCV target detection. Found "
             << detected_set->size() << " corners" );
  return detected_set;
}


} // end namespace
