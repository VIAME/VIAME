#include "detect_calibration_targets.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

namespace kv = kwiver::vital;

namespace viame {

// -------------------------------------------------------------------------------------------------
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

  LOG_DEBUG( logger(), "Start OCV target detection." );

  // Convert image to OpenCV format and grayscale
  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv(
    image_data->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR );
  cv::Mat gray = calibrate_stereo_cameras::to_grayscale( src );

  // Determine grid size to use
  cv::Size grid_size;
  ChessboardDetectionResult detection;

  if( c_auto_detect_grid && !m_grid_detected )
  {
    // Auto-detect grid size on first frame
    LOG_DEBUG( logger(), "Auto-detecting grid size..." );
    detection = m_calibrator.detect_chessboard_auto( gray );

    if( detection.found )
    {
      m_grid_detected = true;
      m_detected_grid_size = detection.grid_size;
      LOG_INFO( logger(), "Auto-detected grid size: "
                << detection.grid_size.width << "x" << detection.grid_size.height );
    }
  }
  else if( c_auto_detect_grid && m_grid_detected )
  {
    // Use previously detected grid size
    grid_size = m_detected_grid_size;
    detection = m_calibrator.detect_chessboard( gray, grid_size );
  }
  else
  {
    // Use configured grid size
    grid_size = cv::Size( c_target_width, c_target_height );
    detection = m_calibrator.detect_chessboard( gray, grid_size );
  }

  if( !detection.found )
  {
    LOG_WARN( logger(), "Unable to find an OCV target" );
    return detected_set;
  }

  // Get the actual grid size used (may have been auto-detected)
  grid_size = detection.grid_size;

  // Generate world points for the detected grid
  std::vector<cv::Point3f> world_corners =
    calibrate_stereo_cameras::make_object_points( grid_size, c_square_size );

  if( detection.corners.size() != world_corners.size() )
  {
    LOG_WARN( logger(), "Corner count mismatch: detected "
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
    auto dot = std::make_shared< kv::detected_object_type >( c_object_type, 1.0 );

    // Add detected OCV target corners and world coordinates corners into notes
    kv::detected_object_sptr detected_object =
      std::make_shared< kv::detected_object >( bbox, 1.0, dot );
    detected_object->add_note( ":stereo3d_x=" + std::to_string( world_corners[i].x ) );
    detected_object->add_note( ":stereo3d_y=" + std::to_string( world_corners[i].y ) );
    detected_object->add_note( ":stereo3d_z=" + std::to_string( world_corners[i].z ) );
    detected_set->add( detected_object );
  }

  LOG_DEBUG( logger(), "End of OCV target detection. Found "
             << detected_set->size() << " corners" );
  return detected_set;
}


} // end namespace
