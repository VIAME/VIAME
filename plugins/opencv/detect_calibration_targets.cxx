#include "detect_calibration_targets.h"
#include "calibrate_stereo_cameras.h"

#include <vital/algo/algorithm.txx>

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>


#include <cmath>

namespace kv = kwiver::vital;

namespace viame {

// -----------------------------------------------------------------------------------------------




// -------------------------------------------------------------------------------------------------
void
detect_calibration_targets::
initialize()
{
  attach_logger( "viame.opencv.detect_calibration_targets" );
  m_calibrator.set_logger( logger() );
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

  LOG_DEBUG( logger(), "Start OCV target detection (target_type="
             << c_target_type << ")." );

  // Convert image to OpenCV format and grayscale
  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv(
    image_data->get_image(), kwiver::arrows::ocv::image_container::RGB_COLOR );
  cv::Mat gray = calibrate_stereo_cameras::to_grayscale( src );

  // Determine grid size to use
  cv::Size grid_size;
  ChessboardDetectionResult detection;
  bool is_dot_detection = false;

  bool try_checkerboard = ( c_target_type == "checkerboard" || c_target_type == "auto" );
  bool try_dots = ( c_target_type == "dots" || c_target_type == "auto" );

  // --- Checkerboard detection ---
  if( try_checkerboard )
  {
    if( c_auto_detect_grid && !m_grid_detected )
    {
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
      grid_size = m_detected_grid_size;
      detection = m_calibrator.detect_chessboard( gray, grid_size );
    }
    else
    {
      grid_size = cv::Size( c_target_width, c_target_height );
      detection = m_calibrator.detect_chessboard( gray, grid_size );
    }
  }

  // --- Dot detection (if checkerboard not found or not attempted) ---
  if( !detection.found && try_dots )
  {
    LOG_DEBUG( logger(), "Attempting dot detection..." );
    detection = m_calibrator.detect_dots(
      gray, 5000, c_dot_min_area, c_dot_max_area, c_dot_min_circularity );

    if( detection.found )
    {
      int raw_count = static_cast< int >( detection.corners.size() );

      // Filter out isolated background dots (labels, reflections, etc.)
      calibrate_stereo_cameras::filter_target_cluster( detection.corners );

      is_dot_detection = true;
      int filtered_count = raw_count - static_cast< int >( detection.corners.size() );

      if( filtered_count > 0 )
      {
        LOG_DEBUG( logger(), "Dot detection found " << detection.corners.size()
                   << " dots (filtered " << filtered_count << " background)" );
      }
      else
      {
        LOG_DEBUG( logger(), "Dot detection found " << detection.corners.size() << " dots" );
      }
    }
  }

  // Apply ROI filter if specified
  bool has_roi = ( c_roi_x1 >= 0 && c_roi_y1 >= 0 &&
                   c_roi_x2 > c_roi_x1 && c_roi_y2 > c_roi_y1 );

  if( detection.found && has_roi )
  {
    std::vector< cv::Point2f > filtered;
    for( const auto& pt : detection.corners )
    {
      if( pt.x >= c_roi_x1 && pt.x <= c_roi_x2 &&
          pt.y >= c_roi_y1 && pt.y <= c_roi_y2 )
      {
        filtered.push_back( pt );
      }
    }
    int removed = static_cast< int >( detection.corners.size() ) -
                  static_cast< int >( filtered.size() );
    if( removed > 0 )
    {
      LOG_DEBUG( logger(), "ROI filter removed " << removed << " points ("
                 << filtered.size() << " remaining)" );
    }
    detection.corners = filtered;
    if( detection.corners.empty() )
    {
      detection.found = false;
    }
  }

  if( !detection.found )
  {
    LOG_WARN( logger(), "Unable to find an OCV target" );
    return detected_set;
  }

  // Get the actual grid size used (may have been auto-detected)
  grid_size = detection.grid_size;

  const unsigned targetWidth = 5;

  if( is_dot_detection )
  {
    // For dot detections: output each dot center as a detected_object.
    // No stereo3d notes since dots have no known 3D layout from grid geometry.
    for( unsigned i = 0; i < detection.corners.size(); ++i )
    {
      kv::bounding_box_d bbox(
        kv::bounding_box_d::vector_type(
          detection.corners[i].x - targetWidth / 2.0,
          detection.corners[i].y - targetWidth / 2.0 ),
        targetWidth, targetWidth );

      auto dot = std::make_shared< kv::detected_object_type >( c_object_type, 1.0 );

      kv::detected_object_sptr detected_object =
        std::make_shared< kv::detected_object >( bbox, 1.0, dot );
      detected_set->add( detected_object );
    }
  }
  else
  {
    // Checkerboard detection: generate world points and attach stereo3d notes
    std::vector<cv::Point3f> world_corners =
      calibrate_stereo_cameras::make_object_points( grid_size, c_square_size );

    if( detection.corners.size() != world_corners.size() )
    {
      LOG_WARN( logger(), "Corner count mismatch: detected "
                << detection.corners.size() << ", expected " << world_corners.size() );
      return detected_set;
    }

    for( unsigned i = 0; i < detection.corners.size(); ++i )
    {
      kv::bounding_box_d bbox(
        kv::bounding_box_d::vector_type(
          detection.corners[i].x - targetWidth / 2.0,
          detection.corners[i].y - targetWidth / 2.0 ),
        targetWidth, targetWidth );

      auto dot = std::make_shared< kv::detected_object_type >( c_object_type, 1.0 );

      kv::detected_object_sptr detected_object =
        std::make_shared< kv::detected_object >( bbox, 1.0, dot );
      detected_object->add_note( ":stereo3d_x=" + std::to_string( world_corners[i].x ) );
      detected_object->add_note( ":stereo3d_y=" + std::to_string( world_corners[i].y ) );
      detected_object->add_note( ":stereo3d_z=" + std::to_string( world_corners[i].z ) );
      detected_set->add( detected_object );
    }
  }

  LOG_DEBUG( logger(), "End of OCV target detection. Found "
             << detected_set->size() << ( is_dot_detection ? " dots" : " corners" ) );
  return detected_set;
}


} // end namespace
