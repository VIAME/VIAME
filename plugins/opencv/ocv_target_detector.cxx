#include "ocv_target_detector.h"

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
class ocv_target_detector::priv
{
public:

  /// Constructor
  priv()
    : m_target_width(7),
      m_target_height(5),
      m_square_size(1.0),
      m_object_type("unknown")
  {}

  /// Destructor
  ~priv() {}

  /// Parameters
  std::string m_config_file;
  unsigned m_target_width;
  unsigned m_target_height;
  float m_square_size;
  std::string m_object_type;

  kv::logger_handle_t m_logger;
}; // end class ocv_target_detector::priv


// =================================================================================================

ocv_target_detector::
ocv_target_detector()
  : d( new priv )
{
  attach_logger( "viame.opencv.ocv_target_detector" );  
  
  d->m_logger = logger();  
}


ocv_target_detector::
  ~ocv_target_detector()
{}


// -------------------------------------------------------------------------------------------------
kv::config_block_sptr
ocv_target_detector::
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
  
  return config;
}


// -------------------------------------------------------------------------------------------------
void
ocv_target_detector::
set_configuration( kv::config_block_sptr config_in )
{
  kv::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_config_file = config->get_value< std::string >( "config_file" );
  d->m_target_width = config->get_value< unsigned >( "target_width" );
  d->m_target_height = config->get_value< unsigned >( "target_height" );
  d->m_square_size = config->get_value< float >( "square_size" );
  d->m_object_type = config->get_value< std::string >( "object_type" );
}


// -------------------------------------------------------------------------------------------------
bool
ocv_target_detector::
check_configuration( kv::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kv::detected_object_set_sptr
ocv_target_detector::
detect( kv::image_container_sptr image_data ) const
{
  auto detected_set = std::make_shared< kv::detected_object_set >();
  
  if(!image_data)
  {
    return detected_set;
  }
  
  LOG_DEBUG( d->m_logger, "Start OCV target detection." );
  
  
  std::vector<cv::Point2f> corners;
  const cv::Size boardSize(d->m_target_width, d->m_target_height);
  const unsigned targetWidth = 5;
  bool cornersFound = false;
  
  // Construct corners in world coordinate space (with board in Z = 0) corresponding to leftImgCorners
  std::vector<cv::Point3f> world_corners;
  for( int j = 0; j < boardSize.height; j++ )
  {
    for( int k = 0; k < boardSize.width; k++ )
    {
      world_corners.push_back(cv::Point3f(k* d->m_square_size, j* d->m_square_size, 0));
    }
  }
  
  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
    kwiver::arrows::ocv::image_container::RGB_COLOR );
  if(src.channels() == 3)
  {
    cv::cvtColor( src, src, cv::COLOR_RGB2GRAY );
  }

  cornersFound = cv::findChessboardCorners(src, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE | cv::CALIB_CB_FAST_CHECK);
    
  if( !cornersFound && corners.size() != world_corners.size() )
  {
    LOG_WARN( d->m_logger, "Unable to find an OCV target. Found " << corners.size() << " corners" );
    return detected_set;
  }
  
  // refine subpixel corner location
  cv::cornerSubPix(src, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

  for(unsigned i = 0; i < corners.size();i++)
  {
    // Create kwiver style bounding box
    kv::bounding_box_d bbox( kv::bounding_box_d::vector_type( corners[i].x - targetWidth/2.0, corners[i].y - targetWidth/2.0), targetWidth, targetWidth );
    
    // Create possible object types.
    auto dot = std::make_shared< kv::detected_object_type >( d->m_object_type, 1.0 );
    
    // Add detected OCV target corners and world coordinates corners into notes
    kv::detected_object_sptr detected_object = std::make_shared< kv::detected_object >( bbox, 1.0, dot );
    detected_object->add_note(":x=" + std::to_string( world_corners[i].x ));
    detected_object->add_note(":y=" + std::to_string( world_corners[i].y ));
    detected_object->add_note(":z=" + std::to_string( world_corners[i].z ));
    detected_set->add( detected_object );
  }
  
  LOG_DEBUG( d->m_logger, "End of OCV target detection. Found " << detected_set->size() << " corners" );
  return detected_set;
}


} // end namespace
