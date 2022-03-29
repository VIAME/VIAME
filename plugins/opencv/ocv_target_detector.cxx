#include "ocv_target_detector.h"

#include <arrows/ocv/image_container.h>

#include <opencv2/core/core.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>


#include <cmath>

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
      m_square_size(1.0)
  {}

  /// Destructor
  ~priv() {}

  /// Parameters
  std::string m_config_file;
  unsigned m_target_width;
  unsigned m_target_height;
  float m_square_size;

  kwiver::vital::logger_handle_t m_logger;
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
kwiver::vital::config_block_sptr
ocv_target_detector::
get_configuration() const
{
  // Get base config from base class
  kwiver::vital::config_block_sptr config = kwiver::vital::algorithm::get_configuration();

  config->set_value( "config_file", d->m_config_file,
                     "Name of OCV Target Detector configuration file." );

  config->set_value( "target_width", d->m_target_width, "Number of width corners of the detected ocv target" );
  config->set_value( "target_height", d->m_target_height, "Number of height corners of the detected ocv target" );
  config->set_value( "square_size", d->m_square_size, "Square size of the detected ocv target" );
  
  return config;
}


// -------------------------------------------------------------------------------------------------
void
ocv_target_detector::
set_configuration( kwiver::vital::config_block_sptr config_in )
{
  kwiver::vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_config_file = config->get_value< std::string >( "config_file" );
  d->m_target_width = config->get_value< unsigned >( "target_width" );
  d->m_target_height = config->get_value< unsigned >( "target_height" );
  d->m_square_size = config->get_value< float >( "square_size" );
}


// -------------------------------------------------------------------------------------------------
bool
ocv_target_detector::
check_configuration( kwiver::vital::config_block_sptr config ) const
{
  return true;
}


// -------------------------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
ocv_target_detector::
detect( kwiver::vital::image_container_sptr image_data ) const
{
  LOG_DEBUG( d->m_logger, "Start OCV target detection." );
  
  auto detected_set = std::make_shared< kwiver::vital::detected_object_set >();
  std::vector<cv::Point2f> corners;
  
  const cv::Size boardSize(d->m_target_width, d->m_target_height);
  const unsigned targetWidth = 5;
  bool cornersFound = false;

  cv::Mat src = kwiver::arrows::ocv::image_container::vital_to_ocv( image_data->get_image(),
    kwiver::arrows::ocv::image_container::RGB_COLOR );

  cornersFound = cv::findChessboardCorners(src, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE);
    
  if( !cornersFound )
  {
    LOG_WARN( d->m_logger, "Unable to find an OCV target. Corners : " << corners.size() );
    return detected_set;
  }
  
  // refine subpixel corner location
  cv::cornerSubPix(src, corners, cv::Size(11,11), cv::Size(-1,-1), cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01));

  for(const auto corner: corners)
  {
    // Create kwiver style bounding box
    kwiver::vital::bounding_box_d bbox( kwiver::vital::bounding_box_d::vector_type( corner.x - targetWidth/2.0, corner.y - targetWidth/2.0), targetWidth, targetWidth );
    
    // Create possible object types.
    auto dot = std::make_shared< kwiver::vital::detected_object_type >( "c", 1.0 );
    
    // Create detection
    detected_set->add( std::make_shared< kwiver::vital::detected_object >( bbox, 1.0, dot ) );
  }
  
  LOG_DEBUG( d->m_logger, "End of OCV target detection. Found " << detected_set->size() << " corners" );
  return detected_set;
}


} // end namespace
