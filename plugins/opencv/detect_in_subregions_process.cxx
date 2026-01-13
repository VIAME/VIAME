/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "detect_in_subregions_process.h"

#include <arrows/ocv/image_container.h>

#include <vital/algo/image_object_detector.h>
#include <vital/util/wall_timer.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace viame
{

create_config_trait( detector, std::string, "", "Algorithm configuration "
  "subblock.\n\nUse 'detector:type' to select desired detector implementation." );

create_config_trait( method, std::string, "", "Method to extract subregions "
  "to process. Options: detection_box, fixed_size." );
create_config_trait( max_subregion_count, int, "-1", "Maximum number of "
  "subregions to process." );
create_config_trait( fixed_size, int, "0", "If mode is fixed size, "
  "the width and height of the chip around each input detection to process" );
create_config_trait( threshold, double, "0.0", "If the input is object detections "
  "only consider regions above this confidence" );
create_config_trait( include_input_dets, bool, "false", "Include the input "
  "detected object set in the output detected object set." );

// -----------------------------------------------------------------------------
// Private implementation class
class detect_in_subregions_process::priv
{
public:
  priv()
    : m_method(DETECTION_BOX)
    , m_max_subregion_count(-1)
    , m_fixed_size(0)
    , m_threshold(0.0)
    , m_include_input_dets(false)
  {
  }

  ~priv()
  {
  }

  enum{ DETECTION_BOX, FIXED_SIZE } m_method;
  int m_max_subregion_count;
  int m_fixed_size;
  double m_threshold;
  bool m_include_input_dets;
  kwiver::vital::logger_handle_t m_logger;

  kwiver::vital::algo::image_object_detector_sptr m_detector;


  // ---------------------------------------------------------------------------
  /**
   * @brief Classify regions defined within the provided detected object set.
   *
  * Replace multi-channel image with a single channel image equal to the root
  * mean square over the channels.
  *
  * @param src_image Full-resolution image
  * @param dets_in dets_in_sptr Detections with bounding boxes defined in
   *  the full-image coordinate system. These will define ROI that will be
   *  chipped and sent individually to the detector.
  * @param dets_out_sptr Output classifications.
  */
  void
  classify( const kwiver::vital::image_container_sptr &src_image,
            const kwiver::vital::detected_object_set_sptr &dets_in,
            kwiver::vital::detected_object_set_sptr &dets_out_sptr )
  {
    kwiver::vital::wall_timer timer;
    cv::Mat cv_src = kwiver::arrows::ocv::image_container::vital_to_ocv(
      src_image->get_image(), kwiver::arrows::ocv::image_container::BGR_COLOR );

    kwiver::vital::detected_object::vector_t dets_out;

    if( m_include_input_dets )
    {
      // Add in the detections that defined the ROI
      dets_out.insert( dets_out.end(), dets_in->cbegin(), dets_in->cend() );
    }

    // Define the bound box representing the entire image.
    cv::Size s = cv_src.size();
    kwiver::vital::bounding_box_d img( kwiver::vital::bounding_box_d::vector_type( 0, 0 ),
                               kwiver::vital::bounding_box_d::vector_type( s.width, s.height ) );

    kwiver::vital::image_container_sptr windowed_image;

    LOG_DEBUG( m_logger, "Considering " << dets_in->size() << " ROI" );

    int processed_count = 0;

    std::vector< cv::Rect > previous_regions;

    auto ordered_dets = dets_in->select( m_threshold );

    for( auto det : *ordered_dets )
    {
      if( m_max_subregion_count > 0 && processed_count >= m_max_subregion_count )
      {
        break;
      }

      timer.start();
      kwiver::vital::bounding_box_d bbox = det->bounding_box();

      // Clip bounding box to the image
      bbox = intersection( img, bbox );

      if( bbox.height() <= 0 || bbox.width() <= 0 )
      {
        continue;
      }

      int x = bbox.upper_left()[0];
      int y = bbox.upper_left()[1];
      int w = bbox.width();
      int h = bbox.height();

      if( m_method == FIXED_SIZE )
      {
        int cx = x + w / 2;
        int cy = y + h / 2;

        bool intersect_found = false;

        for( auto region : previous_regions )
        {
          if( region.contains( cv::Point( cx, cy ) ) )
          {
            intersect_found = true;
            break;
          }
        }
        if( intersect_found )
        {
          continue;
        }

        x = std::max( 0, cx - m_fixed_size / 2 );
        y = std::max( 0, cy - m_fixed_size / 2 );

        w = m_fixed_size;
        h = m_fixed_size;

        if( x + w > img.width() )
        {
          w = img.width() - x;
        }
        if( y + h > img.height() )
        {
          h = img.height() - y;
        }
        if( w <= 1 || h <= 1 )
        {
          continue;
        }
      }

      LOG_TRACE( m_logger, "Processing ROI window with upper left coordinates (" +
                 std::to_string(x) + "," + std::to_string(y) + ") of size (" +
                 std::to_string(w) + "x" + std::to_string(h) + ")" );

      // TODO: ocv is only used to crop the image. This can be replaced by a
      // vital image cropping utility once that becomes available, and then this
      // can be moved to a core process.

      // Make CV rect for bbox
      cv::Rect roi( x, y, w, h );

      // Detect within the region of interest.
      windowed_image = kwiver::vital::image_container_sptr(
        new kwiver::arrows::ocv::image_container(
          cv_src( roi ), kwiver::arrows::ocv::image_container::BGR_COLOR ) );

      auto dets = m_detector->detect( windowed_image );

      for( auto det : *dets )
      {
        auto det_bbox = det->bounding_box();
        kwiver::vital::bounding_box_d::vector_type offset( 2 );
        offset[0] = x; offset[1] = y;
        kwiver::vital::translate( det_bbox, offset );
        det->set_bounding_box( det_bbox );
      }

      // Add detections set to the output detection set
      dets_out.insert( dets_out.end(), dets->cbegin(), dets->cend() );

      timer.stop();

      processed_count++;
      previous_regions.push_back( roi );

      LOG_TRACE( m_logger, "Time to classify window: " << timer.elapsed() );
    } // end foreach

    dets_out_sptr = std::make_shared<kwiver::vital::detected_object_set>( dets_out );
  }

}; // end priv class


// =============================================================================
detect_in_subregions_process
::detect_in_subregions_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detect_in_subregions_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach
  d->m_logger = logger();

  make_ports();
  make_config();
}


detect_in_subregions_process
::~detect_in_subregions_process()
{
}


//------------------------------------------------------------------------------
void
detect_in_subregions_process
::_configure()
{
  scoped_configure_instrumentation();

  kwiver::vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if( !kwiver::vital::algo::image_object_detector::
       check_nested_algo_configuration( "detector", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  kwiver::vital::algo::image_object_detector::
    set_nested_algo_configuration( "detector", algo_config, d->m_detector );

  if( !d->m_detector )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create detector" );
  }

  std::string method = config_value_using_trait( method );

  if( method == "detection_box" )
  {
    d->m_method = priv::DETECTION_BOX;
  }
  else if( method == "fixed_size" )
  {
    d->m_method = priv::FIXED_SIZE;
  }
  else
  {
    throw sprokit::invalid_configuration_exception( name(), "Invalid method: " + method );
  }

  d->m_max_subregion_count = config_value_using_trait( max_subregion_count );
  d->m_fixed_size = config_value_using_trait( fixed_size );
  d->m_threshold = config_value_using_trait( threshold );
  d->m_include_input_dets = config_value_using_trait( include_input_dets );
}


// -----------------------------------------------------------------------------
void
detect_in_subregions_process
::_step()
{
  kwiver::vital::detected_object_set_sptr dets_out;
  double elapsed_time(0);;

  LOG_TRACE( logger(), "Starting process" );

  kwiver::vital::wall_timer timer;
  timer.start();

  kwiver::vital::image_container_sptr src_image =
    grab_from_port_using_trait( image );
  kwiver::vital::detected_object_set_sptr dets_in =
    grab_from_port_using_trait( detected_object_set );

  if( src_image )
  {
    scoped_step_instrumentation();

    // Get detections from detector on image
    d->classify( src_image, dets_in, dets_out );

    timer.stop();
    elapsed_time = timer.elapsed();

    LOG_DEBUG( logger(), "Total processing time: " << elapsed_time );
  }

  push_to_port_using_trait( detection_time, elapsed_time);
  push_to_port_using_trait( detected_object_set, dets_out );
}


// -----------------------------------------------------------------------------
void
detect_in_subregions_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( detected_object_set, optional );
  declare_output_port_using_trait( detection_time, optional );
}


// -----------------------------------------------------------------------------
void
detect_in_subregions_process
::make_config()
{
  declare_config_using_trait( detector );
  declare_config_using_trait( method );
  declare_config_using_trait( max_subregion_count );
  declare_config_using_trait( fixed_size );
  declare_config_using_trait( threshold );
  declare_config_using_trait( include_input_dets );
}

// =============================================================================

} // end namespace viame
