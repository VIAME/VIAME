/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "windowed_detector.h"

#include <vital/algo/algorithm.txx>

#include "windowed_utils.h"

#include <vital/util/wall_timer.h>
#include <vital/types/image_container.h>

#include <algorithm>
#include <string>
#include <sstream>
#include <exception>
#include <limits>

namespace viame {

namespace kv = kwiver::vital;

// =============================================================================
class windowed_detector::priv
{
public:
  priv( windowed_detector& parent )
    : parent( parent )
  {}

  ~priv() {}

  windowed_detector& parent;
  kv::logger_handle_t m_logger;

  // Helper to get current settings as window_settings struct
  window_settings get_settings() const;

  // Accessor to nested detector
  kv::algo::image_object_detector_sptr detector() const
  {
    return m_detector;
  }

  kv::algo::image_object_detector_sptr m_detector;
};


// -----------------------------------------------------------------------------
window_settings
windowed_detector::priv
::get_settings() const
{
  window_settings settings;

  rescale_option_converter conv;
  settings.mode = conv.from_string( parent.c_mode );
  settings.scale = parent.c_scale;
  settings.chip_width = parent.c_chip_width;
  settings.chip_height = parent.c_chip_height;
  settings.chip_step_width = parent.c_chip_step_width;
  settings.chip_step_height = parent.c_chip_step_height;
  settings.chip_edge_filter = parent.c_chip_edge_filter;
  settings.chip_edge_max_prob = parent.c_chip_edge_max_prob;
  settings.chip_adaptive_thresh = parent.c_chip_adaptive_thresh;
  settings.batch_size = parent.c_batch_size;
  settings.min_detection_dim = parent.c_min_detection_dim;
  settings.original_to_chip_size = parent.c_original_to_chip_size;
  settings.black_pad = parent.c_black_pad;

  return settings;
}


// =============================================================================
void
windowed_detector
::initialize()
{
  KWIVER_INITIALIZE_UNIQUE_PTR( priv, d );
  attach_logger( "viame.core.windowed_detector" );
  d->m_logger = logger();
}


windowed_detector
::~windowed_detector()
{}


// -----------------------------------------------------------------------------
void
windowed_detector
::set_configuration_internal( kv::config_block_sptr config )
{
  kv::algo::image_object_detector_sptr det;
  kv::set_nested_algo_configuration<kv::algo::image_object_detector>(
    "detector", config, det );
  
  if( det )
  {
    d->m_detector = det;
  }
}


// -----------------------------------------------------------------------------
bool
windowed_detector
::check_configuration( kv::config_block_sptr config ) const
{
  return kv::check_nested_algo_configuration<kv::algo::image_object_detector>(
    "detector", config );
}


// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
windowed_detector
::detect( kv::image_container_sptr image_data ) const
{
  kv::scoped_wall_timer t( "Time to Detect Objects" );

  if( !image_data )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< kv::detected_object_set >();
  }

  kv::image input_image = image_data->get_image();

  if( input_image.height() == 0 || input_image.width() == 0 )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< kv::detected_object_set >();
  }

  // Get current settings
  window_settings settings = d->get_settings();

  // Prepare image regions using utility function
  std::vector< kv::image > regions_to_process;
  std::vector< windowed_region_prop > region_properties;

  prepare_image_regions( input_image, settings,
    regions_to_process, region_properties );

  // Run detector
  kv::detected_object_set_sptr detections = std::make_shared< kv::detected_object_set >();

  unsigned max_count = settings.batch_size;

  for( unsigned i = 0; i < regions_to_process.size(); i += max_count )
  {
    unsigned batch_size = std::min( max_count,
      static_cast< unsigned >( regions_to_process.size() ) - i );

    std::vector< kv::image_container_sptr > imgs;

    for( unsigned j = 0; j < batch_size; j++ )
    {
      imgs.push_back(
        kv::image_container_sptr(
          new kv::simple_image_container( regions_to_process[i+j] ) ) );
    }

    std::vector< kv::detected_object_set_sptr > out =
      d->detector()->batch_detect( imgs );

    for( unsigned j = 0; j < batch_size; j++ )
    {
      detections->add( rescale_detections( out[ j ],
        region_properties[ i + j ], settings.chip_edge_max_prob ) );
    }
  }

  const int min_dim = settings.min_detection_dim;

  detections->filter([&min_dim](kv::detected_object_sptr dos)
  {
    return !dos || dos->bounding_box().width() < min_dim
                || dos->bounding_box().height() < min_dim;
  });

  return detections;
} // windowed_detector::detect

} // end namespace viame
