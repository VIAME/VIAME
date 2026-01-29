/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

#include "windowed_detector.h"

#include <vital/algo/algorithm.txx>

#include "windowed_utils.h"

#include <vital/util/wall_timer.h>
#include <vital/exceptions/io.h>
#include <vital/config/config_block_formatter.h>

#include <arrows/ocv/image_container.h>
#include <kwiversys/SystemTools.hxx>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <algorithm>
#include <string>
#include <sstream>
#include <exception>
#include <limits>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

// =============================================================================
// =============================================================================

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
    LOG_WARN( logger(), "Input image is empty." );
    return std::make_shared< kv::detected_object_set >();
  }

  cv::Mat cv_image = ocv::image_container::vital_to_ocv(
    image_data->get_image(), ocv::image_container::RGB_COLOR );

  if( cv_image.rows == 0 || cv_image.cols == 0 )
  {
    LOG_WARN( logger(), "Input image is empty." );
    return std::make_shared< kv::detected_object_set >();
  }

  // Construct settings from current configuration
  window_settings settings;
  rescale_option_converter conv;
  settings.mode = conv.from_string( c_mode );
  settings.scale = c_scale;
  settings.chip_width = c_chip_width;
  settings.chip_height = c_chip_height;
  settings.chip_step_width = c_chip_step_width;
  settings.chip_step_height = c_chip_step_height;
  settings.chip_edge_filter = c_chip_edge_filter;
  settings.chip_edge_max_prob = c_chip_edge_max_prob;
  settings.chip_adaptive_thresh = c_chip_adaptive_thresh;
  settings.batch_size = c_batch_size;
  settings.min_detection_dim = c_min_detection_dim;
  settings.original_to_chip_size = c_original_to_chip_size;
  settings.black_pad = c_black_pad;

  // Prepare image regions using utility function
  std::vector< cv::Mat > regions_to_process;
  std::vector< windowed_region_prop > region_properties;

  prepare_image_regions( cv_image, settings, regions_to_process, region_properties );

  // Run detector
  kv::detected_object_set_sptr detections = std::make_shared< kv::detected_object_set >();

  unsigned max_count = settings.batch_size;

  for( unsigned i = 0; i < regions_to_process.size(); i+= max_count )
  {
    unsigned batch_size = std::min( max_count,
      static_cast< unsigned >( regions_to_process.size() ) - i );

    std::vector< kv::image_container_sptr > imgs;

    for( unsigned j = 0; j < batch_size; j++ )
    {
      imgs.push_back(
        kv::image_container_sptr(
          new ocv::image_container( regions_to_process[i+j],
            ocv::image_container::RGB_COLOR ) ) );
    }

    std::vector< kv::detected_object_set_sptr > out =
      c_detector->batch_detect( imgs );

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
