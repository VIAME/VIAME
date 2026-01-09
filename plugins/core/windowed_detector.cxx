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

#include "windowed_detector.h"
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
  priv()
  {}

  ~priv() {}

  // Settings from the config
  window_settings m_settings;

  kv::algo::image_object_detector_sptr m_detector;
  kv::logger_handle_t m_logger;
};


// =============================================================================
windowed_detector
::windowed_detector()
  : d( new priv() )
{
  attach_logger( "viame.core.windowed_detector" );

  d->m_logger = logger();
}


windowed_detector
::~windowed_detector()
{}


// -----------------------------------------------------------------------------
kv::config_block_sptr
windowed_detector
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  // Merge window settings configuration
  config->merge_config( d->m_settings.config() );

  kv::algo::image_object_detector::get_nested_algo_configuration(
    "detector", config, d->m_detector );

  return config;
}


// -----------------------------------------------------------------------------
void
windowed_detector
::set_configuration( kv::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  kv::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  // Set window settings from configuration
  d->m_settings.set_config( config );

  kv::algo::image_object_detector::set_nested_algo_configuration(
    "detector", config, d->m_detector );
}


// -----------------------------------------------------------------------------
bool
windowed_detector
::check_configuration( kv::config_block_sptr config ) const
{
  return kv::algo::image_object_detector::check_nested_algo_configuration(
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

  // Prepare image regions using utility function
  std::vector< kv::image > regions_to_process;
  std::vector< windowed_region_prop > region_properties;

  prepare_image_regions( input_image, d->m_settings,
    regions_to_process, region_properties );

  // Run detector
  kv::detected_object_set_sptr detections = std::make_shared< kv::detected_object_set >();

  unsigned max_count = d->m_settings.batch_size;

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
      d->m_detector->batch_detect( imgs );

    for( unsigned j = 0; j < batch_size; j++ )
    {
      detections->add( rescale_detections( out[ j ],
        region_properties[ i + j ], d->m_settings.chip_edge_max_prob ) );
    }
  }

  const int min_dim = d->m_settings.min_detection_dim;

  detections->filter([&min_dim](kv::detected_object_sptr dos)
  {
    return !dos || dos->bounding_box().width() < min_dim
                || dos->bounding_box().height() < min_dim;
  });

  return detections;
} // windowed_detector::detect

} // end namespace viame
