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

#include "windowed_refiner.h"

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
#include <set>

namespace viame {

namespace kv = kwiver::vital;
namespace ocv = kwiver::arrows::ocv;

// =============================================================================
class windowed_refiner::priv
{
public:
  priv()
    : m_process_boundary_dets( false )
    , m_overlapping_proc_once( true )
  {}

  ~priv() {}

  // Settings from the config
  window_settings m_settings;

  bool m_process_boundary_dets;
  bool m_overlapping_proc_once;

  kv::algo::refine_detections_sptr m_refiner;
  kv::logger_handle_t m_logger;
};


// =============================================================================
windowed_refiner
::windowed_refiner()
  : d( new priv() )
{
  attach_logger( "viame.opencv.windowed_refiner" );

  d->m_logger = logger();
}


windowed_refiner
::~windowed_refiner()
{}


// -----------------------------------------------------------------------------
kv::config_block_sptr
windowed_refiner
::get_configuration() const
{
  // Get base config from base class
  kv::config_block_sptr config = kv::algorithm::get_configuration();

  // Merge window settings configuration
  config->merge_config( d->m_settings.config() );

  // Other refiner-specific settings
  config->set_value( "process_boundary_dets", d->m_process_boundary_dets,
    "Pass through detections touching tile boundaries unmodified in refiner" );
  config->set_value( "overlapping_proc_once", d->m_overlapping_proc_once,
    "Only refine each detection once if it appears in multiple tiles" );

  kv::get_nested_algo_configuration<kv::algo::refine_detections>(
    "refiner", config, d->m_refiner );

  return config;
}


// -----------------------------------------------------------------------------
void
windowed_refiner
::set_configuration( kv::config_block_sptr config_in )
{
  // Starting with our generated config_block to ensure that assumed values
  // are present. An alternative is to check for key presence before performing
  // a get_value() call.
  kv::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );

  d->m_settings.set_config( config );

  d->m_process_boundary_dets = config->get_value< bool >( "process_boundary_dets" );
  d->m_overlapping_proc_once = config->get_value< bool >( "overlapping_proc_once" );

  kv::set_nested_algo_configuration<kv::algo::refine_detections>(
    "refiner", config, d->m_refiner );
}


// -----------------------------------------------------------------------------
bool
windowed_refiner
::check_configuration( kv::config_block_sptr config ) const
{
  return kv::check_nested_algo_configuration<kv::algo::refine_detections>(
    "refiner", config );
}


// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
windowed_refiner
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  kv::scoped_wall_timer t( "Time to Refine Objects" );

  if( !image_data )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< kv::detected_object_set >();
  }

  cv::Mat cv_image = ocv::image_container::vital_to_ocv(
    image_data->get_image(), ocv::image_container::RGB_COLOR );

  if( cv_image.rows == 0 || cv_image.cols == 0 )
  {
    LOG_WARN( d->m_logger, "Input image is empty." );
    return std::make_shared< kv::detected_object_set >();
  }

  // Prepare image regions using utility function
  std::vector< cv::Mat > regions_to_process;
  std::vector< windowed_region_prop > region_properties;

  prepare_image_regions( cv_image, d->m_settings, regions_to_process, region_properties );

  kv::detected_object_set_sptr refined_detections =
    std::make_shared< kv::detected_object_set >();

  // Track which original detections have been processed (if option enabled)
  std::set< kv::detected_object_sptr > processed_detections;

  // Process all regions
  for( unsigned i = 0; i < regions_to_process.size(); i++ )
  {
    // Get mapping of original to scaled detections for this region
    std::vector< kv::detected_object_sptr > original_dets;
    std::vector< kv::detected_object_sptr > scaled_dets;

    scale_detections_to_region_with_mapping( detections,
      region_properties[i], original_dets, scaled_dets );

    // Skip empty regions if there are no detections
    if( original_dets.empty() )
    {
      continue;
    }

    // Separate detections by processing category
    kv::detected_object_set_sptr detections_to_refine =
      std::make_shared< kv::detected_object_set >();
    kv::detected_object_set_sptr detections_to_pass_through =
      std::make_shared< kv::detected_object_set >();
    std::vector< kv::detected_object_sptr > original_to_refine;

    for( size_t j = 0; j < original_dets.size(); j++ )
    {
      auto original_det = original_dets[j];
      auto scaled_det = scaled_dets[j];

      // Check if already processed (if option enabled)
      if( d->m_overlapping_proc_once &&
          processed_detections.find( original_det ) != processed_detections.end() )
      {
        // Skip - already processed in a previous region
        continue;
      }

      // Check if detection touches boundary (if option enabled)
      bool touches_boundary = false;
      if( d->m_process_boundary_dets )
      {
        kv::bounding_box_d bbox = scaled_det->bounding_box();
        touches_boundary =
          ( bbox.min_x() <= 0.0 ) ||
          ( bbox.min_y() <= 0.0 ) ||
          ( bbox.max_x() >= regions_to_process[i].cols - 1 ) ||
          ( bbox.max_y() >= regions_to_process[i].rows - 1 );
      }

      if( touches_boundary )
      {
        // Pass through unmodified
        detections_to_pass_through->add( scaled_det );
        processed_detections.insert( original_det );
      }
      else
      {
        // Queue for refinement
        detections_to_refine->add( scaled_det );
        original_to_refine.push_back( original_det );
      }
    }

    // Scale and add pass-through detections to output
    if( !detections_to_pass_through->empty() )
    {
      refined_detections->add( rescale_detections( detections_to_pass_through,
        region_properties[i], d->m_settings.chip_edge_max_prob ) );
    }

    // Refine the remaining detections
    if( !detections_to_refine->empty() )
    {
      // Convert region to image container
      kv::image_container_sptr region_image(
        new ocv::image_container( regions_to_process[i],
          ocv::image_container::RGB_COLOR ) );

      // Refine detections in this region
      kv::detected_object_set_sptr region_refined =
        d->m_refiner->refine( region_image, detections_to_refine );

      // Scale refined detections back to original image space
      if( region_refined && !region_refined->empty() )
      {
        refined_detections->add( rescale_detections( region_refined,
          region_properties[i], d->m_settings.chip_edge_max_prob ) );
      }

      // Mark these detections as processed
      if( d->m_overlapping_proc_once )
      {
        for( auto original_det : original_to_refine )
        {
          processed_detections.insert( original_det );
        }
      }
    }
  }

  const int min_dim = d->m_settings.min_detection_dim;

  refined_detections->filter([&min_dim](kv::detected_object_sptr dos)
  {
    return !dos || dos->bounding_box().width() < min_dim
                || dos->bounding_box().height() < min_dim;
  });

  return refined_detections;
} // windowed_refiner::refine


} // end namespace viame
