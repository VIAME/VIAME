/*ckwg +29
 * Copyright 2018 by Kitware, Inc.
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

#include "burnout_detector.h"

#include <string>
#include <sstream>
#include <exception>

#include <arrows/vxl/image_container.h>

#include <vital/exceptions.h>

#include <pipelines/remove_burnin_pipeline.h>

#include <object_detectors/conn_comp_super_process.h>


namespace viame {


// ==================================================================================
class burnout_detector::priv
{
public:
  priv()
    : m_clf_config( "burnout_classification.conf" )
    , m_det_config( "burnout_detector.conf" )
    , m_clf_process( "classifier" )
    , m_det_process( "detector" )
  {}

  ~priv()
  {}

  // Items from the config
  std::string m_clf_config;
  std::string m_det_config;

  vidtk::remove_burnin_pipeline< vxl_byte > m_clf_process;
  vidtk::conn_comp_super_process< vxl_byte > m_det_process;

  vital::logger_handle_t m_logger;
};


// ==================================================================================
burnout_detector
::burnout_detector()
  : d( new priv() )
{
}


burnout_detector
::~burnout_detector()
{}


// ----------------------------------------------------------------------------------
vital::config_block_sptr
burnout_detector
::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "clf_config", d->m_clf_config,  "Name of config file." );
  config->set_value( "det_config", d->m_det_config,  "Name of config file." );

  return config;
}


// ----------------------------------------------------------------------------------
void
burnout_detector
::set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config( config_in );

  d->m_clf_config = config->get_value< std::string >( "clf_config" );
  d->m_det_config = config->get_value< std::string >( "det_config" );

  vidtk::config_block vidtk_clf_config = d->m_clf_process.params();
  vidtk_clf_config.parse( d->m_clf_config );

  if( !d->m_clf_process.set_params( vidtk_clf_config ) )
  {
    std::string reason = "Failed to set pipeline parameters";
    VITAL_THROW( vital::algorithm_configuration_exception, type_name(), impl_name(), reason );
  }

  if( !d->m_clf_process.initialize() )
  {
    std::string reason = "Failed to initialize pipeline";
    VITAL_THROW( vital::algorithm_configuration_exception, type_name(), impl_name(), reason );
  }

  vidtk::config_block vidtk_det_config = d->m_det_process.params();
  vidtk_det_config.parse( d->m_det_config );

  if( !d->m_det_process.set_params( vidtk_det_config ) )
  {
    std::string reason = "Failed to set pipeline parameters";
    VITAL_THROW( vital::algorithm_configuration_exception, type_name(), impl_name(), reason );
  }

  if( !d->m_det_process.initialize() )
  {
    std::string reason = "Failed to initialize pipeline";
    VITAL_THROW( vital::algorithm_configuration_exception, type_name(), impl_name(), reason );
  }
}


// ----------------------------------------------------------------------------------
bool
burnout_detector
::check_configuration( vital::config_block_sptr config ) const
{
  std::string config_fn = config->get_value< std::string >( "clf_config" );

  if( config_fn.empty() )
  {
    return false;
  }

  config_fn = config->get_value< std::string >( "det_config" );

  if( config_fn.empty() )
  {
    return false;
  }

  return true;
}


// ----------------------------------------------------------------------------------
kwiver::vital::detected_object_set_sptr
burnout_detector
::detect( kwiver::vital::image_container_sptr image_data ) const
{
  // Convert inputs to burnout style inputs
  vil_image_view< vxl_byte > input_image;
  auto output = std::make_shared< kwiver::vital::detected_object_set >();

  if( image_data )
  {
    input_image = vxl::image_container::vital_to_vxl( image_data->get_image() );
  }
  else
  {
    return output;
  }

  // Process imagery
  d->m_clf_process.set_image( input_image );

  if( !d->m_clf_process.step() )
  {
    throw std::runtime_error( "Unable to step burnout filter process" );
  }

  d->m_det_process.set_source_image( input_image );
  d->m_det_process.set_fg_image( d->m_clf_process.detected_mask() );
  d->m_det_process.set_world_units_per_pixel( 0.05 );

  if( !d->m_det_process.step() )
  {
    throw std::runtime_error( "Unable to step burnout detect process" );
  }

  // Read outputs and convert
  auto vidtk_detections = d->m_det_process.output_objects();

  for( auto det : vidtk_detections )
  {
    if( !det )
    {
      continue;
    }

    // Create kwiver style bounding box
    kwiver::vital::bounding_box_d bbox(
      kwiver::vital::bounding_box_d::vector_type(
        det->get_bbox().min_x(),
        det->get_bbox().min_y() ),
      det->get_bbox().width(),
      det->get_bbox().height() );

    double conf = 0.5;

    if( det->get_confidence() > 0.0 && det->get_confidence() < 1.0 )
    {
      conf = det->get_confidence();
    }

    // Create possible object types.
    auto dot = std::make_shared< kwiver::vital::detected_object_type >(
      "detection", conf );

    auto dop = std::make_shared< kwiver::vital::detected_object >(
      bbox, conf, dot );

    vil_image_view< bool > mask;
    vidtk::image_object::image_point_type origin;

    if( det->get_object_mask( mask, origin ) )
    {
      dop->set_mask(
        kwiver::vital::image_container_sptr(
          new arrows::vxl::image_container( mask ) ) );
    }

    // Create detection
    output->add( dop );
  }

  return output;
}


} // end namespace viame
