/*ckwg +29
 * Copyright 2017-2025 by Kitware, Inc.
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

/**
 * \file
 * \brief Algorithm to add keypoints to detections from mask
 */

#include "ocv_refiner_add_kps_from_mask.h"
#include "ocv_keypoints_from_mask.h"

#include <vital/types/point.h>

namespace kv = kwiver::vital;

namespace viame
{

// =============================================================================
// Private implementation class
class ocv_refiner_add_kps_from_mask::priv
{
public:
  priv()
    : method( "oriented_bbox" )
  {
  }

  ~priv()
  {
  }

  // Configuration
  std::string method;
};

// =============================================================================
ocv_refiner_add_kps_from_mask
::ocv_refiner_add_kps_from_mask()
  : d( new priv() )
{
}

ocv_refiner_add_kps_from_mask
::~ocv_refiner_add_kps_from_mask()
{
}

// -----------------------------------------------------------------------------
kv::config_block_sptr
ocv_refiner_add_kps_from_mask
::get_configuration() const
{
  kv::config_block_sptr config = kv::algo::refine_detections::get_configuration();

  config->set_value( "method", d->method, keypoint_method_description() );

  return config;
}

// -----------------------------------------------------------------------------
void
ocv_refiner_add_kps_from_mask
::set_configuration( kv::config_block_sptr config )
{
  d->method = config->get_value<std::string>( "method", d->method );
}

// -----------------------------------------------------------------------------
bool
ocv_refiner_add_kps_from_mask
::check_configuration( kv::config_block_sptr config ) const
{
  std::string method = config->get_value<std::string>( "method", "oriented_bbox" );

  if( !is_valid_keypoint_method( method ) )
  {
    LOG_ERROR( logger(), "Invalid method: " << method );
    return false;
  }

  return true;
}

// -----------------------------------------------------------------------------
kv::detected_object_set_sptr
ocv_refiner_add_kps_from_mask
::refine( kv::image_container_sptr image_data,
          kv::detected_object_set_sptr detections ) const
{
  auto output = std::make_shared<kv::detected_object_set>();

  for( auto det : *detections )
  {
    if( det->mask() )
    {
      auto keypoints = compute_keypoints( det, d->method );

      det->add_keypoint( "head", kv::point_2d( keypoints.first.x, keypoints.first.y ) );
      det->add_keypoint( "tail", kv::point_2d( keypoints.second.x, keypoints.second.y ) );
    }

    output->add( det );
  }

  return output;
}

} // end namespace viame
