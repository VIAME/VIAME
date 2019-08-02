/*ckwg +29
 * Copyright 2019 by Kitware, Inc.
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

#include "transform_detected_object_set.h"

#include <vital/io/camera_io.h>
#include <vital/config/config_difference.h>
#include <vital/util/string.h>
#include <vital/types/bounding_box.h>

namespace kwiver {
namespace arrows {
namespace core {

// ------------------------------------------------------------------
transform_detected_object_set::transform_detected_object_set()
  : src_camera_krtd_file_name( "" )
  , dest_camera_krtd_file_name( "" )
{
}


// ------------------------------------------------------------------
vital::config_block_sptr
transform_detected_object_set::get_configuration() const
{
  // Get base config from base class
  vital::config_block_sptr config = vital::algorithm::get_configuration();

  config->set_value( "src_camera_krtd_file_name", src_camera_krtd_file_name,
                     "Source camera KRTD file name path" );

  config->set_value( "dest_camera_krtd_file_name", dest_camera_krtd_file_name,
                     "Destination camera KRTD file name path" );

  return config;
}


// ------------------------------------------------------------------
void
transform_detected_object_set::
set_configuration( vital::config_block_sptr config_in )
{
  vital::config_block_sptr config = this->get_configuration();

  config->merge_config( config_in );
  this->src_camera_krtd_file_name = config->get_value< std::string > ( "src_camera_krtd_file_name" );
  this->dest_camera_krtd_file_name = config->get_value< std::string > ( "dest_camera_krtd_file_name" );

  this->src_camera = kwiver::vital::read_krtd_file( this->src_camera_krtd_file_name );
  this->dest_camera = kwiver::vital::read_krtd_file( this->dest_camera_krtd_file_name );
}


// ------------------------------------------------------------------
bool
transform_detected_object_set::
check_configuration( vital::config_block_sptr config ) const
{
  kwiver::vital::config_difference cd( this->get_configuration(), config );
  const auto key_list = cd.extra_keys();

  if ( ! key_list.empty() )
  {
    LOG_WARN( logger(), "Additional parameters found in config block that are not required or desired: "
              << kwiver::vital::join( key_list, ", " ) );
  }

  return true;
}

// ------------------------------------------------------------------
vital::bounding_box<double>
transform_detected_object_set::
transform_bounding_box( vital::bounding_box<double>& bbox ) const
{
  // TODO
  return bbox;
}

// ------------------------------------------------------------------
vital::detected_object_set_sptr
transform_detected_object_set::
filter( const vital::detected_object_set_sptr input_set ) const
{
  auto ret_set = std::make_shared<vital::detected_object_set>();

  // loop over all detections
  auto ie = input_set->cend();
  for ( auto det = input_set->cbegin(); det != ie; ++det )
  {
    auto out_det = (*det)->clone();
    auto out_box = out_det->bounding_box();
    this->transform_bounding_box(out_box);
    out_det->set_bounding_box( out_box );
    ret_set->add( out_det );
  } // end foreach detection

  return ret_set;
} // transform_detected_object_set::filter

} } }     // end namespace
