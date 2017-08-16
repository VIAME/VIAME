/*ckwg +29
 * Copyright 2017 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS ``AS IS''
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
 * \brief Implementation of handle_descriptor_request_core
 */

#include "handle_descriptor_request_core.h"

#include <algorithm>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include <iterator>

#include <vital/vital_foreach.h>
#include <vital/types/descriptor_request.h>
#include <vital/algo/algorithm.h>
#include <vital/exceptions/algorithm.h>
#include <vital/exceptions/image.h>

using namespace kwiver::vital;

namespace kwiver {
namespace arrows {
namespace core {

/// Default Constructor
handle_descriptor_request_core
::handle_descriptor_request_core()
{
}


/// Get this alg's \link vital::config_block configuration block \endlink
vital::config_block_sptr
handle_descriptor_request_core
::get_configuration() const
{
  // get base config from base class
  vital::config_block_sptr config = algorithm::get_configuration();

  // Sub-algorithm implementation name + sub_config block
  // - Feature Detector algorithm
  algo::image_io::get_nested_algo_configuration(
    "image_reader", config, reader_ );

  // - Descriptor Extractor algorithm
  algo::compute_track_descriptors::get_nested_algo_configuration(
    "descriptor_extractor", config, extractor_ );

  return config;
}


/// Set this algo's properties via a config block
void
handle_descriptor_request_core
::set_configuration(vital::config_block_sptr in_config)
{
  // Starting with our generated config_block to ensure that assumed values are present
  // An alternative is to check for key presence before performing a get_value() call.
  vital::config_block_sptr config = this->get_configuration();
  config->merge_config(in_config);

  // Setting nested algorithm instances via setter methods instead of directly
  // assigning to instance property.
  algo::image_io_sptr df;
  algo::image_io::set_nested_algo_configuration( "image_reader", config, df );
  reader_ = df;

  algo::compute_track_descriptors_sptr ed;
  algo::compute_track_descriptors::set_nested_algo_configuration( "descriptor_extractor", config, ed );
  extractor_ = ed;
}


bool
handle_descriptor_request_core
::check_configuration(vital::config_block_sptr config) const
{
  return (
    algo::image_io::check_nested_algo_configuration( "image_reader", config )
    &&
    algo::compute_track_descriptors::check_nested_algo_configuration( "descriptor_extractor", config )
  );
}


/// Extend a previous set of tracks using the current frame
bool
handle_descriptor_request_core
::handle(
  kwiver::vital::descriptor_request_sptr request,
  kwiver::vital::track_descriptor_set_sptr& descs,
  std::vector< kwiver::vital::image_container_sptr >& imgs )
{
  // Verify that all dependent algorithms have been initialized
  if( !reader_ || !extractor_ )
  {
    // Something did not initialize
    throw vital::algorithm_configuration_exception( this->type_name(), this->impl_name(),
        "not all sub-algorithms have been initialized" );
  }

  // load images or video if required by query plan
  std::string data_path = request->data_location();
  kwiver::vital::image_container_sptr image = reader_->load( data_path );

  // extract descriptors on the current frame
  kwiver::vital::object_track_set_sptr tracks;

  descs = extractor_->compute( image, tracks );

  imgs.clear();
  imgs.push_back( image );
  return true;
}

} // end namespace core
} // end namespace arrows
} // end namespace kwiver
