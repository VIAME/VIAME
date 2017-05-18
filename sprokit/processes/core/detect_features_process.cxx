/*ckwg +29
 * Copyright 2015-2016 by Kitware, Inc.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS [yas] elisp error!AS IS''
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

#include "detect_features_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/detect_features.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

  create_config_trait( feature_detector, std::string, "", "Algorithm configuration subblock." )

//----------------------------------------------------------------
// Private implementation class
class detect_features_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  algo::detect_features_sptr m_detector;

}; // end priv class

// ================================================================

detect_features_process
::detect_features_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new detect_features_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach

  make_ports();
  make_config();
}


detect_features_process
::~detect_features_process()
{
}


// ----------------------------------------------------------------
void detect_features_process
::_configure()
{
  start_configure_processing();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::detect_features::check_nested_algo_configuration( "feature_detector", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

  // Instantiate the configured algorithm
  algo::detect_features::set_nested_algo_configuration( "feature_detector", algo_config, d->m_detector );
  if ( ! d->m_detector )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create feature_detector" );
  }

  stop_configure_processing();
}


// ----------------------------------------------------------------
void
detect_features_process
::_step()
{
  // timestamp
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );

  // image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );

  start_step_processing();

  LOG_DEBUG( logger(), "Processing frame " << frame_time );

  // detect features on the current frame
  kwiver::vital::feature_set_sptr curr_feat = d->m_detector->detect( img );

  stop_step_processing();

  // return by value
  push_to_port_using_trait( feature_set, curr_feat );
}


// ----------------------------------------------------------------
void detect_features_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );

  // -- output --
  declare_output_port_using_trait( feature_set, optional );
}


// ----------------------------------------------------------------
void detect_features_process
::make_config()
{
  declare_config_using_trait( feature_detector );
}


// ================================================================
detect_features_process::priv
::priv()
{
}


detect_features_process::priv
::~priv()
{
}

} // end namespace
