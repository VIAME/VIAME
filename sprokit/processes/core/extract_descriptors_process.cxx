/*ckwg +29
 * Copyright 2015-2017, 2020 by Kitware, Inc.
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

#include "extract_descriptors_process.h"

#include <vital/vital_types.h>
#include <vital/types/timestamp.h>
#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_set.h>

#include <vital/algo/extract_descriptors.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( descriptor_extractor );

//----------------------------------------------------------------
// Private implementation class
class extract_descriptors_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values

  // There are many config items for the tracking and stabilization that go directly to
  // the algo.

  algo::extract_descriptors_sptr m_extractor;

}; // end priv class

// ================================================================

extract_descriptors_process
::extract_descriptors_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new extract_descriptors_process::priv )
{
  make_ports();
  make_config();
}


extract_descriptors_process
::~extract_descriptors_process()
{
}


// ----------------------------------------------------------------
void extract_descriptors_process
::_configure()
{
  scoped_configure_instrumentation();

  // Get our process config
  kwiver::vital::config_block_sptr algo_config = get_config();

  // Instantiate the configured algorithm
  algo::extract_descriptors::set_nested_algo_configuration_using_trait(
    descriptor_extractor,
    algo_config,
    d->m_extractor );
  if ( ! d->m_extractor )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create descriptor_extractor" );
  }

  algo::extract_descriptors::get_nested_algo_configuration_using_trait(
    descriptor_extractor,
    algo_config,
    d->m_extractor );

  // Check config so it will give run-time diagnostic if any config problems are found
  if ( ! algo::extract_descriptors::check_nested_algo_configuration_using_trait(
         descriptor_extractor,
         algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }
}


// ----------------------------------------------------------------
void
extract_descriptors_process
::_step()
{
  kwiver::vital::timestamp frame_time = grab_from_port_using_trait( timestamp );
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  kwiver::vital::feature_set_sptr features =  grab_from_port_using_trait( feature_set );

  kwiver::vital::descriptor_set_sptr curr_desc;

  {
    scoped_step_instrumentation();

    // LOG_DEBUG - this is a good thing to have in all processes that handle frames.
    LOG_DEBUG( logger(), "Processing frame " << frame_time );

    // extract stuff on the current frame
    curr_desc = d->m_extractor->extract( img, features );
  }

  // return by value
  push_to_port_using_trait( descriptor_set, curr_desc );
}


// ----------------------------------------------------------------
void extract_descriptors_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( timestamp, required );
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( feature_set, required );

  // -- output --
  declare_output_port_using_trait( descriptor_set, optional );
}


// ----------------------------------------------------------------
void extract_descriptors_process
::make_config()
{
  declare_config_using_trait( descriptor_extractor );
}


// ================================================================
extract_descriptors_process::priv
::priv()
{
}


extract_descriptors_process::priv
::~priv()
{
}

} // end namespace
