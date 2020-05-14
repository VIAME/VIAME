/*ckwg +29
 * Copyright 2018, 2020 by Kitware, Inc.
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

#include "merge_images_process.h"

#include <kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>
#include <vital/algo/merge_images.h>
#include <vital/types/image_container.h>
#include <vital/util/string.h>
#include <vital/vital_types.h>

namespace algo = kwiver::vital::algo;

namespace kwiver {

create_algorithm_name_config_trait( merge_images );

// ----------------------------------------------------------------------------
// Private implementation class
class merge_images_process::priv
{
public:
  priv();
  ~priv();

  algo::merge_images_sptr m_images_merger;
  std::set< std::string > p_port_list;
};

// ============================================================================
merge_images_process
::merge_images_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new merge_images_process::priv )
{
  make_ports();
  make_config();
}


merge_images_process
::~merge_images_process()
{
}


// ----------------------------------------------------------------------------
void merge_images_process
::_configure()
{
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::merge_images::set_nested_algo_configuration_using_trait(
    merge_images,
    algo_config,
    d->m_images_merger );

  if( !d->m_images_merger )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception,
                 name(), "Unable to create \"merge_images\"" );
  }

  algo::merge_images::get_nested_algo_configuration_using_trait(
    merge_images,
    algo_config,
    d->m_images_merger );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::merge_images::check_nested_algo_configuration_using_trait(
        merge_images, algo_config ) )
  {
    VITAL_THROW(  sprokit::invalid_configuration_exception,
                  name(), "Configuration check failed." );
  }
}


// ----------------------------------------------------------------------------
void
merge_images_process
::_step()
{
  std::vector<kwiver::vital::image_container_sptr> image_list;

  for( const auto port_name : d->p_port_list )
  {
    kwiver::vital::image_container_sptr image_sptr =
      grab_from_port_as< kwiver::vital::image_container_sptr >( port_name );

    image_list.push_back( image_sptr );
  }

  kwiver::vital::image_container_sptr output;

  // Merge images sequentially
  if( image_list.empty() )
  {
    LOG_WARN( logger(), "No input images provided" );
  }
  else
  {
    output = image_list[0];
  }

  for( unsigned i = 1; i < image_list.size(); ++i )
  {
    output = d->m_images_merger->merge( output, image_list[i] );
  }

  // Return by value
  push_to_port_using_trait( image, output);
}


// ----------------------------------------------------------------------------
void merge_images_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  required.insert( flag_required );
  required.insert( flag_output_shared );

  // -- output --
  declare_output_port_using_trait( image, required );
}


// ----------------------------------------------------------------------------
void merge_images_process
::make_config()
{
  declare_config_using_trait( merge_images );
}


// ----------------------------------------------------------------------------
/*
 * This method accepts port names when connections are made and
 * dynamically created the required ports.
 *
 * Note that only two connections are accepted and the ports are
 * typed for images.
 */
void
merge_images_process
::input_port_undefined(port_t const& port_name)
{
  LOG_TRACE( logger(), "Processing input port info: \"" << port_name << "\"" );

  // Just create an input port to read detections from
  if (! kwiver::vital::starts_with( port_name, "_" ) )
  {
    if ( d->p_port_list.size() >= 2)
    {
      LOG_ERROR( logger(), "Attempt to connect more than 2 input ports. "
                 "Connection aborted.");
      return;
    }

    // Check for unique port name
    if ( d->p_port_list.count( port_name ) == 0 )
    {
      port_flags_t required;
      required.insert( flag_required );

      // Create input port
      declare_input_port(
          port_name,                   // port name
          image_port_trait::type_name, // port type
          required,                    // port flags
          "image input" );

      d->p_port_list.insert( port_name );
    }
  }
}

// ================================================================
merge_images_process::priv
::priv()
{
}


merge_images_process::priv
::~priv()
{
}


} // end namespace
