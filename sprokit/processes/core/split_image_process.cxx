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

#include "split_image_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>

#include <vital/algo/split_image.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

//----------------------------------------------------------------
// Private implementation class
class split_image_process::priv
{
public:
  priv();
  ~priv();

  algo::split_image_sptr         m_image_splitter;
};

// ================================================================

split_image_process
::split_image_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new split_image_process::priv )
{
  attach_logger( kwiver::vital::get_logger( name() ) );

  make_ports();
  make_config();
}


split_image_process
::~split_image_process()
{
}


// ----------------------------------------------------------------
void split_image_process
::_configure()
{
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::split_image::set_nested_algo_configuration(
    "split_image", algo_config, d->m_image_splitter );

  if( !d->m_image_splitter )
  {
    throw sprokit::invalid_configuration_exception(
      name(), "Unable to create \"split_image\"" );
  }
  algo::split_image::get_nested_algo_configuration(
    "split_image", algo_config, d->m_image_splitter );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::split_image::check_nested_algo_configuration(
        "split_image", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Configuration check failed." );
  }

}


// ----------------------------------------------------------------
void
split_image_process
::_step()
{
  // Get image
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  std::vector< kwiver::vital::image_container_sptr > outputs;

  // Get feature tracks
  outputs = d->m_image_splitter->split( img );

  // Return by value
  if( outputs.size() >= 1 )
  {
    push_to_port_using_trait( left_image, outputs[0] );
  }

  if( outputs.size() >= 2 )
  {
    push_to_port_using_trait( right_image, outputs[1] );
  }
}


// ----------------------------------------------------------------
void split_image_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );

  declare_output_port_using_trait( left_image, optional );
  declare_output_port_using_trait( right_image, optional );
}


// ----------------------------------------------------------------
void split_image_process
::make_config()
{

}


// ================================================================
split_image_process::priv
::priv()
{
}


split_image_process::priv
::~priv()
{
}

} // end namespace
