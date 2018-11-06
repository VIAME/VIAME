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

#include <vital/vital_types.h>
#include <vital/types/image_container.h>

#include <vital/algo/merge_images.h>

#include <kwiver_type_traits.h>

#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

create_port_trait( image1, image, "Single frame first image." );
create_port_trait( image2, image, "Single frame second image." );
//----------------------------------------------------------------
// Private implementation class
class merge_images_process::priv
{
public:
  priv();
  ~priv();

  algo::merge_images_sptr         m_images_merger;
};

// ================================================================

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


// ----------------------------------------------------------------
void merge_images_process
::_configure()
{
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::merge_images::set_nested_algo_configuration(
    "merge_images", algo_config, d->m_images_merger );

  if( !d->m_images_merger )
  {
    throw sprokit::invalid_configuration_exception(
        name(), "Unable to create \"merge_images\"" );
  }
  algo::merge_images::get_nested_algo_configuration(
      "merge_images", algo_config, d->m_images_merger );

  // Check config so it will give run-time diagnostic of config problems
  if( !algo::merge_images::check_nested_algo_configuration(
        "merge_images", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(),
      "Configuration check failed." );
  }

}


// ----------------------------------------------------------------
void
merge_images_process
::_step()
{
  // Get image
  kwiver::vital::image_container_sptr image1 =
      grab_from_port_using_trait( image1 );
  kwiver::vital::image_container_sptr image2 =
      grab_from_port_using_trait( image2 );
  kwiver::vital::image_container_sptr output;

  // Get feature tracks
  output = d->m_images_merger->merge( image1, image2 );

  // Return by value
  push_to_port_using_trait( image, output);
}


// ----------------------------------------------------------------
void merge_images_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t optional;
  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image1, required );
  declare_input_port_using_trait( image2, required );

  declare_output_port_using_trait( image, required );
}


// ----------------------------------------------------------------
void merge_images_process
::make_config()
{

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
