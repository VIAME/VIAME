/*ckwg +29
 * Copyright 2017, 2020 by Kitware, Inc.
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


#include "draw_detected_object_set_process.h"

#include <vital/algo/draw_detected_object_set.h>

#include <sprokit/processes/kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace kwiver {

// (config-key, value-type, default-value, description )
create_algorithm_name_config_trait( draw_algo );

// ----------------------------------------------------------------
/**
 * \class draw_detected_object_set_process
 *
 * \brief Draw detected objects on reference imagery.
 *
 * \process This process draws the bounding box outlines of the
 * detections in the supplied set on the supplied reference
 * imagery. The actual rendering is done by the selected \b
 * draw_detected_object_set algorithm implementation.
 *
 * \iports
 *
 * \iport{detected_object_set} Set of detected objects to render on reference image.
 *
 * \iport{image} Reference image for rendering.
 *
 * \oports
 *
 * \oport{image} A copy of the input image with bounding boxes rendered.
 *
 * \configs
 *
 * \config{draw_algo} Name of the configuration subblock that selects
 * and configures the drawing algorithm.
 */

//----------------------------------------------------------------
// Private implementation class
  class  draw_detected_object_set_process::priv
{
public:
  priv();
  ~priv();

  vital::algo::draw_detected_object_set_sptr m_algo;

}; // end priv class


// ================================================================

draw_detected_object_set_process
::draw_detected_object_set_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new draw_detected_object_set_process::priv )
{
  make_ports();
  make_config();
}


draw_detected_object_set_process
::~draw_detected_object_set_process()
{
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::_configure()
{
  scoped_configure_instrumentation();

  auto algo_config = get_config();

  // Check config so it will give run-time diagnostic of config problems
  if ( ! vital::algo::draw_detected_object_set::check_nested_algo_configuration_using_trait(
         draw_algo, algo_config ) )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Configuration check failed." );
  }

  vital::algo::draw_detected_object_set::set_nested_algo_configuration_using_trait(
    draw_algo,
    algo_config,
    d->m_algo );
  if ( ! d->m_algo )
  {
    VITAL_THROW( sprokit::invalid_configuration_exception, name(), "Unable to create algorithm." );
  }
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::_step()
{
  auto input_image = grab_from_port_using_trait( image );
  auto obj_set = grab_from_port_using_trait( detected_object_set );

  kwiver::vital::image_container_sptr out_image;

  {
    scoped_step_instrumentation();

    out_image = d->m_algo->draw( obj_set, input_image );
  }

  push_to_port_using_trait( image, out_image );
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t output;
  output.insert( flag_output_shared );

  sprokit::process::port_flags_t required;
  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( detected_object_set, required );

  // -- output --
  declare_output_port_using_trait( image, output );
}


// ----------------------------------------------------------------
void draw_detected_object_set_process
::make_config()
{
  declare_config_using_trait( draw_algo );
}

// ================================================================
draw_detected_object_set_process::priv
::priv()
{
}


draw_detected_object_set_process::priv
::~priv()
{
}

} // end namespace
