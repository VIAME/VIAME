/*ckwg +29
 * Copyright 2015 by Kitware, Inc.
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
 * \brief Draw tracks process implementation.
 */

#include "draw_tracks_process.h"

#include <vital/vital_types.h>
#include <vital/types/image_container.h>
#include <vital/types/feature_track_set.h>

#include <vital/algo/draw_tracks.h>

#include <kwiver_type_traits.h>
#include <sprokit/pipeline/process_exception.h>

namespace algo = kwiver::vital::algo;

namespace kwiver
{

create_port_trait( output_image, image, "Image with tracks" );

// config items
//          None for now

//----------------------------------------------------------------
// Private implementation class
class draw_tracks_process::priv
{
public:
  priv();
  ~priv();

  algo::draw_tracks_sptr         m_draw_tracks;
};


// ================================================================

draw_tracks_process
::draw_tracks_process( kwiver::vital::config_block_sptr const& config )
  : process( config ),
    d( new draw_tracks_process::priv )
{
  // Attach our logger name to process logger
  attach_logger( kwiver::vital::get_logger( name() ) ); // could use a better approach

  make_ports();
  make_config();
}


draw_tracks_process
::~draw_tracks_process()
{
}


// ----------------------------------------------------------------
void
draw_tracks_process
::_configure()
{
  kwiver::vital::config_block_sptr algo_config = get_config();

  algo::draw_tracks::set_nested_algo_configuration( "draw_tracks", algo_config, d->m_draw_tracks );
  if ( ! d->m_draw_tracks )
  {
    throw sprokit::invalid_configuration_exception( name(), "Unable to create draw_tracks" );
  }

  algo::draw_tracks::get_nested_algo_configuration( "draw_tracks", algo_config, d->m_draw_tracks );

  // Check config so it will give run-time diagnostic of config problems
  if ( ! algo::draw_tracks::check_nested_algo_configuration( "draw_tracks", algo_config ) )
  {
    throw sprokit::invalid_configuration_exception( name(), "Configuration check failed." );
  }

}


// ----------------------------------------------------------------
void
draw_tracks_process
::_step()
{
  kwiver::vital::image_container_sptr img = grab_from_port_using_trait( image );
  vital::feature_track_set_sptr tracks = grab_from_port_using_trait( feature_track_set );
  kwiver::vital::image_container_sptr_list image_list;
  image_list.push_back( img );

  kwiver::vital::image_container_sptr annotated_image =
    d->m_draw_tracks->draw( tracks, image_list );

  // ( port, value )
  push_to_port_using_trait( output_image, annotated_image );

}


// ----------------------------------------------------------------
void
draw_tracks_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( feature_track_set, required );

  // -- output --
  declare_output_port_using_trait( output_image, optional );
}


// ----------------------------------------------------------------
void
draw_tracks_process
::make_config()
{
}


// ================================================================
draw_tracks_process::priv
::priv()
{
}


draw_tracks_process::priv
::~priv()
{
}

} // end namespace
