/*ckwg +29
 * Copyright 2022 by Kitware, Inc.
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
 * \brief Stack frames with some gap into one output image temporally.
 */

#include "frame_stacker_process.h"

#include <vital/vital_types.h>

#include <vital/types/timestamp_config.h>
#include <vital/types/image_container.h>

#include <sstream>
#include <iostream>
#include <list>
#include <limits>
#include <cmath>


namespace viame
{

namespace core
{

create_config_trait( target_frame_gap, double, "1.0",
  "Target time gap between frames if timestamp is valid" );
create_config_trait( target_time_gap, unsigned, "10",
  "Target frame count gap between frames if timestamp not valid" );

//------------------------------------------------------------------------------
// Private implementation class
class frame_stacker_process::priv
{
public:
  priv();
  ~priv();

  // Configuration values
  unsigned m_target_frame_gap;
  double m_target_time_gap;

  // Internal buffer
  std::list< buffered_frame > m_frames;
};

// =============================================================================

frame_stacker_process
::frame_stacker_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
  , d( new frame_stacker_process::priv() )
{
  make_ports();
  make_config();
}


frame_stacker_process
::~frame_stacker_process()
{
}


// -----------------------------------------------------------------------------
void
frame_stacker_process
::_configure()
{
  d->m_target_frame_gap =
    config_value_using_trait( target_frame_gap );
  d->m_target_time_gap =
    config_value_using_trait( target_time_gap );
}


// -----------------------------------------------------------------------------
void
frame_stacker_process
::_step()
{
}


// -----------------------------------------------------------------------------
void
frame_stacker_process
::make_ports()
{
  // Set up for required ports
  sprokit::process::port_flags_t required;
  sprokit::process::port_flags_t optional;

  required.insert( flag_required );

  // -- input --
  declare_input_port_using_trait( image, required );
  declare_input_port_using_trait( timestamp, optional );

  // -- output --
  declare_output_port_using_trait( image, required );
}


// -----------------------------------------------------------------------------
void
frame_stacker_process
::make_config()
{
  declare_config_using_trait( target_time_gap );
  declare_config_using_trait( target_frame_gap );
}


// =============================================================================
frame_stacker_process::priv
::priv()
{
}


frame_stacker_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
