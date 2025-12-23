/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Stack frames with some gap into one output image temporally.
 */

#include "stack_frames_process.h"

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
class stack_frames_process::priv
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

stack_frames_process
::stack_frames_process( kwiver::vital::config_block_sptr const& config )
  : process( config )
  , d( new stack_frames_process::priv() )
{
  make_ports();
  make_config();
}


stack_frames_process
::~stack_frames_process()
{
}


// -----------------------------------------------------------------------------
void
stack_frames_process
::_configure()
{
  d->m_target_frame_gap =
    config_value_using_trait( target_frame_gap );
  d->m_target_time_gap =
    config_value_using_trait( target_time_gap );
}


// -----------------------------------------------------------------------------
void
stack_frames_process
::_step()
{
}


// -----------------------------------------------------------------------------
void
stack_frames_process
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
stack_frames_process
::make_config()
{
  declare_config_using_trait( target_time_gap );
  declare_config_using_trait( target_frame_gap );
}


// =============================================================================
stack_frames_process::priv
::priv()
{
}


stack_frames_process::priv
::~priv()
{
}


} // end namespace core

} // end namespace viame
