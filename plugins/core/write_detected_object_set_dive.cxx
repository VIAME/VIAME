/* This file is part of VIAME, and is distributed under an OSI-approved *
 * BSD 3-Clause License. See either the root top-level LICENSE file or  *
 * https://github.com/VIAME/VIAME/blob/main/LICENSE.txt for details.    */

/**
 * \file
 * \brief Implementation of the DIVE JSON detection writer
 */

#include "write_detected_object_set_dive.h"
#include "write_object_track_set_dive.h"

#include <vital/types/object_track_set.h>

namespace viame {

namespace kv = kwiver::vital;

// -----------------------------------------------------------------------------
void
write_detected_object_set_dive
::initialize()
{
  attach_logger( "viame.core.write_detected_object_set_dive" );
  m_frame_number = 0;
  m_next_id = 1;
}

// -----------------------------------------------------------------------------
bool
write_detected_object_set_dive
::check_configuration( kv::config_block_sptr ) const
{
  return true;
}

// -----------------------------------------------------------------------------
void
write_detected_object_set_dive
::write_set( const kv::detected_object_set_sptr set,
             std::string const& )
{
  if( set )
  {
    for( auto const& det : *set )
    {
      if( !det )
      {
        continue;
      }
      auto track = kv::track::create();
      track->set_id( m_next_id++ );
      track->append( std::make_shared< kv::object_track_state >(
        m_frame_number, static_cast< kv::time_usec_t >( m_frame_number ), det ) );
      m_tracks.push_back( track );
    }
  }
  ++m_frame_number;
}

// -----------------------------------------------------------------------------
void
write_detected_object_set_dive
::close()
{
  dive_write_options options;
  options.frame_id_adjustment = c_frame_id_adjustment;
  options.top_n_classes = c_top_n_classes;
  options.pretty_print = c_pretty_print;

  write_dive_json( stream(), m_tracks, options );
  m_tracks.clear();
  m_frame_number = 0;
  m_next_id = 1;

  detected_object_set_output::close();
}

} // end namespace viame
