// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_oracle_frame_view.h"
#include <stdexcept>

namespace kwiver {
namespace track_oracle {

track_oracle_frame_view
::track_oracle_frame_view( track_oracle_row_view& parent )
  : track_oracle_row_view(), parent_track_view( parent )
{
}

frame_handle_type
track_oracle_frame_view
::create()
{
  // move to a new row
  this->set_cursor( track_oracle_core::get_next_handle() );

  // persistently record our parent track
  // see design.txt for the eventual evolution of this
  track_field< oracle_entry_handle_type > pt( "__parent_track" );
  pt( this->get_cursor() ) = this->parent_track_view.get_cursor();

  // add ourselves to our parent track
  track_field<frame_handle_list_type> frame_list( "__frame_list" );
  frame_list( this->parent_track_view.get_cursor() ).push_back( frame_handle_type( this->get_cursor() ));

  return frame_handle_type( this->get_cursor() );
}

const track_oracle_frame_view&
track_oracle_frame_view
::operator[]( frame_handle_type f ) const
{
  this->set_cursor( f.row );
  return *this;
}

bool
track_oracle_frame_view
::unlink( const oracle_entry_handle_type& row )
{
  // only delete if we're in our parent track
  track_handle_type p( this->parent_track_view.get_cursor() );
  frame_handle_list_type frames = track_oracle_core::get_frames( p );
  size_t n = frames.size();
  size_t found_index = 0;
  bool found = false;
  for (unsigned i=0; ( ! found ) && (i<n); ++i)
  {
    if (frames[i].row == row)
    {
      found_index = i;
      found = true;
    }
  }
  // fail if we're not actually a member of our parent track
  if ( ! found )
  {
    return false;
  }

  // remove from our parent's list of frames
  if (n == 1)
  {
    frames.clear();
  }
  else
  {
    for (size_t i=found_index; i<n-1; ++i)
    {
      frames[i] = frames[i+1];
    }
    frames.resize( n-1 );
  }
  track_oracle_core::set_frames( p, frames );

  return true;
}

} // ...track_oracle
} // ...kwiver
