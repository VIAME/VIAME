/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_base_impl.h"

#include <map>

#include <track_oracle/core/track_oracle_core.h>
#include <track_oracle/core/element_descriptor.h>


using std::make_pair;
using std::map;
using std::ostream;
using std::pair;
using std::vector;

namespace kwiver {
namespace track_oracle {

track_base_impl
::track_base_impl()
  : Track(), Frame( Track )
{
}


track_handle_type
track_base_impl
::create()
{
  oracle_entry_handle_type new_row = track_oracle_core::get_next_handle();
  this->Track.set_cursor( new_row );
  return track_handle_type( new_row );
}

frame_handle_type
track_base_impl
::create_frame()
{
  return frame_handle_type( this->Frame.create() );
}

bool
track_base_impl
::remove_frame( const frame_handle_type& f )
{
  if ( ! this->Frame.unlink( f.row ) ) return false;
  this->Frame.remove_row( track_handle_type( f.row )); // hmm, hack
  return true;
}

void
track_base_impl
::remove_me()
{
  // first, remove the frames.
  //
  // since the whole track is going away, don't call the frame's remove method--
  // we don't need to repeatedly update the parent frame list --
  // just nuke the rows.
  //
  // but we do need to remove the __frame_list and __parent_track fields
  // ...can a frame have multiple parents?


  vector<frame_handle_type> frames = track_oracle_core::get_frames( track_handle_type( this->Track.get_cursor() ) );
  for (unsigned i=0; i<frames.size(); ++i)
  {
    this->Frame.remove_row( track_handle_type( frames[i].row ));
  }
  // then nuke the track
  this->Track.remove_row( track_handle_type( this->Track.get_cursor() ));

  // all done
}


void
track_base_impl
::add_frames( const frame_handle_list_type& frames )
{
  frame_handle_list_type these_frames = track_oracle_core::get_frames( track_handle_type( this->Track.get_cursor() ) );
  these_frames.insert( these_frames.end(), frames.begin(), frames.end() );
  track_oracle_core::set_frames( track_handle_type( this->Track.get_cursor() ), these_frames );
}

const track_oracle_frame_view&
track_base_impl
::frame() const
{
  return this->Frame;
}

const track_oracle_row_view&
track_base_impl
::track() const
{
  return this->Track;
}

bool
track_base_impl
::is_complete( const track_handle_type& row ) const
{
  if ( ! this->Track(row).is_complete()) return false;
  const frame_handle_list_type frames = track_oracle_core::get_frames( row );
  for (unsigned i=0; i<frames.size(); ++i)
  {
    if ( ! this->Frame[ frames[i] ].is_complete() ) return false;
  }
  return true;
}

ostream&
operator<<( ostream& os, track_base_impl& track )
{
  os << "Track is: " << *(dynamic_cast<track_oracle_row_view const*>(&track.Track));
  frame_handle_list_type frame_list = track_oracle_core::get_frames( track_handle_type( track.Track.get_cursor() ) );
  os << frame_list.size() << " frames\n";
  for (unsigned i=0; i<frame_list.size(); ++i)
  {
    os << "...Frame " << i << " of " << frame_list.size()
       << " ";
    os << *(dynamic_cast<track_oracle_row_view const*>(&track.Frame[ frame_list[i] ])) << "\n";
  }
  return os;
}

vector< field_handle_type >
track_base_impl
::list_missing_elements( const track_handle_type& _track ) const
{
  // The visiting track (and its frames) may or may not contain all
  // the elements we define.  Return a list of the fields which are
  // defined in our Track and Frame views but not present in the
  // given track handle (or its frames.)

  map< field_handle_type, bool > field_is_missing;
  {
    vector< field_handle_type > missing_fields = this->Track.list_missing_elements( _track.row );
    for (size_t i=0; i<missing_fields.size(); ++i) field_is_missing[ missing_fields[i] ] = true;
  }

  frame_handle_list_type frame_list = track_oracle_core::get_frames( _track );
  for (unsigned i=0; i<frame_list.size(); ++i)
  {
    vector< field_handle_type > missing_fields = this->Frame.list_missing_elements( frame_list[i].row );
    for (size_t j=0; j<missing_fields.size(); ++j) field_is_missing[ missing_fields[j] ] = true;
  }

  vector< field_handle_type > fields;
  for (map< field_handle_type, bool >::const_iterator i = field_is_missing.begin();
       i != field_is_missing.end();
       ++i )
  {
    fields.push_back( i->first );
  }
  return fields;
}

bool
track_base_impl
::schema_contains_element( const element_descriptor& e ) const
{
  return (this->Track.contains_element(e) || this->Frame.contains_element(e) );
}

bool
track_base_impl
::schema_contains_element( field_handle_type f ) const
{
  element_descriptor e = track_oracle_core::get_element_descriptor( f );
  return this->schema_contains_element( e );
}

pair< track_field_base*, track_base_impl::schema_position_type >
track_base_impl
::clone_field_from_element( const element_descriptor& e ) const
{
  track_field_base* ret = this->Track.clone_field_from_element( e );
  if (ret)
  {
    return make_pair( ret, IN_TRACK );
  }
  ret = this->Frame.clone_field_from_element( e );
  if (ret)
  {
    return make_pair( ret, IN_FRAME );
  }
  return make_pair( static_cast<track_field_base*>(0), INVALID );
}

bool
track_base_impl
::add_field_at_position( const pair< track_field_base*, schema_position_type >& f )
{
  if ( ( ! f.first ) || ( f.second == INVALID )) return false;

  return
    ( f.second == IN_TRACK )
    ? this->Track.add_field( *(f.first), /* row view owns = */ true )
    : this->Frame.add_field( *(f.first), /* row view owns = */ true );
}

map< field_handle_type, track_base_impl::schema_position_type >
track_base_impl
::list_schema_elements() const
{
  map< field_handle_type, schema_position_type > ret;
  {
    vector< field_handle_type > f = this->Track.list_elements();
    for (size_t i=0; i<f.size(); ++i) ret[ f[i] ] = IN_TRACK;
  }
  {
    vector< field_handle_type > f = this->Frame.list_elements();
    for (size_t i=0; i<f.size(); ++i) ret[ f[i] ] = IN_FRAME;
  }
  return ret;
}

} // ...track_oracle
} // ...kwiver
