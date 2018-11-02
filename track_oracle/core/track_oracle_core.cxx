/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_oracle_core.h"
#include <track_oracle/core/element_descriptor.h>
#include <track_oracle/core/track_oracle_core_impl.h>
#include <mutex>


using std::ostream;
using std::string;
using std::vector;


namespace // anon
{
std::mutex instance_lock;
};

namespace kwiver {
namespace track_oracle {

track_oracle_core_impl&
track_oracle_core
::get_instance()
{
  if ( ! track_oracle_core::impl )
  {
    std::lock_guard< std::mutex > lock( instance_lock );
    if ( ! track_oracle_core::impl )
    {
      track_oracle_core::impl = new track_oracle_core_impl();
    }
  }
  return *track_oracle_core::impl;
}

vector< field_handle_type >
track_oracle_core
::get_all_field_handles()
{
  return track_oracle_core::get_instance().get_all_field_handles();
}

field_handle_type
track_oracle_core
::lookup_by_name( const string& name )
{
  return track_oracle_core::get_instance().lookup_by_name( name );
}

element_descriptor
track_oracle_core
::get_element_descriptor( field_handle_type f )
{
  return track_oracle_core::get_instance().get_element_descriptor( f );
}

const element_store_base*
track_oracle_core
::get_element_store_base( field_handle_type f )
{
  return track_oracle_core::get_instance().get_element_store_base( f );
}

element_store_base*
track_oracle_core
::get_mutable_element_store_base( field_handle_type f )
{
  return track_oracle_core::get_instance().get_mutable_element_store_base( f );
}

bool
track_oracle_core
::field_has_row( oracle_entry_handle_type row, field_handle_type field )
{
  return track_oracle_core::get_instance().field_has_row( row, field );
}

vector< field_handle_type >
track_oracle_core
::fields_at_row( oracle_entry_handle_type row )
{
  return track_oracle_core::get_instance().fields_at_row( row );
}

vector< vector< field_handle_type > >
track_oracle_core
::fields_at_rows( const vector< oracle_entry_handle_type > rows )
{
  return track_oracle_core::get_instance().fields_at_rows( rows );
}

oracle_entry_handle_type
track_oracle_core
::get_next_handle()
{
  return track_oracle_core::get_instance().get_next_handle();
}

handle_list_type
track_oracle_core
::get_domain( domain_handle_type domain )
{
  return track_oracle_core::get_instance().get_domain( domain );
}

domain_handle_type
track_oracle_core
::lookup_domain( const string& domain_name, bool create_if_not_found )
{
  return track_oracle_core::get_instance().lookup_domain( domain_name, create_if_not_found );
}

domain_handle_type
track_oracle_core
::create_domain()
{
  return track_oracle_core::get_instance().create_domain();
}

domain_handle_type
track_oracle_core
::create_domain( const handle_list_type& handles )
{
  return track_oracle_core::get_instance().create_domain( handles );
}

bool
track_oracle_core
::is_domain_defined( const domain_handle_type& domain )
{
  return track_oracle_core::get_instance().is_domain_defined( domain );
}

bool
track_oracle_core
::release_domain( const domain_handle_type& domain )
{
  return track_oracle_core::get_instance().release_domain( domain );
}

bool
track_oracle_core
::add_to_domain( const handle_list_type& handles, const domain_handle_type& domain )
{
  return track_oracle_core::get_instance().add_to_domain( handles, domain );
}

bool
track_oracle_core
::add_to_domain( const track_handle_type& track, const domain_handle_type& domain )
{
  return track_oracle_core::get_instance().add_to_domain( track, domain );
}

bool
track_oracle_core
::set_domain( const handle_list_type& handles, domain_handle_type domain )
{
  return track_oracle_core::get_instance().set_domain( handles, domain );
}

track_handle_list_type
track_oracle_core
::generic_to_track_handle_list( const handle_list_type& handles )
{
  track_handle_list_type ret;
  for (unsigned i=0; i<handles.size(); ++i)
  {
    ret.push_back( track_handle_type( handles[i] ));
  }
  return ret;
}

frame_handle_list_type
track_oracle_core
::generic_to_frame_handle_list( const handle_list_type& handles )
{
  frame_handle_list_type ret;
  for (unsigned i=0; i<handles.size(); ++i)
  {
    ret.push_back( frame_handle_type( handles[i] ));
  }
  return ret;
}

handle_list_type
track_oracle_core
::track_to_generic_handle_list( const track_handle_list_type& handles )
{
  handle_list_type ret;
  for (unsigned i=0; i<handles.size(); ++i)
  {
    ret.push_back( handles[i].row );
  }
  return ret;
}

handle_list_type
track_oracle_core
::frame_to_generic_handle_list( const frame_handle_list_type& handles )
{
  handle_list_type ret;
  for (unsigned i=0; i<handles.size(); ++i)
  {
    ret.push_back( handles[i].row );
  }
  return ret;
}

frame_handle_list_type
track_oracle_core
::get_frames( const track_handle_type& t )
{
  return track_oracle_core::get_instance().get_frames( t );
}

void
track_oracle_core
::set_frames( const track_handle_type& t, const frame_handle_list_type& f )
{
  track_oracle_core::get_instance().set_frames( t, f );
}

size_t
track_oracle_core
::get_n_frames( const track_handle_type& t )
{
  return track_oracle_core::get_instance().get_n_frames( t );
}

bool
track_oracle_core
::clone_nonsystem_fields( const track_handle_type& src,
                          const track_handle_type& dst )
{
  return track_oracle_core::clone_nonsystem_fields( src.row, dst.row );
}

bool
track_oracle_core
::clone_nonsystem_fields( const frame_handle_type& src,
                          const frame_handle_type& dst )
{
  return track_oracle_core::clone_nonsystem_fields( src.row, dst.row );
}

bool
track_oracle_core
::clone_nonsystem_fields( const oracle_entry_handle_type& src,
                          const oracle_entry_handle_type& dst )
{
  return track_oracle_core::get_instance().clone_nonsystem_fields( src, dst );
}

bool
track_oracle_core
::write_kwiver( ostream& os, const track_handle_list_type& tracks )
{
  return track_oracle_core::get_instance().write_kwiver( os, tracks );
}


bool
track_oracle_core
::write_csv( ostream& os, const track_handle_list_type& tracks, bool csv_v1_semantics )
{
  return track_oracle_core::get_instance().write_csv( os, tracks, csv_v1_semantics );
}

csv_handler_map_type
track_oracle_core
::get_csv_handler_map( const vector< string >& headers )
{
  return track_oracle_core::get_instance().get_csv_handler_map( headers );
}

} // ...track_oracle
} // ...kwiver
