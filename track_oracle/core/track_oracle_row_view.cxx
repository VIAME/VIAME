// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_oracle_row_view.h"

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

using std::cerr;
using std::ostream;
using std::vector;

namespace kwiver {
namespace track_oracle {

track_oracle_row_view
::track_oracle_row_view()
{
}

track_oracle_row_view
::track_oracle_row_view( const track_oracle_row_view& other )
  : track_field_host()
{
  for (size_t i=0; i<other.field_list.size(); ++i)
  {
    this->field_list.push_back( other.field_list[i]->clone() );
    this->this_owns_ptr.push_back( true );
  }
}

track_oracle_row_view&
track_oracle_row_view
::operator=( const track_oracle_row_view& other )
{
  if (this != &other)
  {
    this->reset();
    for (size_t i=0; i<other.field_list.size(); ++i)
    {
      this->field_list.push_back( other.field_list[i]->clone() );
      this->this_owns_ptr.push_back( true );
    }
  }
  return *this;
}

track_oracle_row_view
::~track_oracle_row_view()
{
  this->reset();
}

void
track_oracle_row_view
::reset()
{
  for (unsigned i=0; i<this->field_list.size(); ++i)
  {
    if ( this->this_owns_ptr[i] )
    {
      delete this->field_list[i];
    }
  }
  this->field_list.clear();
}

const track_oracle_row_view&
track_oracle_row_view
::operator()( const track_handle_type& h ) const
{
  this->set_cursor( h.row );
  return *this;
}

void
track_oracle_row_view
::remove_row( const track_handle_type& t )
{
  for (unsigned i=0; i<this->field_list.size(); ++i) {
    this->field_list[i]->remove_at_row( t.row );
  }
}

vector< field_handle_type >
track_oracle_row_view
::list_missing_elements( const oracle_entry_handle_type& row ) const
{
  vector< field_handle_type > missing_fields;
  for (unsigned i=0; i<this->field_list.size(); ++i)
  {
    field_handle_type f = this->field_list[i]->get_field_handle();
    if (! track_oracle_core::field_has_row(row, f ))
    {
      missing_fields.push_back( f );
    }
  }
  return missing_fields;
}

bool
track_oracle_row_view
::is_complete() const
{
  for (unsigned i=0; i<this->field_list.size(); ++i)
  {
    if ( ! this->field_list[i]->exists() )
    {
      LOG_INFO( main_logger, "Row " << this->get_cursor() << " is missing ");
      LOG_INFO( main_logger, this->field_list[i] );
      LOG_INFO( main_logger, "\ncomplete dump:");
      LOG_INFO( main_logger, *this << "");
      return false;
    }
  }
  return true;
}

void
track_oracle_row_view
::copy_values( const oracle_entry_handle_type& src,
               const oracle_entry_handle_type& dst ) const
{
  for ( size_t i=0; i<this->field_list.size(); ++i )
  {
    this->field_list[i]->copy_value( src, dst );
  }
}

ostream&
operator<<( ostream& os, const track_oracle_row_view& r )
{
  size_t n = r.field_list.size();
  os << "h=" << r.get_cursor() << "; " << n;
  if (n == 1)
  {
    os << " field:";
  }
  else
  {
    os << " fields:";
  }
  os << "\n";
  for (unsigned i=0; i<r.field_list.size(); ++i)
  {
    os << "..";
    os << r.field_list[i];
    os << "\n";
  }
  return os;
}

bool
track_oracle_row_view
::add_field( track_field_base& field, bool this_owns )
{
  if (this->contains_element( field.get_field_handle() )) return false;
  field.set_host( this );
  this->field_list.push_back( &field );
  this->this_owns_ptr.push_back( this_owns );
  return true;
}

bool
track_oracle_row_view
::contains_element( const element_descriptor& e ) const
{
  return this->contains_element( track_oracle_core::lookup_by_name( e.name ));
}

bool
track_oracle_row_view
::contains_element( field_handle_type f ) const
{
  if ( f == INVALID_FIELD_HANDLE ) return false;

  for (size_t i=0; i<this->field_list.size(); ++i)
  {
    if ( this->field_list[i]->get_field_handle() == f) return true;
  }
  return false;
}

vector< field_handle_type >
track_oracle_row_view
::list_elements() const
{
  vector< field_handle_type > ret;
  for (size_t i=0; i<this->field_list.size(); ++i)
  {
    ret.push_back( this->field_list[i]->get_field_handle() );
  }
  return ret;
}

track_field_base*
track_oracle_row_view
::clone_field_from_element( const element_descriptor& e ) const
{
  field_handle_type f = track_oracle_core::lookup_by_name( e.name );
  if ( f == INVALID_FIELD_HANDLE ) return 0;

  for (size_t i=0; i<this->field_list.size(); ++i)
  {
    if (this->field_list[i]->get_field_handle() == f)
    {
      return this->field_list[i]->clone();
    }
  }
  return 0;
}

} // ...track_oracle
} // ...kwiver
