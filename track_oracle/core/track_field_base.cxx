// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_field_base.h"
#include <stdexcept>

#include <vital/logger/logger.h>
static kwiver::vital::logger_handle_t main_logger( kwiver::vital::get_logger( __FILE__ ) );

#include <track_oracle/core/element_descriptor.h>

using std::ostream;
using std::runtime_error;
using std::string;

namespace kwiver {
namespace track_oracle {

track_field_base
::track_field_base( const string& n )
  : name(n), host(0)
{
  // field_handle now set in track_field<T>
}

track_field_base
::track_field_base( const string& n, track_field_host* h )
  : name(n), host(h)
{
}

track_field_base
::~track_field_base()
{
}

string
track_field_base
::get_field_name() const
{
  return name;
}

field_handle_type
track_field_base
::get_field_handle() const
{
  return field_handle;
}

//
// can't make exists() pure virtual because typeless instances are
// used as helper classes for e.g. the __parent_track field.
//

bool
track_field_base
::exists() const
{
  LOG_ERROR( main_logger, "exists() called on abstract field base?");
  return false;
}

void
track_field_base
::remove_at_row( const oracle_entry_handle_type& /*row*/ )
{
  throw runtime_error("remove_row called on abstract field base");
}

void
track_field_base
::set_host( track_field_host* h )
{
  this->host = h;
}

} // ...track_oracle
} // ...kwiver
