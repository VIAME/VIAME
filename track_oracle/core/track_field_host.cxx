// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "track_field_host.h"

namespace kwiver {
namespace track_oracle {

track_field_host
::track_field_host()
  : cursor( INVALID_ROW_HANDLE )
{
}

track_field_host
::~track_field_host()
{
}

oracle_entry_handle_type
track_field_host
::get_cursor() const
{
  return this->cursor;
}

void
track_field_host
::set_cursor( oracle_entry_handle_type h ) const
{
  this->cursor = h;
}

} // ...track_oracle
} // ...kwiver
