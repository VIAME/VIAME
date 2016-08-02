/*ckwg +5
 * Copyright 2010-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

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
