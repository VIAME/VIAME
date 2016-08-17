/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_FUNCTOR_H
#define INCL_TRACK_FIELD_FUNCTOR_H

#include <vital/vital_config.h>
#include <track_oracle/track_oracle_export.h>

#include <track_oracle/track_oracle_api_types.h>

namespace kwiver {
namespace track_oracle {

template< typename T >
class TRACK_ORACLE_EXPORT track_field_functor
{
public:

  oracle_entry_handle_type result_handle;
  T result_value;

  track_field_functor():
    result_handle( INVALID_ROW_HANDLE )
  {}

  virtual ~track_field_functor() {}

  virtual void apply_at_row( const oracle_entry_handle_type& row, const T& value ) = 0;

};

} // ...track_oracle
} // ...kwiver

#endif
