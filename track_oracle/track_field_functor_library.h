/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_FUNCTOR_LIBRARY_H
#define INCL_TRACK_FIELD_FUNCTOR_LIBRARY_H

#include <vital/vital_config.h>
#include <track_oracle/track_oracle_export.h>

#include <track_oracle/track_oracle_core.h>
#include <track_oracle/track_field_functor.h>

namespace kwiver {
namespace track_oracle {

class TRACK_ORACLE_EXPORT max_unsigned_functor: public track_field_functor<unsigned >
{
private:
  size_t c;
  unsigned max;

public:
  max_unsigned_functor()
    : track_field_functor<unsigned>(), c(0), max(0) {}
  virtual void apply_at_row( const oracle_entry_handle_type& row, const unsigned& i )
  {
    if ((c == 0) || ( i > max ))
    {
      this->result_handle = row;
      this->result_value = i;
      max = i;
    }
    ++c;
  }
};

} // ...track_oracle
} // ...kwiver

#endif
