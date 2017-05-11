/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#ifndef INCL_TRACK_FIELD_IO_PROXY_H
#define INCL_TRACK_FIELD_IO_PROXY_H

///
/// This is a trampoline class to facilitate I/O from
/// track_fields. (Currently, just output.)
///
/// This class exists because we want to have consistent output of
/// track_field data via the kwiver_io_base, but operator() on
/// track_fields returns an instance of the track_field's type, rather
/// than a track_field with access to the kwiver_io_base.
///
///

#include <iostream>
#include <track_oracle/core/kwiver_io_base.h>

namespace kwiver {
namespace track_oracle {

template <typename T>
class track_field_io_proxy
{
  template <typename Tio > friend std::ostream& operator<<( std::ostream& os, const track_field_io_proxy<Tio>& iop );
public:
  track_field_io_proxy( kwiver_io_base<T>* p, const T& v ):
    io_ptr(p), val(v)
  {}

private:
  kwiver_io_base<T>* io_ptr;
  T val;
};

template < typename Tio > std::ostream& operator<<( std::ostream& os, const track_field_io_proxy<Tio>& iop );

} // ...track_oracle
} // ...kwiver

#endif
