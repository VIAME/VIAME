/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "track_field_io_proxy.h"

namespace kwiver {
namespace track_oracle {

template< typename T >
std::ostream& operator<<( std::ostream& os, const track_field_io_proxy<T>& iop )
{
  return iop.io_ptr->to_stream( os, iop.val );
}

} // ...track_oracle
} // ...kwiver
