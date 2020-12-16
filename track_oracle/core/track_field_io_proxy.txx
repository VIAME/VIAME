// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
