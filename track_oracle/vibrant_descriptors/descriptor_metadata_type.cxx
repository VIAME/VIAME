/*ckwg +5
 * Copyright 2011-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "descriptor_metadata_type.h"

using std::ostream;
using std::istream;
using std::ios;

namespace kwiver {
namespace track_oracle {

ostream&
operator<<( ostream& os, const descriptor_metadata_type& )
{
  return os;
}

istream&
operator>>( istream& is, descriptor_metadata_type& )
{
  return is;
}

} // ...track_oracle
} // ...kwiver
