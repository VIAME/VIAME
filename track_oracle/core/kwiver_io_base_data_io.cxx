/*ckwg +5
 * Copyright 2014-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "kwiver_io_base_data_io.h"

using std::ostream;
using std::istream;
using std::set;
using std::string;
using std::pair;

namespace kwiver {
namespace track_oracle {


ostream& operator<<(ostream& os, const set< string >&  )
{
  // todo
  return os;
}

istream& operator>>(istream& is, set< string >& )
{
  // todo
  return is;
}

ostream& operator<<(ostream& os, const pair<unsigned, unsigned >& )
{
  // todo
  return os;
}

istream& operator>>(istream& is, pair<unsigned, unsigned>& )
{
  // todo
  return is;
}

} // ...track_oracle
} // ...kwiver
