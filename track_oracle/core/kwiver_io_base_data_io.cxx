// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

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
