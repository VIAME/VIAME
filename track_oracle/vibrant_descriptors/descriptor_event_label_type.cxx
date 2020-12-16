// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "descriptor_event_label_type.h"

#include <ostream>

using std::ostream;
using std::istream;
using std::ios;

namespace kwiver {
namespace track_oracle {

ostream&
operator<<( ostream& os, const descriptor_event_label_type& d )
{
  os << "  <labels domain=\"" << d.domain << "\">\n";
  for (size_t i=0; i<d.labels.size(); ++i)
  {
    const single_event_label_type& s = d.labels[i];
    os << "    <event type=\"" << s.activity_name
       << "\" spatialOverlap=\"" << s.spatial_overlap
       << "\" temporalOverlap=\"" << s.temporal_overlap
       << "\" />\n";
  }
  os << "  </labels>\n";
  return os;
}

istream&
operator>>( istream& is, descriptor_event_label_type& /* d */ )
{
  // TODO
  is.setstate( ios::failbit );
  return is;
}

} // ...track_oracle
} // ...kwiver
