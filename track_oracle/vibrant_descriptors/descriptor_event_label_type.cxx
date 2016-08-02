/*ckwg +5
 * Copyright 2013-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

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
