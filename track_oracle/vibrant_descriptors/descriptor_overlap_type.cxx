// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "descriptor_overlap_type.h"

#include <iostream>

#include <track_oracle/aries_interface/aries_interface.h>

using std::string;
using std::ostream;
using std::istream;
using std::ios;
using std::map;

namespace kwiver {
namespace track_oracle {

ostream&
operator<<( ostream& os, const descriptor_overlap_type& d )
{
  const map< size_t, string >& i2a = aries_interface::index_to_activity_map();
  map< size_t, string >::const_iterator probe = i2a.find( d.src_activity_id );
  string src_activity_name = (probe == i2a.end()) ? "BAD_INDEX" : probe->second;
  probe = i2a.find( d.dst_activity_id );
  string dst_activity_name = (probe == i2a.end()) ? "BAD_INDEX" : probe->second;

  os << "<descriptor type=\"overlap\">\n"
     << "  <src_trk_id value =\"" << d.src_trk_id << "\"/>\n"
     << "  <dst_trk_id value =\"" << d.dst_trk_id << "\"/>\n"
     << "  <src_activity_id value =\"" << d.src_activity_id << "\"/>\n"
     << "  <src_activity_name value =\"" << src_activity_name<< "\"/>\n"
     << "  <dst_activity_id value =\"" << d.dst_activity_id << "\"/>\n"
     << "  <dst_activity_name value =\"" << dst_activity_name << "\"/>\n"
     << "  <n_frames_src value =\"" << d.n_frames_src << "\"/>\n"
     << "  <n_frames_dst value =\"" << d.n_frames_dst << "\"/>\n"
     << "  <n_frames_overlap value =\"" << d.n_frames_overlap << "\"/>\n"
     << "  <mean_centroid_distance value =\"" << d.mean_centroid_distance << "\"/>\n"
     << "  <radial_overlap_flag value =\"" << static_cast<unsigned>( d.radial_overlap_flag ) << "\"/>\n"
     << "  <mean_percentage_overlap value =\"" << d.mean_percentage_overlap << "\"/>\n"
     << "</descriptor>\n";
  return os;
}

istream&
operator>>( istream& is, descriptor_overlap_type& /* d */ )
{
  return is;
}

} // ...track_oracle
} // ...kwiver
