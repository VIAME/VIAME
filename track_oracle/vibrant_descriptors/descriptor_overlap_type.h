// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_DESCRIPTOR_OVERLAP_H
#define INCL_DESCRIPTOR_OVERLAP_H

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <ostream>

namespace kwiver {
namespace track_oracle {

struct VIBRANT_DESCRIPTORS_EXPORT descriptor_overlap_type
{
  unsigned src_trk_id;             // external ID of "source" (usually GT)
  unsigned dst_trk_id;             // external ID of "dest" (usually computed)
  unsigned src_activity_id;        // activity associated with source track (or "NotScored" if none)
  unsigned dst_activity_id;        // activity associated with dst track (or "NotScored")
  unsigned n_frames_src;           // number of frames in src track
  unsigned n_frames_dst;           // number of frames in dst track
  unsigned n_frames_overlap;       // number of frames in overlap
  double mean_centroid_distance;   // mean of centroid distances in overlap
  bool radial_overlap_flag;        // true: radial overlap (no boxes); false: spatial overlap (boxes)
  double mean_percentage_overlap;  // mean of overlap_area / box_union_area (only if ! radial_overlap_flag)
  descriptor_overlap_type():
    src_trk_id( static_cast<unsigned>( -1 )),
    dst_trk_id( static_cast<unsigned>( -1 )),
    src_activity_id(0),
    dst_activity_id(0),
    n_frames_src(0),
    n_frames_dst(0),
    n_frames_overlap(0),
    mean_centroid_distance(0),
    radial_overlap_flag( false ),
    mean_percentage_overlap(0)
  {}
  bool operator==( const descriptor_overlap_type& rhs) const
  {
    return
      ( this->src_trk_id == rhs.src_trk_id ) &&
      ( this->dst_trk_id == rhs.dst_trk_id ) &&
      ( this->src_activity_id == rhs.src_activity_id ) &&
      ( this->dst_activity_id == rhs.dst_activity_id ) &&
      ( this->n_frames_src == rhs.n_frames_src ) &&
      ( this->n_frames_dst == rhs.n_frames_dst ) &&
      ( this->n_frames_overlap == rhs.n_frames_overlap ) &&
      ( this->mean_centroid_distance == rhs.mean_centroid_distance ) &&
      ( this->radial_overlap_flag == rhs.radial_overlap_flag ) &&
      ( this->mean_percentage_overlap == rhs.mean_percentage_overlap );
  }
};

VIBRANT_DESCRIPTORS_EXPORT std::ostream& operator<<( std::ostream& os, const descriptor_overlap_type& d );
VIBRANT_DESCRIPTORS_EXPORT std::istream& operator>>( std::istream& is, descriptor_overlap_type& );

} // ...track_oracle
} // ...kwiver

#endif
