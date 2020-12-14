// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_DESCRIPTOR_CUTIC_H
#define INCL_DESCRIPTOR_CUTIC_H

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <vector>
#include <iostream>

namespace kwiver {
namespace track_oracle {

struct VIBRANT_DESCRIPTORS_EXPORT descriptor_cutic_type
{
  //object level classifiers..
  //      (size=number of active models)
  std::vector< double >           score_class;
  std::vector< int >              score_type;

  //temporal continuity
  //      (size= usually 3, within interval, between last interval, overall track)
  std::vector< double >           sim_temporal;

  //raw descriptors (exclusively used in distance computation)
  //      (size=non-zero BOW), index=vector index, raw=value for that word index
  //              *NOTE* assumption of unique,increasing index values in vector
  std::vector< short >            desc_index;
  std::vector< double >           desc_raw;

  bool operator==(const descriptor_cutic_type& a) const
  {
    return (this->score_class == a.score_class) &&
      (this->score_type == a.score_type) &&
      (this->sim_temporal == a.sim_temporal) &&
      (this->desc_index == a.desc_index) &&
      (this->desc_raw == a.desc_raw);
  }

};

VIBRANT_DESCRIPTORS_EXPORT std::ostream& operator<<( std::ostream& os, const descriptor_cutic_type& );
VIBRANT_DESCRIPTORS_EXPORT std::istream& operator>>( std::istream& is, descriptor_cutic_type& );

} // ...track_oracle
} // ...kwiver

#endif /* INCL_DESCRIPTOR_CUTIC_H */
