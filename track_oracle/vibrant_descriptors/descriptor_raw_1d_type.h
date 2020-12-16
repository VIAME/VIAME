// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_DESCRIPTOR_RAW_1D_TYPE_H
#define INCL_DESCRIPTOR_RAW_1D_TYPE_H

#include <vital/vital_config.h>
#include <track_oracle/vibrant_descriptors/vibrant_descriptors_export.h>

#include <iostream>
#include <vector>

#include <vnl/vnl_vector.h>

namespace kwiver {
namespace track_oracle {

struct VIBRANT_DESCRIPTORS_EXPORT descriptor_raw_1d_type
{
  std::vector< double > data;

  descriptor_raw_1d_type() {}
  explicit descriptor_raw_1d_type( const std::vector<double>& d ): data(d) {}
  explicit descriptor_raw_1d_type( const vnl_vector<double>& d );
  bool operator==( const descriptor_raw_1d_type& rhs ) const
  {
    if ( this->data.size() != rhs.data.size() ) return false;
    for (size_t i=0; i<this->data.size(); ++i)
    {
      if (this->data[i] != rhs.data[i]) return false;
    }
    return true;
  }
};

VIBRANT_DESCRIPTORS_EXPORT std::ostream& operator<<( std::ostream& os, const descriptor_raw_1d_type& d );
VIBRANT_DESCRIPTORS_EXPORT std::istream& operator>>( std::istream& os, descriptor_raw_1d_type& d );

} // ...track_oracle
} // ...kwiver

#endif
