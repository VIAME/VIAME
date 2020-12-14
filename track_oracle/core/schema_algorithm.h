// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_SCHEMA_ALGORITHM_H
#define INCL_SCHEMA_ALGORITHM_H

//
// Various operations on schemas.
//

#include <vital/vital_config.h>
#include <track_oracle/core/track_oracle_export.h>

#include <string>
#include <vector>
#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/element_descriptor.h>

namespace kwiver {
namespace track_oracle {

class track_base_impl;

namespace schema_algorithm {

std::vector< element_descriptor > TRACK_ORACLE_EXPORT
name_missing_fields( const track_base_impl& required_fields,
                     const track_handle_list_type tracks );

// what elements in the first (reference) schema are missing in the second (candidate) schema?
std::vector< element_descriptor > TRACK_ORACLE_EXPORT
schema_compare( const track_base_impl& ref,
                const track_base_impl& candidate );

} // ...schema_algorithm
} // ...track_oracle
} // ...kwiver

#endif
