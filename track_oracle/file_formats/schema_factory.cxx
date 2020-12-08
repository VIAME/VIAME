// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "schema_factory.h"

#include <track_oracle/core/track_base_impl.h>
#include <track_oracle/file_formats/file_format_manager.h>

using std::pair;
using std::string;

namespace kwiver {
namespace track_oracle {

namespace schema_factory {

bool
clone_field_into_schema( track_base_impl& schema,
                         const string& name )
{
  // tickle the file formats before we query for the name
  file_format_manager::initialize();

  field_handle_type fh = track_oracle_core::lookup_by_name( name );
  if ( fh == INVALID_FIELD_HANDLE ) return false;

  element_descriptor e = track_oracle_core::get_element_descriptor( fh );
  pair< track_field_base*, track_base_impl::schema_position_type > f =
    file_format_manager::clone_field_from_element( e );

  if ( ! f.first ) return false;

  return schema.add_field_at_position( f );
}

} // ...schema_factory
} // ...track_oracle
} // ...kwiver
