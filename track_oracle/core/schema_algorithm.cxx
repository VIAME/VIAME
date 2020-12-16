// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#include "schema_algorithm.h"

#include <map>
#include <track_oracle/core/element_descriptor.h>
#include <track_oracle/core/track_base_impl.h>

using std::map;
using std::vector;

namespace kwiver {
namespace track_oracle {
namespace schema_algorithm {

vector< element_descriptor >
name_missing_fields( const track_base_impl& required_fields,
                     const track_handle_list_type tracks )
{
  map< field_handle_type, bool > missing_fields_all_tracks;
  for (size_t i=0; i<tracks.size(); ++i)
  {
    vector< field_handle_type > missing_fields_this_track =
      required_fields.list_missing_elements( tracks[i] );
    for (size_t j=0; j<missing_fields_this_track.size(); ++j)
    {
      missing_fields_all_tracks[ missing_fields_this_track[j] ] = true;
    }
  }

  vector< element_descriptor > ret;
  for (map< field_handle_type, bool >::const_iterator i = missing_fields_all_tracks.begin();
       i != missing_fields_all_tracks.end();
       ++i)
  {
    ret.push_back( track_oracle_core::get_element_descriptor( i->first ));
  }
  return ret;
}

// what elements in the first (reference) schema are missing in the second (candidate) schema?
vector< element_descriptor >
schema_compare( const track_base_impl& ref,
                const track_base_impl& candidate )
{
  vector< element_descriptor > ret;
  typedef map< field_handle_type, track_base_impl::schema_position_type > schema_map_type;
  typedef map< field_handle_type, track_base_impl::schema_position_type >::const_iterator schema_map_cit;

  schema_map_type ref_map = ref.list_schema_elements();
  schema_map_type candidate_map = candidate.list_schema_elements();

  for (schema_map_cit i = ref_map.begin(); i != ref_map.end(); ++i)
  {
    if ( candidate_map.find( i->first ) == candidate_map.end() )
    {
      ret.push_back( track_oracle_core::get_element_descriptor( i->first ) );
    }
  }
  return ret;
}

} // ...schema_algorithm
} // ...track_oracle
} // ...kwiver
