/*ckwg +5
 * Copyright 2012-2016 by Kitware, Inc. All Rights Reserved. Please refer to
 * KITWARE_LICENSE.TXT for licensing information, or contact General Counsel,
 * Kitware, Inc., 28 Corporate Drive, Clifton Park, NY 12065.
 */

#include "file_format_schema.h"

using std::map;
using std::string;

const unsigned kwiver::track_oracle::file_format_schema_type::SOURCE_FILE_NOT_FOUND = 0;
kwiver::track_oracle::file_format_schema_impl* kwiver::track_oracle::file_format_schema_type::impl = 0;

namespace kwiver {
namespace track_oracle {

typedef map< string, unsigned > src_fn_to_id_map_type;
typedef map< string, unsigned >::const_iterator src_fn_to_id_map_cit;
typedef map< unsigned, string > id_to_src_fn_map_type;
typedef map< unsigned, string >::const_iterator id_to_src_fn_map_cit;

struct file_format_schema_impl
{
  src_fn_to_id_map_type src_fn_to_id;  // all the files we've loaded
  id_to_src_fn_map_type id_to_src_fn;  // backmap of ids to filename
};

file_format_schema_impl&
file_format_schema_type
::get_instance()
{
  if ( ! file_format_schema_type::impl )
  {
    file_format_schema_type::impl = new file_format_schema_impl();
  }
  return *file_format_schema_type::impl;
}

string
file_format_schema_type
::source_id_to_filename( unsigned id )
{
  const id_to_src_fn_map_type& m = get_instance().id_to_src_fn;
  id_to_src_fn_map_cit probe = m.find( id );
  return
    (probe == m.end())
    ? ""
    : probe->second;
}


unsigned
file_format_schema_type
::source_filename_to_id( const string& fn )
{
  const src_fn_to_id_map_type& m = get_instance().src_fn_to_id;
  src_fn_to_id_map_cit probe = m.find( fn );
  return
    (probe == m.end())
    ? SOURCE_FILE_NOT_FOUND
    : probe->second;
}

void
file_format_schema_type
::record_track_source( const track_handle_list_type& tracks,
                       const string& src_fn,
                       file_format_enum fmt )
{
  unsigned id = SOURCE_FILE_NOT_FOUND;

  // lookup or create an id for this source filename
  src_fn_to_id_map_type& m = get_instance().src_fn_to_id;
  map< string, unsigned >::iterator probe = m.find( src_fn );
  if (probe == m.end())
  {
    id = static_cast<unsigned>(m.size()) + 1; // first ID is 1
    m[ src_fn ] = id;
    id_to_src_fn_map_type& m2 = get_instance().id_to_src_fn;
    m2[ id ] = src_fn;
  }
  else
  {
    id = probe->second;
  }

  file_format_schema_type ffs;
  for (size_t i=0; i<tracks.size(); ++i)
  {
    ffs( tracks[i] ).format() = fmt;
    ffs( tracks[i] ).source_file_id() = id;
  }
}

} // ...track_oracle
} // ...kwiver
