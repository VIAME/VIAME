// This file is part of KWIVER, and is distributed under the
// OSI-approved BSD 3-Clause License. See top-level LICENSE file or
// https://github.com/Kitware/kwiver/blob/master/LICENSE for details.

#ifndef INCL_FILE_FORMAT_SCHEMA_H
#define INCL_FILE_FORMAT_SCHEMA_H

#include <vital/vital_config.h>
#include <track_oracle/file_formats/track_oracle_format_base_export.h>

#include <vector>
#include <string>
#include <track_oracle/core/track_oracle_api_types.h>
#include <track_oracle/core/track_base.h>
#include <track_oracle/file_formats/file_format_type.h>

namespace kwiver {
namespace track_oracle {

struct file_format_schema_impl;

class TRACK_ORACLE_FORMAT_BASE_EXPORT
file_format_schema_type: public track_base< file_format_schema_type >
{
public:
  // Use this schema to determine the file and format
  // used when a track handle was created

  track_field< file_format_enum >& format;
  track_field< unsigned >& source_file_id;
  file_format_schema_type()
    : format( Track.add_field< file_format_enum >( "track_format" )),
      source_file_id( Track.add_field< unsigned >( "track_source_file_id" ))
  {}

  // utility function called by readers to introduce filenames
  // into the map of track source filenames
  static void record_track_source( const track_handle_list_type& tracks,
                                   const std::string& src_fn,
                                   file_format_enum fmt );

  // filename, or "" if not found
  static std::string source_id_to_filename( unsigned id );

  // schema aspect helper: id, or SOURCE_FILE_NOT_FOUND
  static unsigned source_filename_to_id( const std::string& fn );

  static const unsigned SOURCE_FILE_NOT_FOUND;

private:
  static file_format_schema_impl& get_instance();
  static file_format_schema_impl* impl;

};

} // ...track_oracle
} // ...kwiver

#endif
